"""Punar service API -- Razorpay ``payment.failed`` receiver and read surface.

Security posture
----------------
* Webhook signature verification **fails closed**. With no signing secret the
  service refuses to start, unless ``PUNAR_ALLOW_UNVERIFIED_WEBHOOKS=1`` is set
  for local development -- which then warns on every single request.
* The read endpoints require an API key. Message bodies and customer
  identifiers are only returned to a key holding the ``pii`` scope.
* Configuration is validated at startup: an unloadable policy or unwritable
  database stops the process rather than silently dropping payments while
  reporting ``200 OK``.

Honesty
-------
Nothing here sends a real message. Every response and every audit record
carries the provider provenance from :mod:`punar.api.providers`, so a
simulated outcome is never presented as a delivered one.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import random
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from punar.api.config import Settings, get_settings
from punar.api.jobs import JobQueue
from punar.api.logging_setup import configure_logging, get_logger, new_request_id
from punar.api.providers import OutboundMessage, ProviderRegistry
from punar.audit import AuditStore, open_store_from_settings
from punar.core.agent import run_agent, runner_name
from punar.core.bandit_store import BanditStore
from punar.core.classify import enrich
from punar.core.gate import load_policy, next_contact_window
from punar.core.select import read_posteriors, set_bandit_store

logger = get_logger("punar.api")

RAZORPAY_FAILED_EVENT = "payment.failed"


# --------------------------------------------------------------- request models
class RazorpayError(BaseModel):
    """The error block of a Razorpay payment entity (all fields optional)."""

    code: str | None = None
    description: str | None = None
    reason: str | None = None
    source: str | None = None
    step: str | None = None


class RazorpayPaymentEntity(BaseModel):
    """A Razorpay payment entity, accepting both the flat and nested shapes."""

    id: str | None = None
    amount: int = Field(default=0, ge=0, le=10_000_000_000)
    currency: str = Field(default="INR", max_length=8)
    method: str = Field(default="", max_length=32)
    email: str | None = Field(default=None, max_length=320)
    contact: str | None = Field(default=None, max_length=32)
    customer_id: str | None = Field(default=None, max_length=64)
    order_id: str | None = Field(default=None, max_length=64)
    error_code: str | None = Field(default=None, max_length=128)
    error_description: str | None = Field(default=None, max_length=512)
    error_reason: str | None = Field(default=None, max_length=128)
    error_source: str | None = Field(default=None, max_length=64)
    error_step: str | None = Field(default=None, max_length=64)
    error: RazorpayError | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class _PaymentPayload(BaseModel):
    entity: RazorpayPaymentEntity = Field(default_factory=RazorpayPaymentEntity)
    model_config = {"extra": "allow"}


class _Payload(BaseModel):
    payment: _PaymentPayload | None = None
    model_config = {"extra": "allow"}


class WebhookBody(BaseModel):
    """A Razorpay webhook envelope. Unknown top-level keys are tolerated."""

    event: str = Field(default="", max_length=128)
    account_id: str | None = Field(default=None, max_length=64)
    created_at: int | None = None
    payload: _Payload = Field(default_factory=_Payload)
    model_config = {"extra": "allow"}

    def payment_entity(self) -> RazorpayPaymentEntity:
        if self.payload.payment is None:
            return RazorpayPaymentEntity()
        return self.payload.payment.entity


class WebhookAccepted(BaseModel):
    accepted: bool
    queued: bool
    duplicate: bool = False
    case_id: str | None = None
    job_id: int | None = None
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, Any]
    runner: str
    providers: dict[str, Any]
    version: str


# ------------------------------------------------------------------ app state
class AppState:
    """Everything the request handlers need, created once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.policy: dict[str, Any] = {}
        # Populated by open() before the app serves a request; the lifespan
        # refuses to start if either fails, so handlers can rely on them.
        self._audit: AuditStore | None = None
        self._jobs: JobQueue | None = None
        self.bandit: BanditStore | None = None
        self.providers = ProviderRegistry.from_settings(settings)
        self.started_at = datetime.now(UTC)
        self.worker: asyncio.Task | None = None
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self.metrics: dict[str, float] = {
            "requests": 0, "errors": 0, "webhooks": 0, "duplicates": 0,
            "recovered": 0, "written_off": 0, "blocked_by_judge": 0,
            "latency_ms_total": 0.0,
        }

    @property
    def audit(self) -> AuditStore:
        if self._audit is None:
            raise RuntimeError("audit store not opened")
        return self._audit

    @property
    def jobs(self) -> JobQueue:
        if self._jobs is None:
            raise RuntimeError("job queue not opened")
        return self._jobs

    # -- lifecycle
    def open(self) -> None:
        self.policy = load_policy(self.settings.policy_path)
        self._audit = open_store_from_settings(self.settings)
        self._jobs = JobQueue(self.settings.jobs_db_path,
                              max_attempts=self.settings.job_max_attempts,
                              lease_seconds=self.settings.job_lease_seconds)
        self.bandit = BanditStore(self.settings.bandit_db_path)
        set_bandit_store(self.bandit)
        recovered = self.jobs.recover_stale()
        if recovered:
            logger.warning("recovered %s stale job(s) from a previous run", recovered)

    def close(self) -> None:
        set_bandit_store(None)
        for resource in (self._audit, self._jobs):
            try:
                if resource is not None:
                    resource.close()
            except Exception:                     # pragma: no cover - shutdown path
                logger.exception("error closing a resource during shutdown")

    # -- rate limiting (per-process buckets; a shared store in production)
    def allow_request(self, key: str, limit_per_minute: int) -> bool:
        if limit_per_minute <= 0:
            return True
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit_per_minute:
            return False
        window.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.require_valid()
    configure_logging(settings.log_level, json_output=settings.log_json)
    state = AppState(settings)
    state.open()
    app.state.punar = state
    if settings.worker_enabled:
        state.worker = asyncio.create_task(_worker_loop(state))
    logger.info("punar api started", extra={"config": settings.redacted()})
    try:
        yield
    finally:
        if state.worker is not None:
            state.worker.cancel()
            try:
                await state.worker
            except BaseException:                  # pragma: no cover - shutdown
                pass
        state.close()


app = FastAPI(title="Punar -- AI Revenue Recovery", version="1.0.0", lifespan=lifespan)


def _state(request: Request) -> AppState:
    return request.app.state.punar


# ------------------------------------------------------------------ middleware
@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign a request id, enforce rate limits, add security headers."""
    started = time.perf_counter()
    request_id = new_request_id()
    state: AppState | None = getattr(request.app.state, "punar", None)

    if state is not None:
        state.metrics["requests"] += 1
        client = request.client.host if request.client else "unknown"
        is_webhook = request.url.path.startswith("/webhooks/")
        limit = (state.settings.webhook_rate_limit_per_minute if is_webhook
                 else state.settings.rate_limit_per_minute)
        if not state.allow_request(f"{client}:{is_webhook}", limit):
            return JSONResponse(
                {"detail": "rate limit exceeded", "request_id": request_id},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "60", "X-Request-ID": request_id})

    try:
        response = await call_next(request)
    except Exception:
        if state is not None:
            state.metrics["errors"] += 1
        # The detail stays server-side; the caller gets an opaque error plus an id.
        logger.exception("unhandled error", extra={"request_id": request_id})
        response = JSONResponse({"detail": "internal error", "request_id": request_id},
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    if state is not None:
        state.metrics["latency_ms_total"] += (time.perf_counter() - started) * 1000.0
        if state.settings.hsts_enabled:
            response.headers["Strict-Transport-Security"] = \
                "max-age=31536000; includeSubDomains"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


_CORS_ORIGINS = get_settings().cors_allow_origins
if _CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=list(_CORS_ORIGINS),
                       allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization"])


# ------------------------------------------------------------------ auth
def _presented_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return (x_api_key or "").strip() or None


def require_api_key(request: Request,
                    authorization: str | None = Header(default=None),
                    x_api_key: str | None = Header(default=None)) -> str | None:
    """Authenticate a read request and return the presented key."""
    settings = _state(request).settings
    if settings.allow_unauthenticated_reads and not settings.api_keys:
        return None
    key = _presented_key(authorization, x_api_key)
    if not key or not any(hmac.compare_digest(key, known) for known in settings.api_keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="valid API key required",
                            headers={"WWW-Authenticate": "Bearer"})
    return key


def verify_signature(raw_body: bytes, signature: str | None,
                     settings: Settings) -> bool:
    """Verify Razorpay's HMAC-SHA256 webhook signature. Fails CLOSED."""
    if not settings.webhook_secret:
        if settings.allow_unverified_webhooks:
            logger.warning(
                "ACCEPTING UNVERIFIED WEBHOOK: no signing secret is configured. "
                "This is a development-only mode and must never run in production.")
            return True
        return False
    if not signature:
        return False
    digest = hmac.new(settings.webhook_secret.encode(), raw_body,
                      hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature.strip(), digest)


# ------------------------------------------------------------------ processing
def normalize_case(entity: RazorpayPaymentEntity, state: AppState) -> dict[str, Any]:
    """Turn a Razorpay payment entity into a Punar case.

    Consent is resolved through the configured lookup rather than assumed:
    hard-coding ``opted_out=False`` here is exactly what would make the
    opt-out guarantee unenforceable in the service path.
    """
    err = entity.error
    code = entity.error_code or (err.code if err else None) or ""
    description = entity.error_description or (err.description if err else None) or ""
    reason_text = entity.error_reason or (err.reason if err else None) or ""

    profile = state.providers.profiles.fetch(entity.customer_id)
    opted_out = state.providers.consent.is_opted_out(entity.customer_id)
    landing = next_contact_window(datetime.now(UTC), state.policy)

    case = {
        "case_id": entity.id or f"pay_{int(time.time() * 1000)}",
        "customer_id": entity.customer_id,
        "amount_inr": round(entity.amount / 100.0, 2),
        "currency": entity.currency,
        "method": entity.method,
        "merchant_name": str(entity.notes.get("merchant_name") or "your merchant"),
        "language": (profile.language if profile else "en"),
        "opted_out": opted_out,
        "day_of_month": landing.day,
        "hour": landing.hour,
        "error": {"code": code, "description": description, "reason": reason_text},
        "error_code": code,
        "error_description": description,
        "error_reason": reason_text,
        "touches": [],
        "active": True,
    }
    return enrich(case, state.policy)


def _dispatch(state: AppState, case: dict[str, Any], result: dict[str, Any]) -> None:
    """Hand every customer-facing touch to the provider and record the receipt."""
    for touch in result.get("touch_history", []):
        if not touch.get("contacts_customer"):
            continue
        receipt = state.providers.sender.send(OutboundMessage(
            case_id=str(case.get("case_id")),
            customer_id=case.get("customer_id"),
            channel=str(touch.get("channel") or ""),
            intervention=str(touch.get("intervention") or ""),
            body=str(touch.get("copy") or ""),
            language=str(case.get("language") or "en"),
            payment_link=case.get("payment_link")))
        touch["delivered"] = receipt.delivered
        touch["delivery_status"] = receipt.status
        touch["delivery_provider"] = receipt.provider
        touch["simulated"] = receipt.simulated


def process_case(state: AppState, case: dict[str, Any]) -> dict[str, Any]:
    """Run the recovery agent for one case and append the audit record."""
    seed = int(hashlib.sha256(str(case.get("case_id")).encode()).hexdigest()[:8], 16)
    observer = state.providers.outcomes

    def simulate(current: dict[str, Any], intervention: str, now: datetime) -> bool:
        return bool(observer.observe(current, intervention, now))

    result = run_agent(case, state.policy, random.Random(seed), simulate)
    _dispatch(state, case, result)

    record = {
        "case_id": case.get("case_id"),
        "customer_id": case.get("customer_id"),
        "amount_inr": case.get("amount_inr"),
        "reason": (result.get("diagnosis") or {}).get("reason"),
        "outcome": result.get("outcome"),
        "exit_code": result.get("exit_code"),
        "touch_history": result.get("touch_history", []),
        "blocked_actions": result.get("blocked_actions", []),
        "escalations": result.get("escalations", []),
        "audit": result.get("audit", []),
        "plan_records": result.get("plan_records", []),
        "arm_log": result.get("arm_log", []),
        "providers": state.providers.describe(),
        "outcome_is_simulated": bool(getattr(observer, "simulated", True)),
    }
    if state.audit is not None:
        state.audit.append(record)

    if result.get("outcome") == "recovered":
        state.metrics["recovered"] += 1
    else:
        state.metrics["written_off"] += 1
    state.metrics["blocked_by_judge"] += len(result.get("blocked_actions", []))
    return record


async def _worker_loop(state: AppState) -> None:
    """Drain the durable job queue. Survives restarts; retries to a dead letter."""
    while True:
        try:
            job = await asyncio.to_thread(state.jobs.claim)
            if job is None:
                await asyncio.sleep(state.settings.worker_poll_seconds)
                continue
            try:
                record = await asyncio.to_thread(process_case, state, job.payload["case"])
                state.jobs.complete(job.id, {"outcome": record.get("outcome"),
                                             "exit_code": record.get("exit_code")})
            except Exception as exc:
                status_after = state.jobs.fail(job.id, repr(exc))
                logger.exception("job %s failed (now %s)", job.id, status_after)
        except asyncio.CancelledError:             # pragma: no cover - shutdown
            raise
        except Exception:                          # pragma: no cover - defensive
            logger.exception("worker loop error")
            await asyncio.sleep(1.0)


# ------------------------------------------------------------------ endpoints
def _health_payload(state: AppState) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    ok = True

    checks["policy"] = {"ok": bool(state.policy), "path": state.settings.policy_path}
    ok = ok and checks["policy"]["ok"]

    try:
        chain = state.audit.verify_chain(limit=64)
        checks["audit"] = {"ok": chain.ok, "rows_checked": chain.rows_checked}
        ok = ok and chain.ok
    except Exception as exc:
        checks["audit"] = {"ok": False, "error": type(exc).__name__}
        ok = False

    try:
        checks["jobs"] = {"ok": state.jobs.healthy(), **state.jobs.stats()}
        ok = ok and checks["jobs"]["ok"]
    except Exception as exc:
        checks["jobs"] = {"ok": False, "error": type(exc).__name__}
        ok = False

    signature_ok = state.settings.signature_required() or not state.settings.is_production
    checks["signature_verification"] = {
        "required": state.settings.signature_required(),
        "ok": signature_ok,
    }
    ok = ok and signature_ok

    return {
        "status": "ok" if ok else "degraded",
        "checks": checks,
        "runner": runner_name(),
        "providers": state.providers.describe(),
        "version": app.version,
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, Any]:
    """Real dependency check -- policy loadable, audit chain intact, queue live."""
    return _health_payload(_state(request))


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: can this instance accept traffic right now?"""
    payload = _health_payload(_state(request))
    code = (status.HTTP_200_OK if payload["status"] == "ok"
            else status.HTTP_503_SERVICE_UNAVAILABLE)
    return JSONResponse(payload, status_code=code)


@app.post("/webhooks/razorpay", response_model=WebhookAccepted,
          status_code=status.HTTP_202_ACCEPTED)
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(default=None),
    x_razorpay_event_id: str | None = Header(default=None),
) -> WebhookAccepted:
    """Accept a ``payment.failed`` event and durably queue one recovery run."""
    state = _state(request)
    settings = state.settings

    raw = await request.body()
    if len(raw) > settings.max_body_bytes:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail=f"body exceeds {settings.max_body_bytes} bytes")

    if not verify_signature(raw, x_razorpay_signature, settings):
        logger.warning("rejected a webhook with an invalid or missing signature")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="invalid webhook signature")

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="body is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="webhook body must be a JSON object")
    try:
        body = WebhookBody.model_validate(parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=json.loads(exc.json())) from exc

    state.metrics["webhooks"] += 1
    if body.event != RAZORPAY_FAILED_EVENT:
        return WebhookAccepted(accepted=True, queued=False,
                               detail=f"event '{body.event}' is not handled")

    case = normalize_case(body.payment_entity(), state)

    # Razorpay retries webhooks; the same event must never drive a second
    # recovery run, or the touch caps mean nothing.
    idempotency_key = x_razorpay_event_id or hashlib.sha256(raw).hexdigest()
    job, created = await asyncio.to_thread(
        state.jobs.enqueue, idempotency_key, str(case["case_id"]), {"case": case})
    if not created:
        state.metrics["duplicates"] += 1
        return WebhookAccepted(accepted=True, queued=False, duplicate=True,
                               case_id=case["case_id"], job_id=job.id,
                               detail="duplicate event; the original run is retained")

    return WebhookAccepted(accepted=True, queued=True, case_id=case["case_id"],
                           job_id=job.id, detail="queued for recovery")


@app.get("/cases/{case_id}")
def get_case(case_id: str, request: Request,
             api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """Latest audit revision for a case. PII requires a pii-scoped key."""
    state = _state(request)
    reveal = state.settings.key_has_pii_scope(api_key)
    record = state.audit.get_latest(case_id, reveal=reveal)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
    record["pii_revealed"] = reveal
    return record


@app.get("/cases/{case_id}/history")
def get_case_history(case_id: str, request: Request,
                     api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """Every revision of a case, oldest first -- the append-only trail itself."""
    state = _state(request)
    reveal = state.settings.key_has_pii_scope(api_key)
    history = state.audit.get_history(case_id, reveal=reveal)
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
    return {"case_id": case_id, "revisions": len(history), "history": history}


@app.get("/jobs/{job_id}")
def get_job(job_id: int, request: Request,
            api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """Status of one queued recovery run, including dead-letter state."""
    job = _state(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return job.to_public_dict()


@app.get("/audit/verify")
def verify_audit(request: Request,
                 api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """Verify the audit hash chain end to end. This is the compliance check."""
    return _state(request).audit.verify_chain().to_dict()


@app.get("/bandit/posteriors")
def bandit_posteriors(request: Request,
                      api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """What the agent has actually learned, per (reason, intervention)."""
    state = _state(request)
    rows = read_posteriors(state.policy)
    return {"count": len(rows), "posteriors": rows}


@app.get("/stats")
def stats(request: Request,
          api_key: str | None = Depends(require_api_key)) -> dict[str, Any]:
    """Business counters, with explicit provenance for simulated outcomes."""
    state = _state(request)
    total = state.metrics["recovered"] + state.metrics["written_off"]
    return {
        "cases_processed": total,
        "recovered": state.metrics["recovered"],
        "written_off": state.metrics["written_off"],
        "recovery_rate": round(state.metrics["recovered"] / total, 4) if total else None,
        "blocked_by_policy_judge": state.metrics["blocked_by_judge"],
        "duplicate_webhooks": state.metrics["duplicates"],
        "jobs": state.jobs.stats(),
        "audit_rows": state.audit.count_rows(),
        "providers": state.providers.describe(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> str:
    """Prometheus exposition format."""
    state = _state(request)
    counters = state.metrics
    total = counters["recovered"] + counters["written_off"]
    uptime = (datetime.now(UTC) - state.started_at).total_seconds()
    lines = [
        "# HELP punar_requests_total HTTP requests served.",
        "# TYPE punar_requests_total counter",
        f"punar_requests_total {counters['requests']:.0f}",
        "# HELP punar_errors_total Unhandled errors.",
        "# TYPE punar_errors_total counter",
        f"punar_errors_total {counters['errors']:.0f}",
        "# HELP punar_webhooks_total Razorpay webhooks accepted.",
        "# TYPE punar_webhooks_total counter",
        f"punar_webhooks_total {counters['webhooks']:.0f}",
        "# HELP punar_duplicate_webhooks_total Deduplicated webhook redeliveries.",
        "# TYPE punar_duplicate_webhooks_total counter",
        f"punar_duplicate_webhooks_total {counters['duplicates']:.0f}",
        "# HELP punar_cases_total Cases run to a terminal state.",
        "# TYPE punar_cases_total counter",
        f"punar_cases_total {total:.0f}",
        "# HELP punar_recovered_total Cases recovered.",
        "# TYPE punar_recovered_total counter",
        f"punar_recovered_total {counters['recovered']:.0f}",
        "# HELP punar_blocked_by_policy_judge_total Messages blocked pre-send.",
        "# TYPE punar_blocked_by_policy_judge_total counter",
        f"punar_blocked_by_policy_judge_total {counters['blocked_by_judge']:.0f}",
        "# HELP punar_request_latency_ms_total Cumulative request latency.",
        "# TYPE punar_request_latency_ms_total counter",
        f"punar_request_latency_ms_total {counters['latency_ms_total']:.3f}",
        "# HELP punar_jobs Current job-queue depth by status.",
        "# TYPE punar_jobs gauge",
    ]
    for key, value in state.jobs.stats().items():
        lines.append(f'punar_jobs{{status="{key}"}} {value}')
    lines += [
        "# HELP punar_uptime_seconds Process uptime.",
        "# TYPE punar_uptime_seconds gauge",
        f"punar_uptime_seconds {uptime:.1f}",
    ]
    return "\n".join(lines) + "\n"
