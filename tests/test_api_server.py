"""Service-layer tests: authentication, validation, idempotency, honesty.

These assert the security posture rather than the handler internals. Each one
corresponds to a defect found in review: an endpoint that accepted unsigned
webhooks, an unauthenticated PII dump, 500s from attacker-controlled fields,
duplicate webhook deliveries driving duplicate customer contact, and a
``/health`` that reported ``ok`` while every request was failing.
"""
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

SECRET = "test_webhook_secret"
API_KEY = "test-key-readonly-0123456789"
PII_KEY = "test-key-pii-scope-0123456789"


def _payload(payment_id="pay_test_1", amount=250000, code="INSUFFICIENT_FUNDS"):
    return {
        "event": "payment.failed",
        "payload": {"payment": {"entity": {
            "id": payment_id,
            "amount": amount,
            "currency": "INR",
            "method": "upi",
            "customer_id": "cust_test_1",
            "error_code": "BAD_REQUEST_ERROR",
            "error_description": "insufficient balance in the account",
            "error_reason": code,
            "notes": {"merchant_name": "Test Merchant"},
        }}},
    }


def _sign(body: bytes) -> str:
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _post(client, payload, *, sign=True, event_id=None):
    raw = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if sign:
        headers["X-Razorpay-Signature"] = _sign(raw)
    if event_id:
        headers["X-Razorpay-Event-Id"] = event_id
    return client.post("/webhooks/razorpay", content=raw, headers=headers)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A fully configured app instance backed by throwaway databases."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("PUNAR_API_KEYS", f"{API_KEY},{PII_KEY}")
    monkeypatch.setenv("PUNAR_PII_API_KEYS", PII_KEY)
    monkeypatch.setenv("PUNAR_DB_PATH", str(tmp_path / "audit.db"))
    monkeypatch.setenv("PUNAR_JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("PUNAR_BANDIT_DB", str(tmp_path / "bandit.db"))
    monkeypatch.setenv("PUNAR_WORKER_ENABLED", "0")     # drive the queue by hand
    monkeypatch.setenv("PUNAR_LOG_JSON", "0")
    monkeypatch.setenv("PUNAR_ALLOW_UNVERIFIED_WEBHOOKS", "0")

    from punar.api import config, server
    config.reset_settings_cache()
    with TestClient(server.app) as c:
        yield c
    config.reset_settings_cache()


def _drain(client):
    """Run every queued job synchronously."""
    from punar.api.server import process_case
    state = client.app.state.punar
    processed = []
    while True:
        job = state.jobs.claim()
        if job is None:
            return processed
        record = process_case(state, job.payload["case"])
        state.jobs.complete(job.id, {"outcome": record.get("outcome")})
        processed.append(record)


# ----------------------------------------------------------------- signatures
def test_unsigned_webhook_is_rejected(client):
    """It used to fail OPEN: no secret configured meant every payload accepted."""
    assert _post(client, _payload(), sign=False).status_code == 401


def test_forged_signature_is_rejected(client):
    raw = json.dumps(_payload()).encode()
    response = client.post("/webhooks/razorpay", content=raw,
                           headers={"X-Razorpay-Signature": "deadbeef"})
    assert response.status_code == 401


def test_correctly_signed_webhook_is_accepted(client):
    response = _post(client, _payload())
    assert response.status_code == 202
    assert response.json()["queued"] is True


def test_signature_verification_fails_closed_without_a_secret(monkeypatch):
    """No secret and no explicit dev override -> reject, never accept."""
    from punar.api.config import Settings
    from punar.api.server import verify_signature
    settings = Settings(webhook_secret="", allow_unverified_webhooks=False)
    assert verify_signature(b"{}", None, settings) is False
    assert verify_signature(b"{}", "anything", settings) is False


# ---------------------------------------------------------------------- authn
def test_reads_require_an_api_key(client):
    assert client.get("/cases/pay_test_1").status_code == 401
    assert client.get("/stats").status_code == 401
    assert client.get("/audit/verify").status_code == 401


def test_read_with_a_valid_key_succeeds(client):
    _post(client, _payload())
    _drain(client)
    response = client.get("/cases/pay_test_1", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == 200
    assert response.json()["case_id"] == "pay_test_1"


def test_pii_is_withheld_from_a_key_without_the_scope(client):
    _post(client, _payload())
    _drain(client)
    plain = client.get("/cases/pay_test_1",
                       headers={"Authorization": f"Bearer {API_KEY}"}).json()
    scoped = client.get("/cases/pay_test_1",
                        headers={"Authorization": f"Bearer {PII_KEY}"}).json()
    assert plain["pii_revealed"] is False
    assert scoped["pii_revealed"] is True
    assert plain.get("customer_id") != "cust_test_1", "raw customer id must not leak"


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize("body", [b"[1,2,3]", b'"hello"', b"123", b"null"])
def test_non_object_json_is_a_400_not_a_500(client, body):
    """`WebhookBody(**payload)` used to raise TypeError -> 500 on these."""
    response = client.post("/webhooks/razorpay", content=body,
                           headers={"X-Razorpay-Signature": _sign(body)})
    assert response.status_code == 400


def test_malformed_json_is_a_400(client):
    body = b"{not json"
    response = client.post("/webhooks/razorpay", content=body,
                           headers={"X-Razorpay-Signature": _sign(body)})
    assert response.status_code == 400


def test_string_amount_is_a_400_not_a_500(client):
    """A string `amount` used to reach `/ 100` and raise TypeError -> 500."""
    payload = _payload()
    payload["payload"]["payment"]["entity"]["amount"] = "abc"
    assert _post(client, payload).status_code == 400


def test_oversized_body_is_rejected(client):
    payload = _payload()
    payload["padding"] = "x" * (300 * 1024)
    assert _post(client, payload).status_code == 413


def test_unhandled_event_is_acknowledged_but_not_queued(client):
    payload = _payload()
    payload["event"] = "payment.captured"
    body = _post(client, payload).json()
    assert body["accepted"] is True and body["queued"] is False


# ---------------------------------------------------------------- idempotency
def test_redelivered_webhook_does_not_queue_a_second_run(client):
    """Razorpay retries webhooks. A retry must not double-contact the customer."""
    first = _post(client, _payload(), event_id="evt_1").json()
    second = _post(client, _payload(), event_id="evt_1").json()
    third = _post(client, _payload(), event_id="evt_1").json()

    assert first["queued"] is True and first["duplicate"] is False
    assert second["duplicate"] is True and second["queued"] is False
    assert third["duplicate"] is True
    assert second["job_id"] == first["job_id"]
    assert len(_drain(client)) == 1


# --------------------------------------------------------------------- health
def test_health_checks_real_dependencies(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["policy"]["ok"] is True
    assert body["checks"]["audit"]["ok"] is True
    assert body["checks"]["jobs"]["ok"] is True


def test_health_reports_degraded_when_the_audit_chain_is_broken(client):
    _post(client, _payload())
    _drain(client)
    state = client.app.state.punar
    raw = state.audit._conn
    raw.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
    raw.execute("UPDATE audit_log SET data = ? WHERE id = 1",
                (json.dumps({"case_id": "pay_test_1", "outcome": "TAMPERED"}),))
    raw.commit()

    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/ready").status_code == 503


def test_metrics_are_prometheus_format(client):
    text = client.get("/metrics").text
    assert "# TYPE punar_requests_total counter" in text
    assert "punar_uptime_seconds" in text


# -------------------------------------------------------------------- honesty
def test_simulated_delivery_is_labelled_as_such(client):
    """`delivered: true` for a message nothing ever sent is the claim to avoid."""
    _post(client, _payload())
    records = _drain(client)
    record = records[0]
    assert record["outcome_is_simulated"] is True
    assert record["providers"]["message_delivery"]["simulated"] is True
    for touch in record["touch_history"]:
        if touch.get("contacts_customer"):
            assert touch["delivered"] is False
            assert touch["simulated"] is True


def test_opt_out_is_resolved_through_the_consent_provider(client, monkeypatch):
    """`opted_out` used to be hardcoded False, so the guarantee had no input."""
    state = client.app.state.punar
    from punar.api.providers import StubConsentLookup
    state.providers.consent = StubConsentLookup(opted_out=["cust_optout"])

    payload = _payload(payment_id="pay_optout")
    payload["payload"]["payment"]["entity"]["customer_id"] = "cust_optout"
    _post(client, payload, event_id="evt_optout")
    record = _drain(client)[0]

    contacting = [t for t in record["touch_history"] if t.get("contacts_customer")]
    assert contacting == [], "an opted-out customer must never be contacted"


def test_audit_trail_is_verifiable_over_the_api(client):
    _post(client, _payload())
    _drain(client)
    body = client.get("/audit/verify",
                      headers={"Authorization": f"Bearer {API_KEY}"}).json()
    assert body["ok"] is True
    assert body["rows_checked"] >= 1


def test_case_history_exposes_every_revision(client):
    _post(client, _payload(), event_id="evt_a")
    _drain(client)
    state = client.app.state.punar
    state.audit.append({"case_id": "pay_test_1", "outcome": "manual_review"})

    body = client.get("/cases/pay_test_1/history",
                      headers={"Authorization": f"Bearer {API_KEY}"}).json()
    assert body["revisions"] == 2
