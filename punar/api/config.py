"""Runtime configuration for the Punar API service.

Every tunable is read from the environment through a frozen ``Settings``
object, so that:

* secrets are never captured in module-level constants (they can be rotated by
  restarting the process, or by building a new ``Settings``),
* configuration is validated at STARTUP and the service refuses to start when
  it is invalid (rather than accepting webhooks it can never process),
* file paths resolve relative to the installed *package*, not the process CWD.

Fail-closed by default: a missing ``RAZORPAY_WEBHOOK_SECRET`` or a missing API
key set is a fatal configuration error unless an explicit, loudly-logged
development escape hatch is set.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "Settings",
    "ConfigError",
    "packaged_policy_path",
    "default_data_dir",
    "get_settings",
    "reset_settings_cache",
]

# Env-var names kept in one place so docs/tests cannot drift from the code.
ENV_WEBHOOK_SECRET = "RAZORPAY_WEBHOOK_SECRET"
ENV_ALLOW_UNVERIFIED = "PUNAR_ALLOW_UNVERIFIED_WEBHOOKS"
ENV_ALLOW_ANON_READS = "PUNAR_ALLOW_UNAUTHENTICATED_READS"
ENV_API_KEYS = "PUNAR_API_KEYS"
ENV_PII_API_KEYS = "PUNAR_PII_API_KEYS"

_TRUE = {"1", "true", "yes", "on"}


class ConfigError(RuntimeError):
    """Raised when the service is asked to start with invalid configuration."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_tuple(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


_POLICY_CACHE: str | None = None


def packaged_policy_path() -> str:
    """Absolute path to the policy shipped *inside* the installed package.

    Uses ``importlib.resources`` so an installed wheel resolves correctly from
    any working directory. Falls back to materialising the resource into a
    temp file for the (unusual) zip-import case.
    """
    global _POLICY_CACHE
    if _POLICY_CACHE and os.path.exists(_POLICY_CACHE):
        return _POLICY_CACHE
    ref = resources.files("punar").joinpath("config", "policy.json")
    candidate = str(ref)
    if os.path.isfile(candidate):
        _POLICY_CACHE = os.path.abspath(candidate)
        return _POLICY_CACHE
    # zip-safe fallback: extract once for the lifetime of the process.
    tmp_dir = Path(tempfile.gettempdir()) / "punar-config"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / "policy.json"
    target.write_bytes(ref.read_bytes())
    _POLICY_CACHE = str(target)
    return _POLICY_CACHE


def default_data_dir() -> str:
    """Writable directory for the audit + job databases.

    Never the package directory (installed packages are read-only) and never a
    CWD-relative path (that silently forks state per launch directory).
    """
    override = os.getenv("PUNAR_DATA_DIR")
    if override:
        return os.path.abspath(override)
    return str(Path.home() / ".punar")


@dataclass(frozen=True)
class Settings:
    """Immutable, validated view of the service configuration."""

    environment: str = "development"

    # --- security -----------------------------------------------------------
    webhook_secret: str = ""
    allow_unverified_webhooks: bool = False
    api_keys: tuple[str, ...] = ()
    pii_api_keys: tuple[str, ...] = ()
    allow_unauthenticated_reads: bool = False
    cors_allow_origins: tuple[str, ...] = ()
    max_body_bytes: int = 256 * 1024
    rate_limit_per_minute: int = 120
    webhook_rate_limit_per_minute: int = 600
    hsts_enabled: bool = True

    # --- storage ------------------------------------------------------------
    data_dir: str = ""
    policy_path: str = ""
    audit_db_path: str = ""
    jobs_db_path: str = ""
    bandit_db_path: str = ""

    # --- audit / compliance -------------------------------------------------
    audit_pii_mode: str = "hash"          # hash | encrypt | none
    audit_encryption_key: str = ""
    audit_pseudonym_key: str = ""
    audit_retention_days: int = 2555      # ~7 years, RBI-friendly default

    # --- work queue ---------------------------------------------------------
    job_max_attempts: int = 3
    job_lease_seconds: int = 120
    worker_enabled: bool = True
    worker_poll_seconds: float = 0.25

    # --- observability ------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    # --- providers ----------------------------------------------------------
    stub_opted_out_customers: tuple[str, ...] = ()
    outcome_simulation_rate: float = 0.6

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = default_data_dir()
        return cls(
            environment=os.getenv("PUNAR_ENV", "development").strip().lower(),
            webhook_secret=os.getenv(ENV_WEBHOOK_SECRET, "").strip(),
            allow_unverified_webhooks=_env_bool(ENV_ALLOW_UNVERIFIED),
            api_keys=_env_tuple(ENV_API_KEYS),
            pii_api_keys=_env_tuple(ENV_PII_API_KEYS),
            allow_unauthenticated_reads=_env_bool(ENV_ALLOW_ANON_READS),
            cors_allow_origins=_env_tuple("PUNAR_CORS_ALLOW_ORIGINS"),
            max_body_bytes=_env_int("PUNAR_MAX_BODY_BYTES", 256 * 1024),
            rate_limit_per_minute=_env_int("PUNAR_RATE_LIMIT_PER_MINUTE", 120),
            webhook_rate_limit_per_minute=_env_int(
                "PUNAR_WEBHOOK_RATE_LIMIT_PER_MINUTE", 600),
            hsts_enabled=_env_bool("PUNAR_HSTS_ENABLED", True),
            data_dir=data_dir,
            policy_path=os.getenv("PUNAR_POLICY_PATH") or packaged_policy_path(),
            audit_db_path=(os.getenv("PUNAR_DB_PATH")
                           or os.path.join(data_dir, "punar_audit.db")),
            jobs_db_path=(os.getenv("PUNAR_JOBS_DB_PATH")
                          or os.path.join(data_dir, "punar_jobs.db")),
            bandit_db_path=(os.getenv("PUNAR_BANDIT_DB")
                            or os.path.join(data_dir, "punar_bandit.db")),
            audit_pii_mode=os.getenv("PUNAR_AUDIT_PII_MODE", "hash").strip().lower(),
            audit_encryption_key=os.getenv("PUNAR_AUDIT_ENCRYPTION_KEY", ""),
            audit_pseudonym_key=os.getenv("PUNAR_AUDIT_PSEUDONYM_KEY", ""),
            audit_retention_days=_env_int("PUNAR_AUDIT_RETENTION_DAYS", 2555),
            job_max_attempts=_env_int("PUNAR_JOB_MAX_ATTEMPTS", 3),
            job_lease_seconds=_env_int("PUNAR_JOB_LEASE_SECONDS", 120),
            worker_enabled=_env_bool("PUNAR_WORKER_ENABLED", True),
            worker_poll_seconds=_env_float("PUNAR_WORKER_POLL_SECONDS", 0.25),
            log_level=os.getenv("PUNAR_LOG_LEVEL", "INFO").upper(),
            log_json=_env_bool("PUNAR_LOG_JSON", True),
            stub_opted_out_customers=_env_tuple("PUNAR_STUB_OPTED_OUT_CUSTOMERS"),
            outcome_simulation_rate=_env_float("PUNAR_OUTCOME_SIMULATION_RATE", 0.6),
        )

    # ----------------------------------------------------------------- checks
    @property
    def is_production(self) -> bool:
        return self.environment in ("production", "prod")

    def signature_required(self) -> bool:
        """True unless the dev escape hatch is explicitly enabled."""
        return not (self.allow_unverified_webhooks and self.webhook_secret == "")

    def key_has_pii_scope(self, api_key: str | None) -> bool:
        return bool(api_key) and api_key in self.pii_api_keys

    def validate(self) -> list[str]:
        """Return a list of fatal configuration problems (empty == valid)."""
        problems: list[str] = []

        if not self.webhook_secret and not self.allow_unverified_webhooks:
            problems.append(
                f"{ENV_WEBHOOK_SECRET} is not set. Webhook signature verification "
                f"fails CLOSED. Set the secret, or set {ENV_ALLOW_UNVERIFIED}=1 for "
                "local development only.")
        if self.allow_unverified_webhooks and self.is_production:
            problems.append(
                f"{ENV_ALLOW_UNVERIFIED} must never be enabled when PUNAR_ENV=production.")
        if not self.api_keys and not self.allow_unauthenticated_reads:
            problems.append(
                f"{ENV_API_KEYS} is empty, so /cases and /metrics have no "
                f"authentication. Set at least one key, or set "
                f"{ENV_ALLOW_ANON_READS}=1 for local development only.")
        if self.allow_unauthenticated_reads and self.is_production:
            problems.append(
                f"{ENV_ALLOW_ANON_READS} must never be enabled when PUNAR_ENV=production.")
        if any(len(key) < 16 for key in self.api_keys):
            problems.append(
                f"every entry in {ENV_API_KEYS} must be at least 16 characters")
        if any(k not in self.api_keys for k in self.pii_api_keys):
            problems.append(
                f"{ENV_PII_API_KEYS} contains keys that are not in {ENV_API_KEYS}")
        if self.audit_pii_mode not in ("hash", "encrypt", "none"):
            problems.append("PUNAR_AUDIT_PII_MODE must be one of: hash, encrypt, none")
        if self.audit_pii_mode == "encrypt" and not self.audit_encryption_key:
            problems.append(
                "PUNAR_AUDIT_PII_MODE=encrypt requires PUNAR_AUDIT_ENCRYPTION_KEY")
        if self.audit_pii_mode == "none" and self.is_production:
            problems.append("PUNAR_AUDIT_PII_MODE=none is not permitted in production")
        if not 0 < self.max_body_bytes <= 8 * 1024 * 1024:
            problems.append("PUNAR_MAX_BODY_BYTES must be between 1 and 8388608")
        if self.audit_retention_days < 1:
            problems.append("PUNAR_AUDIT_RETENTION_DAYS must be >= 1")
        if self.job_max_attempts < 1:
            problems.append("PUNAR_JOB_MAX_ATTEMPTS must be >= 1")
        if not 0.0 <= self.outcome_simulation_rate <= 1.0:
            problems.append("PUNAR_OUTCOME_SIMULATION_RATE must be between 0 and 1")

        problems.extend(self.validate_policy())
        problems.extend(self._validate_paths())
        return problems

    def validate_policy(self) -> list[str]:
        """Load and sanity-check the guardrail policy document."""
        try:
            with open(self.policy_path, encoding="utf-8") as fh:
                policy = json.load(fh)
        except FileNotFoundError:
            return [f"policy file not found: {self.policy_path}"]
        except (OSError, ValueError) as exc:
            return [f"policy file unreadable/invalid at {self.policy_path}: {exc}"]
        if not isinstance(policy, dict):
            return [f"policy file {self.policy_path} must contain a JSON object"]

        problems: list[str] = []
        guardrails = policy.get("guardrails")
        if not isinstance(guardrails, dict):
            return ["policy.guardrails is missing or not an object"]
        window = guardrails.get("contact_window", {})
        if not isinstance(window, dict):
            return ["policy.guardrails.contact_window must be an object"]
        try:
            start_h = int(window.get("start_hour", 8))
            end_h = int(window.get("end_hour", 19))
        except (TypeError, ValueError):
            return ["contact_window start_hour/end_hour must be integers"]
        if not 0 <= start_h <= 23:
            problems.append(f"contact_window.start_hour must be 0..23 (got {start_h})")
        if not 1 <= end_h <= 24:
            problems.append(f"contact_window.end_hour must be 1..24 (got {end_h})")
        if start_h >= end_h:
            problems.append(
                f"contact_window.start_hour ({start_h}) must be < end_hour ({end_h})")
        if not isinstance(policy.get("channels", {}), dict):
            problems.append("policy.channels must be an object")
        return problems

    def _validate_paths(self) -> list[str]:
        problems: list[str] = []
        for label, path in (("audit", self.audit_db_path), ("jobs", self.jobs_db_path),
                            ("bandit", self.bandit_db_path)):
            parent = os.path.dirname(os.path.abspath(path)) or "."
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                problems.append(
                    f"{label} database directory {parent} is not creatable: {exc}")
                continue
            if not os.access(parent, os.W_OK):
                problems.append(f"{label} database directory {parent} is not writable")
        return problems

    def require_valid(self) -> None:
        """Raise ``ConfigError`` listing every problem, or return silently."""
        problems = self.validate()
        if problems:
            bullets = "\n".join(f"  - {p}" for p in problems)
            raise ConfigError("Punar refuses to start: invalid configuration.\n" + bullets)

    def redacted(self) -> dict[str, Any]:
        """Loggable view with every secret removed."""
        return {
            "environment": self.environment,
            "webhook_secret_configured": bool(self.webhook_secret),
            "signature_required": self.signature_required(),
            "api_keys_configured": len(self.api_keys),
            "pii_scoped_keys": len(self.pii_api_keys),
            "allow_unauthenticated_reads": self.allow_unauthenticated_reads,
            "policy_path": self.policy_path,
            "audit_db_path": self.audit_db_path,
            "jobs_db_path": self.jobs_db_path,
            "bandit_db_path": self.bandit_db_path,
            "audit_pii_mode": self.audit_pii_mode,
            "audit_retention_days": self.audit_retention_days,
            "max_body_bytes": self.max_body_bytes,
            "worker_enabled": self.worker_enabled,
        }


_CACHED: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings. ``refresh=True`` re-reads the environment."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = Settings.from_env()
    return _CACHED


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests and by secret rotation)."""
    global _CACHED
    _CACHED = None
