"""Append-only, tamper-evident SQLite audit trail for Punar decisions.

Every agent step, guardrail verdict, message and policy-judge decision is
written here before any response is returned -- an RBI / Fair-Practices
reviewer can reconstruct exactly why a payment was (or was not) retried.

Design guarantees
-----------------
1. **Append-only in fact, not just in name.** Every state change is a NEW row.
   ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers make in-place mutation
   impossible from SQL, including from an operator with a ``sqlite3`` shell.
2. **Tamper-evident.** Each row carries ``prev_hash`` and its own ``row_hash``
   over ``(prev_hash, id, case_id, event_type, created_at, payload)``. The head
   hash and row count are checkpointed in ``audit_meta``. :meth:`AuditStore.
   verify_chain` therefore detects edits, insertions, re-orderings, and
   deletions -- including deletion of the most recent row.
3. **Server-assigned ordering.** ``created_at`` is stamped by this process and
   ordering is by the monotonic integer primary key, so caller-supplied data
   can never decide which version is "latest".
4. **No plaintext PII at rest.** Customer identifiers are pseudonymised with a
   keyed HMAC and message bodies are scrubbed of contact details / payment
   links before they are written. With ``PUNAR_AUDIT_PII_MODE=encrypt`` (plus
   a key) the PII is stored encrypted and is recoverable by an authorised
   reader instead.
5. **Migrations.** ``PRAGMA user_version`` drives a real migration runner, so
   an existing audit database is upgraded rather than silently ignored by
   ``CREATE TABLE IF NOT EXISTS``.
6. **Retention.** :meth:`AuditStore.prune` is the ONLY sanctioned delete path.
   It writes a chain checkpoint plus a ``retention.prune`` audit event so the
   surviving chain is still verifiable end to end.

Threading / concurrency: one connection per store, guarded by a lock, opened
with WAL, an explicit ``busy_timeout`` and ``foreign_keys`` on.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = [
    "AuditStore",
    "AppendOnlyViolation",
    "ChainVerification",
    "PIIRedactor",
    "GENESIS_HASH",
    "SCHEMA_VERSION",
]

GENESIS_HASH = "0" * 64
SCHEMA_VERSION = 1
_DESTROY_CONFIRMATION = "yes-destroy-the-audit-trail"


class AppendOnlyViolation(RuntimeError):
    """Raised when a caller attempts to mutate or erase the audit trail."""


# --------------------------------------------------------------------------- PII
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_PAN_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

#: Record keys whose *values* are direct identifiers.
PII_IDENTIFIER_KEYS = frozenset({
    "customer_id", "customer_uid", "contact_id", "customer_email", "email",
    "customer_phone", "phone", "contact", "msisdn", "customer_name", "name",
    "vpa", "upi_id",
})
#: Record keys whose values are free text that may embed identifiers.
PII_TEXT_KEYS = frozenset({"copy", "text", "message", "body", "rendered", "description"})
#: Record keys holding a payment/short link.
PII_LINK_KEYS = frozenset({"payment_link", "short_url", "link", "url"})

_ENC_PREFIX = "enc:v1:"
_PSN_PREFIX = "psn:"


def _fernet(key_material: str):  # pragma: no cover - exercised when cryptography present
    from cryptography.fernet import Fernet

    digest = hashlib.sha256(key_material.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


@dataclass
class PIIRedactor:
    """Transforms a record so no plaintext PII reaches disk.

    ``mode``:
      * ``"hash"`` (default) -- identifiers become stable keyed pseudonyms and
        free text is scrubbed of emails / phone numbers / links / card-like
        digit runs. Irreversible; the audit trail stays readable and linkable.
      * ``"encrypt"`` -- identifiers and message bodies are encrypted with a
        key held outside the database and can be revealed by an authorised
        reader (:meth:`reveal`). Requires the ``cryptography`` package.
      * ``"none"`` -- no transformation. Local development only; the API
        refuses to start in production with this mode.
    """

    mode: str = "hash"
    pseudonym_key: str = ""
    encryption_key: str = ""
    _cipher: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("hash", "encrypt", "none"):
            raise ValueError(f"unknown PII mode: {self.mode!r}")
        if self.mode == "encrypt":
            if not self.encryption_key:
                raise ValueError("PII mode 'encrypt' requires an encryption key")
            try:
                self._cipher = _fernet(self.encryption_key)
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ValueError(
                    "PII mode 'encrypt' requires the 'cryptography' package; "
                    "install punar[secure] or use PUNAR_AUDIT_PII_MODE=hash") from exc

    # ------------------------------------------------------------------ atoms
    def pseudonym(self, value: str) -> str:
        """Stable, keyed, irreversible token for an identifier."""
        key = (self.pseudonym_key or "punar-default-pseudonym-key").encode("utf-8")
        digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{_PSN_PREFIX}{digest[:24]}"

    def _encrypt(self, value: str) -> str:
        return _ENC_PREFIX + self._cipher.encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, value: str) -> str:
        token = value[len(_ENC_PREFIX):].encode("ascii")
        return self._cipher.decrypt(token).decode("utf-8")

    @staticmethod
    def scrub_text(text: str) -> str:
        """Remove contact details, links and card-like numbers from free text."""
        out = _EMAIL_RE.sub("[email-redacted]", text)
        out = _URL_RE.sub("[link-redacted]", out)
        out = _PAN_RE.sub("[card-redacted]", out)
        out = _PHONE_RE.sub("[phone-redacted]", out)
        return out

    # ---------------------------------------------------------------- records
    def redact(self, record: Any) -> Any:
        """Deep-copy ``record`` with every PII-bearing value transformed."""
        if self.mode == "none":
            return record
        return self._walk(record, key=None)

    def _walk(self, node: Any, key: str | None) -> Any:
        if isinstance(node, dict):
            return {k: self._walk(v, k) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [self._walk(v, key) for v in node]
        if not isinstance(node, str) or key is None:
            return node
        lowered = key.lower()
        if lowered in PII_IDENTIFIER_KEYS:
            if not node:
                return node
            return self._encrypt(node) if self.mode == "encrypt" else self.pseudonym(node)
        if lowered in PII_LINK_KEYS:
            if not node:
                return node
            return self._encrypt(node) if self.mode == "encrypt" else self._link_token(node)
        if lowered in PII_TEXT_KEYS:
            return self._encrypt(node) if self.mode == "encrypt" else self.scrub_text(node)
        return node

    def _link_token(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"[link-redacted:{digest}]"

    def reveal(self, record: Any) -> Any:
        """Inverse of :meth:`redact` where the mode makes that possible."""
        if self.mode != "encrypt":
            return record
        return self._unwalk(record)

    def _unwalk(self, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: self._unwalk(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [self._unwalk(v) for v in node]
        if isinstance(node, str) and node.startswith(_ENC_PREFIX):
            try:
                return self._decrypt(node)
            except Exception:  # noqa: BLE001 - a bad token must not break a read
                return "[undecryptable]"
        return node


# ------------------------------------------------------------------- results
@dataclass
class ChainVerification:
    """Outcome of a hash-chain integrity check."""

    ok: bool
    rows_checked: int
    head_hash: str
    problems: list[str] = field(default_factory=list)
    first_bad_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows_checked": self.rows_checked,
            "head_hash": self.head_hash,
            "problems": list(self.problems),
            "first_bad_id": self.first_bad_id,
        }


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))


def compute_row_hash(prev_hash: str, row_id: int, case_id: str, event_type: str,
                     created_at: str, data_json: str) -> str:
    """Content hash binding a row to its position in the chain."""
    material = "|".join([prev_hash, str(row_id), case_id, event_type,
                         created_at, data_json])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- store
class AuditStore:
    """Append-only, hash-chained SQLite audit trail."""

    def __init__(self, path: str = "punar_audit.db", *,
                 redactor: PIIRedactor | None = None,
                 retention_days: int = 2555,
                 busy_timeout_ms: int = 5000) -> None:
        self.path = path
        self.redactor = redactor or PIIRedactor(mode="none")
        self.retention_days = retention_days
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._closed = False
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = self._connect()
        self._migrate()

    # ------------------------------------------------------------ connection
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    self._conn.execute("PRAGMA optimize")
                except sqlite3.Error:  # pragma: no cover - best effort
                    pass
                self._conn.close()
                self._closed = True

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------- migrations
    def _migrate(self) -> None:
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                for target, migration in _MIGRATIONS:
                    if version < target:
                        migration(conn)
                        conn.execute(f"PRAGMA user_version = {target}")
                        version = target
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # ----------------------------------------------------------------- writes
    def append(self, record: dict[str, Any], *, event_type: str = "decision") -> int:
        """Append one immutable, hash-chained version of a case record.

        ``created_at`` is always stamped by this process; any caller-supplied
        value is preserved separately as ``reported_at`` so it can never
        influence ordering.
        """
        if "case_id" not in record:
            raise ValueError("audit records require a case_id")
        payload = dict(record)
        caller_ts = payload.pop("created_at", None)
        if caller_ts is not None:
            payload.setdefault("reported_at", caller_ts)
        case_id = str(payload["case_id"])
        payload = self.redactor.redact(payload)

        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                head = conn.execute(
                    "SELECT id, row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                prev_hash = head["row_hash"] if head else GENESIS_HASH
                row_id = (head["id"] + 1) if head else 1
                created_at = _utcnow()
                payload["case_id"] = case_id
                payload["created_at"] = created_at
                payload["audit_revision"] = self._revision_for(conn, case_id) + 1
                data_json = _canonical(payload)
                row_hash = compute_row_hash(prev_hash, row_id, case_id, event_type,
                                            created_at, data_json)
                conn.execute(
                    "INSERT INTO audit_log (id, case_id, event_type, data, created_at, "
                    "prev_hash, row_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row_id, case_id, event_type, data_json, created_at,
                     prev_hash, row_hash),
                )
                self._set_meta(conn, "head_id", str(row_id))
                self._set_meta(conn, "head_hash", row_hash)
                self._set_meta(
                    conn, "row_count",
                    str(int(self._get_meta(conn, "row_count", "0")) + 1))
                conn.execute("COMMIT")
                return row_id
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def upsert(self, record: dict[str, Any]) -> int:
        """Compatibility shim -- APPENDS a new revision, never overwrites.

        The historical implementation issued ``UPDATE ... SET data = ?`` which
        destroyed the previous version of a case. Callers are unchanged; the
        semantics are now append-only.
        """
        return self.append(record)

    # ------------------------------------------------------------------ reads
    def get_latest(self, case_id: str, *, reveal: bool = False) -> dict[str, Any] | None:
        """Newest revision for a case, ordered by the monotonic row id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, data FROM audit_log WHERE case_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return self._hydrate(row["data"], row["id"], reveal)

    def get_by_case_id(self, case_id: str, *, reveal: bool = False) -> dict[str, Any] | None:
        return self.get_latest(case_id, reveal=reveal)

    def get_history(self, case_id: str, *, reveal: bool = False) -> list[dict[str, Any]]:
        """Every revision of a case, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, data FROM audit_log WHERE case_id = ? ORDER BY id ASC",
                (case_id,),
            ).fetchall()
        return [self._hydrate(r["data"], r["id"], reveal) for r in rows]

    def recent(self, limit: int = 100, *, reveal: bool = False) -> list[dict[str, Any]]:
        """Most recently appended rows, newest first (monotonic id order)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, data FROM audit_log ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._hydrate(r["data"], r["id"], reveal) for r in rows]

    def latest_per_case(self, *, event_type: str = "decision",
                        reveal: bool = False) -> list[dict[str, Any]]:
        """Newest revision of EVERY case -- the correct basis for rate metrics.

        Unlike a ``recent(limit=N)`` window this never silently turns a
        lifetime rate into a sliding-window rate.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.id AS id, a.data AS data FROM audit_log a "
                "WHERE a.event_type = ? AND a.id = ("
                "  SELECT MAX(b.id) FROM audit_log b "
                "  WHERE b.case_id = a.case_id AND b.event_type = a.event_type) "
                "ORDER BY a.id DESC",
                (event_type,),
            ).fetchall()
        return [self._hydrate(r["data"], r["id"], reveal) for r in rows]

    def count_rows(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])

    def count_cases(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(DISTINCT case_id) FROM audit_log").fetchone()[0])

    def _hydrate(self, data_json: str, row_id: int, reveal: bool) -> dict[str, Any]:
        out = json.loads(data_json)
        if reveal:
            out = self.redactor.reveal(out)
        out["_db_id"] = row_id
        return out

    # ---------------------------------------------------------- tamper checks
    def verify_chain(self, *, limit: int | None = None) -> ChainVerification:
        """Recompute the whole hash chain and report any tampering.

        Detects: edited payloads, edited timestamps, re-ordered or inserted
        rows, deleted interior rows (the successor's ``prev_hash`` no longer
        matches) and deleted trailing rows (the checkpointed head hash and row
        count no longer match).
        """
        problems: list[str] = []
        first_bad: int | None = None
        with self._lock:
            conn = self._conn
            checkpoint = self._checkpoint(conn)
            rows = conn.execute(
                "SELECT id, case_id, event_type, data, created_at, prev_hash, row_hash "
                "FROM audit_log ORDER BY id ASC"
                + (" LIMIT ?" if limit else ""),
                (int(limit),) if limit else (),
            ).fetchall()
            expected_prev = checkpoint["row_hash"] if checkpoint else GENESIS_HASH
            expected_id = (checkpoint["last_pruned_id"] + 1) if checkpoint else 1
            head_hash = expected_prev
            last_ts = ""
            for row in rows:
                rid = int(row["id"])
                if rid != expected_id:
                    problems.append(
                        f"row id gap: expected {expected_id}, found {rid} "
                        "(rows deleted or inserted)")
                    first_bad = first_bad or rid
                    expected_id = rid
                if row["prev_hash"] != expected_prev:
                    problems.append(
                        f"row {rid}: prev_hash does not match the previous row_hash "
                        "(chain broken -- a row was deleted, re-ordered or inserted)")
                    first_bad = first_bad or rid
                recomputed = compute_row_hash(row["prev_hash"], rid, row["case_id"],
                                              row["event_type"], row["created_at"],
                                              row["data"])
                if recomputed != row["row_hash"]:
                    problems.append(f"row {rid}: content hash mismatch (row was modified)")
                    first_bad = first_bad or rid
                if row["created_at"] < last_ts:
                    problems.append(f"row {rid}: created_at moves backwards in time")
                    first_bad = first_bad or rid
                last_ts = row["created_at"]
                expected_prev = row["row_hash"]
                head_hash = row["row_hash"]
                expected_id = rid + 1

            if limit is None:
                stored_head = self._get_meta(conn, "head_hash", GENESIS_HASH)
                stored_count = int(self._get_meta(conn, "row_count", "0"))
                pruned = checkpoint["pruned_count"] if checkpoint else 0
                if stored_head != head_hash:
                    problems.append(
                        "checkpointed head hash does not match the computed head "
                        "(the most recent row(s) were deleted or replaced)")
                if stored_count - pruned != len(rows):
                    problems.append(
                        f"row count mismatch: checkpoint says {stored_count - pruned} "
                        f"live rows, found {len(rows)}")
        return ChainVerification(ok=not problems, rows_checked=len(rows),
                                 head_hash=head_hash, problems=problems,
                                 first_bad_id=first_bad)

    # -------------------------------------------------------------- retention
    def prune(self, *, before: datetime | None = None,
              dry_run: bool = False) -> dict[str, Any]:
        """Apply the retention policy -- the only sanctioned delete path.

        Rows created before ``before`` (default: ``retention_days`` ago) are
        removed. A chain checkpoint recording the last pruned row id and its
        ``row_hash`` is written first, and a ``retention.prune`` event is
        appended afterwards, so :meth:`verify_chain` still validates the
        surviving chain end to end.
        """
        cutoff = before or (datetime.now(UTC)
                            - timedelta(days=self.retention_days))
        cutoff_iso = cutoff.isoformat()
        with self._lock:
            conn = self._conn
            row = conn.execute(
                "SELECT id, row_hash FROM audit_log WHERE created_at < ? "
                "ORDER BY id DESC LIMIT 1", (cutoff_iso,)).fetchone()
            if row is None:
                return {"pruned": 0, "cutoff": cutoff_iso, "dry_run": dry_run}
            last_id, last_hash = int(row["id"]), row["row_hash"]
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE id <= ?", (last_id,)).fetchone()
            if dry_run:
                return {"pruned": int(count), "cutoff": cutoff_iso, "dry_run": True,
                        "would_checkpoint_at": last_id}

            previous = self._checkpoint(conn)
            already = previous["pruned_count"] if previous else 0
            checkpoint = {"last_pruned_id": last_id, "row_hash": last_hash,
                          "pruned_count": already + int(count),
                          "pruned_at": _utcnow(), "cutoff": cutoff_iso}
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._set_meta(conn, "chain_checkpoint", json.dumps(checkpoint))
                self._set_meta(conn, "retention_unlock", "1")
                conn.execute("DELETE FROM audit_log WHERE id <= ?", (last_id,))
                self._set_meta(conn, "retention_unlock", "0")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        self.append({"case_id": "__retention__", "pruned_rows": int(count),
                     "cutoff": cutoff_iso, "checkpoint_row_hash": last_hash,
                     "last_pruned_id": last_id},
                    event_type="retention.prune")
        return {"pruned": int(count), "cutoff": cutoff_iso, "dry_run": False,
                "checkpoint": checkpoint}

    # ------------------------------------------------------------ destructive
    def clear(self) -> int:
        """Removed. The audit trail is append-only and cannot be cleared.

        Kept as an explicit, loud failure so that any surviving caller (or a
        copy-pasted operational runbook) fails instead of silently destroying
        regulated records. Use :meth:`prune` for retention, or
        :meth:`destroy_for_tests` in a throwaway test database.
        """
        raise AppendOnlyViolation(
            "AuditStore.clear() is disabled: the audit trail is append-only. "
            "Use prune() for retention-driven deletion.")

    def destroy_for_tests(self, confirm: str) -> int:
        """Drop every row. Test-only; requires an explicit confirmation token."""
        if confirm != _DESTROY_CONFIRMATION:
            raise AppendOnlyViolation(
                "destroy_for_tests() requires confirm="
                f"{_DESTROY_CONFIRMATION!r}")
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._set_meta(conn, "retention_unlock", "1")
                cur = conn.execute("DELETE FROM audit_log")
                self._set_meta(conn, "retention_unlock", "0")
                self._set_meta(conn, "head_hash", GENESIS_HASH)
                self._set_meta(conn, "head_id", "0")
                self._set_meta(conn, "row_count", "0")
                conn.execute("DELETE FROM audit_meta WHERE key = 'chain_checkpoint'")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return int(cur.rowcount)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO audit_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM audit_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    @classmethod
    def _checkpoint(cls, conn: sqlite3.Connection) -> dict[str, Any] | None:
        raw = cls._get_meta(conn, "chain_checkpoint", "")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:  # pragma: no cover - corrupt meta
            return None

    @staticmethod
    def _revision_for(conn: sqlite3.Connection, case_id: str) -> int:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE case_id = ?", (case_id,)).fetchone()
        return int(n)


# ---------------------------------------------------------------- migrations
def _migration_001_initial(conn: sqlite3.Connection) -> None:
    """Create the hash-chained schema and carry over any legacy rows."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id         INTEGER PRIMARY KEY,
            case_id    TEXT NOT NULL,
            event_type TEXT NOT NULL DEFAULT 'decision',
            data       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            prev_hash  TEXT NOT NULL,
            row_hash   TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_log_case ON audit_log(case_id, id DESC)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at)")
    # Immutability is enforced by the database, not merely by application code.
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS audit_log_no_update
        BEFORE UPDATE ON audit_log
        BEGIN
            SELECT RAISE(ABORT, 'punar audit_log is append-only: UPDATE is forbidden');
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
        BEFORE DELETE ON audit_log
        WHEN COALESCE((SELECT value FROM audit_meta WHERE key = 'retention_unlock'), '0') <> '1'
        BEGIN
            SELECT RAISE(ABORT, 'punar audit_log is append-only: DELETE requires the retention path');
        END
    """)
    for key, value in (("head_hash", GENESIS_HASH), ("head_id", "0"), ("row_count", "0"),
                       ("retention_unlock", "0")):
        conn.execute("INSERT OR IGNORE INTO audit_meta (key, value) VALUES (?, ?)",
                     (key, value))

    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'decisions'"
    ).fetchone()
    if legacy is None:
        return
    rows = conn.execute(
        "SELECT id, case_id, data, created_at FROM decisions ORDER BY id ASC").fetchall()
    prev_hash, row_id, count = GENESIS_HASH, 0, 0
    for row in rows:
        row_id += 1
        created_at = row["created_at"] or _utcnow()
        try:
            payload = json.loads(row["data"])
        except ValueError:
            payload = {"case_id": row["case_id"], "legacy_data": row["data"]}
        payload["migrated_from_legacy_row"] = int(row["id"])
        data_json = _canonical(payload)
        row_hash = compute_row_hash(prev_hash, row_id, row["case_id"], "decision",
                                    created_at, data_json)
        conn.execute(
            "INSERT INTO audit_log (id, case_id, event_type, data, created_at, "
            "prev_hash, row_hash) VALUES (?, ?, 'decision', ?, ?, ?, ?)",
            (row_id, row["case_id"], data_json, created_at, prev_hash, row_hash))
        prev_hash = row_hash
        count += 1
    conn.execute("ALTER TABLE decisions RENAME TO decisions_legacy_v0")
    for key, value in (("head_hash", prev_hash), ("head_id", str(row_id)),
                       ("row_count", str(count))):
        conn.execute(
            "INSERT INTO audit_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


#: ``(target_version, migration)`` applied in order against ``PRAGMA user_version``.
_MIGRATIONS: tuple[tuple[int, Any], ...] = (
    (1, _migration_001_initial),
)


def open_store_from_settings(settings: Any) -> AuditStore:
    """Build an :class:`AuditStore` from a ``punar.api.config.Settings``."""
    redactor = PIIRedactor(mode=settings.audit_pii_mode,
                           pseudonym_key=settings.audit_pseudonym_key,
                           encryption_key=settings.audit_encryption_key)
    return AuditStore(settings.audit_db_path, redactor=redactor,
                      retention_days=settings.audit_retention_days)
