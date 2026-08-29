"""Durable, idempotent work queue backed by SQLite.

The webhook handler previously dispatched work through FastAPI's in-process
``BackgroundTasks``. That loses every accepted-but-unfinished case on restart,
has no retry, no dead-letter path, and no deduplication -- so Razorpay's
webhook retries produced duplicate customer outreach.

This module provides an at-least-once queue with:

* **Idempotency at ingest.** ``idempotency_key`` is UNIQUE. A repeat delivery
  of the same Razorpay event is a no-op that returns the ORIGINAL job (and its
  original result), so retried webhooks never double-contact a customer.
* **Durability.** Queued intent is committed to disk before the 202 is
  returned. A restart re-claims the work instead of dropping it.
* **Leases.** A claim stamps ``claim_expires_at``; :meth:`JobQueue.recover_stale`
  returns expired claims to the queue, which is how a crash mid-processing is
  recovered.
* **Retry + dead letter.** Failures increment ``attempts``; once
  ``max_attempts`` is reached the job moves to ``dead`` and stays inspectable.

Production upgrade path: this is deliberately a *single-node* broker. Swap
:class:`JobQueue` for Redis Streams / SQS / Kafka behind the same
claim / complete / fail interface when you need multiple workers or
cross-host durability; the call sites do not change.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = ["Job", "JobQueue", "JobStatus"]

QUEUED = "queued"
CLAIMED = "claimed"
DONE = "done"
DEAD = "dead"
JobStatus = str

_SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


@dataclass
class Job:
    """One unit of recovery work."""

    id: int
    idempotency_key: str
    case_id: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    max_attempts: int
    created_at: str
    updated_at: str
    claim_expires_at: str | None = None
    result: dict[str, Any] | None = None
    last_error: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=int(row["id"]),
            idempotency_key=row["idempotency_key"],
            case_id=row["case_id"],
            payload=json.loads(row["payload"]),
            status=row["status"],
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            claim_expires_at=row["claim_expires_at"],
            result=json.loads(row["result"]) if row["result"] else None,
            last_error=row["last_error"],
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Operator-facing view; the raw payload is deliberately excluded."""
        return {
            "id": self.id,
            "case_id": self.case_id,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
        }


class JobQueue:
    """SQLite-backed at-least-once job queue with an idempotency index."""

    def __init__(self, path: str, *, max_attempts: int = 3,
                 lease_seconds: int = 120, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()
        self._closed = False
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    # ------------------------------------------------------------- lifecycle
    def _migrate(self) -> None:
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version < 1:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS jobs (
                            id               INTEGER PRIMARY KEY AUTOINCREMENT,
                            idempotency_key  TEXT NOT NULL UNIQUE,
                            case_id          TEXT NOT NULL,
                            payload          TEXT NOT NULL,
                            status           TEXT NOT NULL,
                            attempts         INTEGER NOT NULL DEFAULT 0,
                            max_attempts     INTEGER NOT NULL DEFAULT 3,
                            claimed_by       TEXT,
                            claim_expires_at TEXT,
                            created_at       TEXT NOT NULL,
                            updated_at       TEXT NOT NULL,
                            result           TEXT,
                            last_error       TEXT
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status "
                                 "ON jobs(status, id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_case "
                                 "ON jobs(case_id)")
                    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> JobQueue:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- ingest
    def enqueue(self, idempotency_key: str, case_id: str,
                payload: dict[str, Any]) -> tuple[Job, bool]:
        """Durably queue work. Returns ``(job, created)``.

        ``created is False`` means this exact event was already accepted; the
        caller must return the original outcome and must NOT queue a second
        recovery run.
        """
        now = _iso(_utcnow())
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "INSERT INTO jobs (idempotency_key, case_id, payload, status, "
                    "attempts, max_attempts, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 0, ?, ?, ?) "
                    "ON CONFLICT(idempotency_key) DO NOTHING",
                    (idempotency_key, case_id, json.dumps(payload, default=str),
                     QUEUED, self.max_attempts, now, now))
                created = cur.rowcount == 1
                row = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?",
                    (idempotency_key,)).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return Job.from_row(row), created

    # ------------------------------------------------------------- processing
    def claim(self, worker: str | None = None) -> Job | None:
        """Atomically take the oldest queued job, with a lease."""
        worker = worker or f"worker-{uuid.uuid4().hex[:8]}"
        now = _utcnow()
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY id ASC LIMIT 1",
                    (QUEUED,)).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE jobs SET status = ?, attempts = attempts + 1, "
                    "claimed_by = ?, claim_expires_at = ?, updated_at = ? WHERE id = ?",
                    (CLAIMED, worker,
                     _iso(now + timedelta(seconds=self.lease_seconds)),
                     _iso(now), row["id"]))
                claimed = conn.execute("SELECT * FROM jobs WHERE id = ?",
                                       (row["id"],)).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return Job.from_row(claimed)

    def complete(self, job_id: int, result: dict[str, Any]) -> None:
        """Mark a job done and persist its result for idempotent replay."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, result = ?, last_error = NULL, "
                "claim_expires_at = NULL, updated_at = ? WHERE id = ?",
                (DONE, json.dumps(result, default=str), _iso(_utcnow()), job_id))

    def fail(self, job_id: int, error: str) -> str:
        """Record a failure; requeue for retry or move to the dead-letter state."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT attempts, max_attempts FROM jobs WHERE id = ?",
                                   (job_id,)).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return DEAD
                status = DEAD if int(row["attempts"]) >= int(row["max_attempts"]) else QUEUED
                conn.execute(
                    "UPDATE jobs SET status = ?, last_error = ?, "
                    "claim_expires_at = NULL, updated_at = ? WHERE id = ?",
                    (status, error[:500], _iso(_utcnow()), job_id))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return status

    def recover_stale(self) -> int:
        """Return expired claims to the queue. Call this on every startup."""
        now = _iso(_utcnow())
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, claim_expires_at = NULL, updated_at = ? "
                "WHERE status = ? AND (claim_expires_at IS NULL OR claim_expires_at < ?)",
                (QUEUED, now, CLAIMED, now))
            return int(cur.rowcount)

    # -------------------------------------------------------------- inspection
    def get(self, job_id: int) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def get_by_key(self, idempotency_key: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE idempotency_key = ?",
                                     (idempotency_key,)).fetchone()
        return Job.from_row(row) if row else None

    def dead_letters(self, limit: int = 50) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY id DESC LIMIT ?",
                (DEAD, int(limit))).fetchall()
        return [Job.from_row(r) for r in rows]

    def requeue_dead(self, job_id: int) -> bool:
        """Operator action: put a dead-lettered job back in the queue."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, attempts = 0, last_error = NULL, "
                "updated_at = ? WHERE id = ? AND status = ?",
                (QUEUED, _iso(_utcnow()), job_id, DEAD))
            return cur.rowcount == 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        counts = {QUEUED: 0, CLAIMED: 0, DONE: 0, DEAD: 0}
        for row in rows:
            counts[row["status"]] = int(row["n"])
        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        return counts

    def healthy(self) -> bool:
        """Cheap writability probe used by /health and /ready."""
        try:
            with self._lock:
                self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
            return True
        except sqlite3.Error:
            return False
