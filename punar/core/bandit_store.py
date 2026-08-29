"""Durable Beta posteriors for the contextual Thompson-Sampling bandit.

Without this, `seed_arms` re-initialises every arm from the frozen PRIORS dict
on every call, so an observed outcome is discarded the moment the round ends --
the bandit has no memory and the "learns per-failure-reason recovery rates"
claim is false in code. This module gives each (reason, intervention) pair a
row that survives the process.

Storage is SQLite, keyed `(reason, intervention)`, holding `alpha`, `beta`,
`updates` and `last_updated`. The default path is the `PUNAR_BANDIT_DB`
environment variable, falling back to `punar_bandit.db`; tests inject their own
path (or their own store object) instead.

Every read and write is defensive: if the database is missing, locked, or
corrupt, the caller silently falls back to the rule-derived priors. A bandit
that cannot reach its posterior store must degrade, never crash a payment
recovery.
"""
import os
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

DEFAULT_DB_ENV = "PUNAR_BANDIT_DB"
DEFAULT_DB_PATH = "punar_bandit.db"


def default_db_path() -> str:
    """Configured posterior-store path (env var wins, else `punar_bandit.db`)."""
    return os.getenv(DEFAULT_DB_ENV) or DEFAULT_DB_PATH


class BanditStore:
    """SQLite-backed Beta posterior store, keyed by (reason, intervention)."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_db_path()
        self._lock = threading.Lock()
        self.available = self._init()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> bool:
        try:
            conn = self._connect()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posteriors (
                    reason       TEXT NOT NULL,
                    intervention TEXT NOT NULL,
                    alpha        REAL NOT NULL,
                    beta         REAL NOT NULL,
                    updates      INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY (reason, intervention)
                )
            """)
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------ reads
    def get(self, reason: str, intervention: str) -> tuple[float, float] | None:
        """Posterior (alpha, beta) for one arm, or None if never observed."""
        if not self.available:
            return None
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT alpha, beta FROM posteriors WHERE reason = ? AND intervention = ?",
                (reason, intervention)).fetchone()
            conn.close()
        except sqlite3.Error:
            return None
        return (float(row["alpha"]), float(row["beta"])) if row else None

    def get_many(self, reason: str,
                 interventions: Iterable[str]) -> dict[str, tuple[float, float]]:
        """Posteriors for several arms of one reason in a single round trip."""
        names = list(interventions)
        if not self.available or not names:
            return {}
        try:
            conn = self._connect()
            marks = ",".join("?" * len(names))
            rows = conn.execute(
                f"SELECT intervention, alpha, beta FROM posteriors "
                f"WHERE reason = ? AND intervention IN ({marks})",
                [reason, *names]).fetchall()
            conn.close()
        except sqlite3.Error:
            return {}
        return {r["intervention"]: (float(r["alpha"]), float(r["beta"])) for r in rows}

    def all_posteriors(self) -> list[dict[str, Any]]:
        """Every stored posterior, for convergence charts and audit exports."""
        if not self.available:
            return []
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT reason, intervention, alpha, beta, updates, last_updated "
                "FROM posteriors ORDER BY reason, intervention").fetchall()
            conn.close()
        except sqlite3.Error:
            return []
        return [{"reason": r["reason"], "intervention": r["intervention"],
                 "alpha": float(r["alpha"]), "beta": float(r["beta"]),
                 "updates": int(r["updates"]),
                 "mean": float(r["alpha"]) / (float(r["alpha"]) + float(r["beta"])),
                 "last_updated": r["last_updated"]} for r in rows]

    # ----------------------------------------------------------------- writes
    def put(self, reason: str, intervention: str, alpha: float, beta: float,
            updates: int | None = None) -> bool:
        """Write an absolute posterior. Returns False if the store is unusable."""
        if not self.available:
            return False
        stamp = datetime.now(UTC).isoformat()
        try:
            with self._lock:
                conn = self._connect()
                if updates is None:
                    conn.execute(
                        "INSERT INTO posteriors (reason, intervention, alpha, beta, updates, last_updated) "
                        "VALUES (?, ?, ?, ?, COALESCE("
                        "  (SELECT updates FROM posteriors WHERE reason = ? AND intervention = ?), 0), ?) "
                        "ON CONFLICT(reason, intervention) DO UPDATE SET "
                        "  alpha = excluded.alpha, beta = excluded.beta, "
                        "  last_updated = excluded.last_updated",
                        (reason, intervention, float(alpha), float(beta),
                         reason, intervention, stamp))
                else:
                    conn.execute(
                        "INSERT INTO posteriors (reason, intervention, alpha, beta, updates, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(reason, intervention) DO UPDATE SET "
                        "  alpha = excluded.alpha, beta = excluded.beta, "
                        "  updates = excluded.updates, last_updated = excluded.last_updated",
                        (reason, intervention, float(alpha), float(beta), int(updates), stamp))
                conn.commit()
                conn.close()
            return True
        except sqlite3.Error:
            return False

    def record_outcome(self, reason: str, intervention: str, success: bool,
                       prior: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float]:
        """Apply one Bayesian update and return the resulting (alpha, beta).

        Seeds the row from `prior` the first time an arm is observed, so the
        rule-derived prior is carried into the store instead of being replaced
        by a flat Beta(1, 1).
        """
        alpha, beta = self.get(reason, intervention) or (float(prior[0]), float(prior[1]))
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        if not self.available:
            return alpha, beta
        stamp = datetime.now(UTC).isoformat()
        try:
            with self._lock:
                conn = self._connect()
                conn.execute(
                    "INSERT INTO posteriors (reason, intervention, alpha, beta, updates, last_updated) "
                    "VALUES (?, ?, ?, ?, 1, ?) "
                    "ON CONFLICT(reason, intervention) DO UPDATE SET "
                    "  alpha = excluded.alpha, beta = excluded.beta, "
                    "  updates = posteriors.updates + 1, last_updated = excluded.last_updated",
                    (reason, intervention, alpha, beta, stamp))
                conn.commit()
                conn.close()
        except sqlite3.Error:
            pass
        return alpha, beta

    def clear(self) -> int:
        if not self.available:
            return 0
        try:
            with self._lock:
                conn = self._connect()
                cur = conn.execute("DELETE FROM posteriors")
                conn.commit()
                n = cur.rowcount
                conn.close()
            return max(n, 0)
        except sqlite3.Error:
            return 0


def store_for_policy(policy: dict[str, Any] | None) -> BanditStore | None:
    """Build the posterior store a policy asks for, or None.

    Persistence is OPT-IN (`bandit.persist_posteriors`). The offline benchmark
    must stay reproducible across runs of the same seed, so the shipped policy
    leaves it off; a production deployment turns it on and the bandit then
    carries learning across cases, rounds and processes.
    """
    cfg = ((policy or {}).get("bandit", {}) or {})
    if not cfg.get("persist_posteriors"):
        return None
    return BanditStore(cfg.get("db_path") or default_db_path())
