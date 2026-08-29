"""Audit-store tests: append-only semantics and tamper evidence.

The audit trail is the project's compliance claim, so these tests assert the
guarantee rather than the implementation: a revision never overwrites its
predecessor, ordering cannot be influenced by caller-supplied data, and any
edit or deletion made behind the store's back is detected.
"""
import json
import sqlite3

import pytest

from punar.audit import AppendOnlyViolation, AuditStore


@pytest.fixture()
def store(tmp_path):
    """A throwaway store that is always closed, so Windows can unlink the file."""
    with AuditStore(str(tmp_path / "audit.db")) as s:
        yield s


def _record(case_id="pay_1", **extra):
    return {"case_id": case_id, "outcome": "recovered", **extra}


# ------------------------------------------------------------------ appending
def test_append_and_get_latest(store):
    store.append(_record(outcome="retry_later"))
    store.append(_record(outcome="recovered"))
    latest = store.get_latest("pay_1")
    assert latest["outcome"] == "recovered"
    assert latest["audit_revision"] == 2


def test_upsert_appends_a_revision_instead_of_overwriting(store):
    """The historical bug: upsert() issued UPDATE and destroyed the prior version."""
    store.upsert(_record(outcome="first"))
    store.upsert(_record(outcome="second"))

    assert store.count_rows() == 2, "a second write must not overwrite the first"
    history = store.get_history("pay_1")
    assert [h["outcome"] for h in history] == ["first", "second"]


def test_history_is_ordered_by_row_id_not_caller_timestamp(store):
    """Caller-supplied created_at must never decide which revision is latest."""
    store.append(_record(outcome="genuine"))
    store.append(_record(outcome="forged", created_at="9999-12-31T00:00:00"))
    store.append(_record(outcome="newest"))

    assert store.get_latest("pay_1")["outcome"] == "newest"
    # The caller's value is preserved for the record, just not as the ordering key.
    assert store.get_history("pay_1")[1]["reported_at"] == "9999-12-31T00:00:00"


def test_rows_for_distinct_cases_are_separated(store):
    store.append(_record("pay_a"))
    store.append(_record("pay_b"))
    assert store.count_cases() == 2
    assert store.get_latest("pay_a")["case_id"] != store.get_latest("pay_b")["case_id"]


# ------------------------------------------------------------- tamper evidence
def test_clean_chain_verifies(store):
    for i in range(5):
        store.append(_record(f"pay_{i}"))
    result = store.verify_chain()
    assert result.ok, result.problems
    assert result.rows_checked == 5


def test_in_place_update_is_blocked_by_trigger(store):
    store.append(_record())
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE audit_log SET data = '{}' WHERE id = 1")


def test_delete_is_blocked_by_trigger(store):
    store.append(_record())
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("DELETE FROM audit_log WHERE id = 1")


def test_verify_chain_detects_out_of_band_tampering(tmp_path):
    """An attacker with file access edits a row; the chain must notice."""
    path = str(tmp_path / "audit.db")
    with AuditStore(path) as s:
        for i in range(3):
            s.append(_record(f"pay_{i}"))
        assert s.verify_chain().ok

    # Bypass the store (and its triggers) exactly as a sqlite3 shell would.
    raw = sqlite3.connect(path)
    raw.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
    forged = json.dumps({"case_id": "pay_1", "outcome": "TAMPERED"})
    raw.execute("UPDATE audit_log SET data = ? WHERE id = 2", (forged,))
    raw.commit()
    raw.close()

    with AuditStore(path) as s:
        result = s.verify_chain()
        assert not result.ok
        assert result.first_bad_id == 2
        assert result.problems


def test_verify_chain_detects_deletion_of_the_head_row(tmp_path):
    path = str(tmp_path / "audit.db")
    with AuditStore(path) as s:
        for i in range(3):
            s.append(_record(f"pay_{i}"))

    raw = sqlite3.connect(path)
    raw.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
    raw.execute("DELETE FROM audit_log WHERE id = 3")
    raw.commit()
    raw.close()

    with AuditStore(path) as s:
        assert not s.verify_chain().ok


# ------------------------------------------------------------------- deletion
def test_clear_is_disabled(store):
    store.append(_record())
    with pytest.raises(AppendOnlyViolation):
        store.clear()
    assert store.count_rows() == 1


def test_destroy_requires_the_confirmation_token(store):
    store.append(_record())
    with pytest.raises(AppendOnlyViolation):
        store.destroy_for_tests("please")
    assert store.count_rows() == 1


# --------------------------------------------------------------------- schema
def test_schema_version_is_set(store):
    assert store.schema_version >= 1
