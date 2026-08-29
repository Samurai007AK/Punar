"""Taxonomy classification: every known decline maps to an 11-class reason."""
from punar.core.classify import classify, enrich
from punar.core.taxonomy import (
    COHORT_REASON_WEIGHTS,
    CONDITIONAL,
    NON_RETRYABLE,
    REASONS,
    RETRIABLE,
    reason_labels,
)


def test_all_reasons_present():
    assert len(REASONS) == 11
    assert len(reason_labels()) == 11
    assert abs(sum(COHORT_REASON_WEIGHTS.values()) - 1.0) < 1e-9


def test_retriability_categories():
    cats = {m.retriability for m in REASONS.values()}
    assert cats <= {RETRIABLE, CONDITIONAL, NON_RETRYABLE}
    # sanity: known exemplars
    assert REASONS["insufficient_funds"].retriability == RETRIABLE
    assert REASONS["fraud_block"].retriability == NON_RETRYABLE
    assert REASONS["expired_card"].retriability == CONDITIONAL


def test_classify_insufficient_funds():
    for code in ["INSUFFICIENT_FUNDS", "NSRF", "NSF", "LOW_BALANCE"]:
        key, meta = classify({"error": {"code": code, "description": "x"}})
        assert key == "insufficient_funds", code
        assert meta.retriability == RETRIABLE


def test_classify_upi_timeout():
    key, _ = classify({"method": "upi", "error": {"code": "TIMEOUT", "description": "intent pending"}})
    assert key == "upi_timeout"


def test_classify_expired_card():
    key, _ = classify({"error": {"code": "EXPIRED_CARD", "description": "x"}})
    assert key == "expired_card"
    key, _ = classify({"error": {"code": "X", "description": "card has expired"}})
    assert key == "expired_card"


def test_classify_fraud_blocks_retry():
    key, meta = classify({"error": {"code": "FRAUD"}})
    assert key == "fraud_block" and meta.retriability == NON_RETRYABLE


def test_classify_mandate_variants():
    assert classify({"notes": "e-mandate revoked by customer"})[0] == "mandate_inactive"
    assert classify({"notes": "nach/e-mandate tenure ended"})[0] == "mandate_expired"


def test_classify_unknown_falls_back():
    key, meta = classify({})
    assert key == "bank_decline_general" and meta.retriability == RETRIABLE


def test_enrich_attaches_metadata():
    p = enrich({"error": {"code": "INVALID_CVV"}, "amount_inr": 1499})
    assert p["punar_reason"] == "invalid_cvv"
    assert p["punar_label"] == "Invalid CVV / authentication failed"
    assert p["punar_retriability"] == CONDITIONAL
    assert p["punar_requires_customer_action"] is True
