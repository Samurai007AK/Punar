"""11-class decline taxonomy for failed recurring payments.

Each reason carries a retriability verdict, a suggested next action and
a recommended retry budget. This is deterministic, explainable logic --
no black-box classifier -- so every recovery decision is auditable to an
RBI/Fair-Practices reviewer.
"""
from dataclasses import dataclass
from typing import Any

RETRIABLE = "retryable"          # safe to retry with timing/channel tweaks
CONDITIONAL = "conditional"      # retry only after customer fixes something
NON_RETRYABLE = "non_retryable"  # never retry; escalate or write off


@dataclass(frozen=True)
class ReasonMeta:
    id: str
    label: str
    retriability: str
    suggested_action: str
    retry_limit: int
    requires_customer_action: bool = False
    priority: int = 0            # higher = recover more aggressively
    # Hours to wait before the next attempt for this reason. Drives the agent's
    # scheduler (agent._schedule_next) so episode length is governed by policy
    # rather than by a hardcoded constant. Overridable per merchant via
    # policy.json -> retry.backoff_hours.<reason>.
    retry_backoff_hours: float = 24.0


REASONS: dict[str, ReasonMeta] = {
    "insufficient_funds": ReasonMeta(
        "insufficient_funds", "Insufficient funds / NSRF", RETRIABLE,
        "silent_retry_aligned_to_payday_then_gentle_nudge", 5,
        retry_backoff_hours=4.0),
    "expired_card": ReasonMeta(
        "expired_card", "Card expired / expiry reached", CONDITIONAL,
        "payment_link_to_update_card", 0, requires_customer_action=True,
        retry_backoff_hours=24.0),
    "lost_stolen_card": ReasonMeta(
        "lost_stolen_card", "Card reported lost or stolen", NON_RETRYABLE,
        "escalate_for_new_mandate_registration", 0, requires_customer_action=True,
        retry_backoff_hours=24.0),
    "invalid_cvv": ReasonMeta(
        "invalid_cvv", "Invalid CVV / authentication failed", CONDITIONAL,
        "payment_link_with_cvv_reentry", 0, requires_customer_action=True,
        retry_backoff_hours=12.0),
    "upi_timeout": ReasonMeta(
        "upi_timeout", "UPI timeout / intent pending expiry", RETRIABLE,
        "immediate_silent_retry_or_upi_reapprove", 4,
        retry_backoff_hours=1.0),
    "upi_limit_exceeded": ReasonMeta(
        "upi_limit_exceeded", "UPI transaction limit exceeded", CONDITIONAL,
        "retry_after_limit_reset_with_nudge", 3, requires_customer_action=True,
        retry_backoff_hours=24.0),
    "mandate_inactive": ReasonMeta(
        "mandate_inactive", "Mandate inactive / revoked by customer", CONDITIONAL,
        "reauth_request_via_email_and_whatsapp", 2, requires_customer_action=True,
        retry_backoff_hours=24.0),
    "mandate_expired": ReasonMeta(
        "mandate_expired", "Mandate expired (NACH/e-mandate tenure ended)", NON_RETRYABLE,
        "re_registration_link_then_write_off_if_no_action", 0, requires_customer_action=True,
        retry_backoff_hours=24.0),
    "bank_decline_general": ReasonMeta(
        "bank_decline_general", "General bank decline (issuer unspecified)", RETRIABLE,
        "silent_retry_later_different_window", 3,
        retry_backoff_hours=6.0),
    "fraud_block": ReasonMeta(
        "fraud_block", "Declined by fraud/risk filter", NON_RETRYABLE,
        "escalate_to_manual_review", 0,
        retry_backoff_hours=24.0),
    "account_closed": ReasonMeta(
        "account_closed", "Customer account closed", NON_RETRYABLE,
        "write_off_and_update_billing_details", 0, requires_customer_action=True,
        retry_backoff_hours=24.0),
}


def get_reason(reason_key: str) -> ReasonMeta:
    if reason_key in REASONS:
        return REASONS[reason_key]
    return REASONS["bank_decline_general"]


def reason_labels() -> list[str]:
    return list(REASONS.keys())


def retry_backoff_hours(reason_key: str, policy: dict[str, Any] | None = None) -> float:
    """Hours to wait before the next attempt, merchant-overridable.

    Order of precedence: policy.retry.backoff_hours.<reason> -> taxonomy default.
    Kept here (not in the agent) so the retry cadence stays part of the auditable
    decline taxonomy rather than a magic constant in the control loop.
    """
    override = ((policy or {}).get("retry", {}).get("backoff_hours", {}) or {}).get(reason_key)
    if override is not None:
        return float(override)
    return float(get_reason(reason_key).retry_backoff_hours)


# Distribution used by the offline simulator (sums to 1). Real merchants can
# reweight from their own payment.failed history.
COHORT_REASON_WEIGHTS = {
    "insufficient_funds": 0.38,
    "upi_timeout": 0.15,
    "expired_card": 0.12,
    "bank_decline_general": 0.10,
    "mandate_inactive": 0.08,
    "upi_limit_exceeded": 0.06,
    "mandate_expired": 0.04,
    "invalid_cvv": 0.03,
    "fraud_block": 0.02,
    "lost_stolen_card": 0.01,
    "account_closed": 0.01,
}
