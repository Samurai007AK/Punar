"""Rule-based decline diagnosis: raw payment.failed fields -> 11-class reason.

Deterministic and auditable on purpose: a reviewer can trace every recovery
decision back to the exact field that drove the classification. An LLM
classifier can replace this later once a labelled history exists.

Two payload shapes are accepted:

* the REAL Razorpay ``payment.failed`` entity, which carries FLAT
  ``error_code`` / ``error_description`` / ``error_reason`` / ``error_source``
  / ``error_step`` fields; and
* the nested ``{"error": {"code": ..., "description": ...}}`` shape used by the
  offline cohort generator and by most sandbox fixtures.

Rules are priority ordered (highest-risk and most specific first) and every
rule that could fire on a substring carries an explicit context guard, so a
"closed loop wallet" is not read as a closed account and a "card mandate
expired" is not read as an expired card. An unmatched payload is reported as
LOW CONFIDENCE rather than silently inheriting the retriable catch-all's retry
budget.
"""
from typing import Any

from punar.core.taxonomy import ReasonMeta, get_reason

# Retry budget granted to a payload no rule recognised. Deliberately smaller
# than bank_decline_general's budget: we do not know it is safe to retry.
UNKNOWN_RETRY_LIMIT = 1
CATCH_ALL = "bank_decline_general"


def _fields(failure: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Extract (code, description, method, reason, notes, source/step) from either shape."""
    err = failure.get("error")
    err = err if isinstance(err, dict) else {}
    # Flat Razorpay fields win when present; nested shape is the fallback.
    error_code = str(failure.get("error_code") or err.get("code") or "")
    error_desc = str(failure.get("error_description") or err.get("description") or "")
    error_reason = str(failure.get("error_reason") or err.get("reason") or "")
    error_src = str(failure.get("error_source") or err.get("source") or "")
    error_step = str(failure.get("error_step") or err.get("step") or "")
    method = str(failure.get("method") or "").lower()
    # `reason` is the simulator's ground-truth label in offline cohorts and a
    # free-text hint in real feeds; it is only ever a weak signal here.
    reason = str(failure.get("reason") or "")
    notes = str(failure.get("notes") or "")
    blob = " ".join([error_code, error_desc, error_reason, reason, notes])
    return (error_code.upper(), _norm(blob), method, _norm(error_reason),
            _norm(notes), _norm(f"{error_src} {error_step}"))


def _norm(text: str) -> str:
    """Lowercase and flatten separators so CODE_LIKE_THIS matches free text."""
    return text.lower().replace("_", " ").replace("-", " ").replace("/", " ")


def _is_card(method: str, c: str) -> bool:
    return method == "card" or "card" in c


def _is_upi(method: str, c: str) -> bool:
    return method == "upi" or "upi" in c or "vpa" in c


def _is_mandate(method: str, c: str) -> bool:
    return method in ("nach", "emandate", "e mandate", "emi") \
        or "mandate" in c or "nach" in c or "subscription" in c or "si " in c


def _rules(ec: str, c: str, method: str):
    """Priority-ordered (key, predicate) rules. First match wins."""
    return [
        # 1. Highest risk first: never retry anything the risk engine blocked.
        ("fraud_block", lambda: ec in ("FRAUD", "RISK_BLOCKED", "FRAUDULENT")
                       or "fraud" in c or "risk blocked" in c or "suspicious" in c),
        # 2. Lost/stolen before account_closed: "card reported lost or stolen".
        ("lost_stolen_card", lambda: ec in ("CARD_LOST", "STOLEN_CARD", "LOST_CARD")
                             or "lost or stolen" in c
                             or (("lost" in c or "stolen" in c) and _is_card(method, c))),
        # 3. Mandate rules BEFORE the card rules: "card mandate expired" is a
        #    mandate problem, not an expired card (different recovery path).
        ("mandate_expired", lambda: ec in ("MANDATE_EXPIRED", "MANDATE_TENURE_OVER")
                            or ("mandate" in c and ("expired" in c or "tenure" in c
                                                    or "validity" in c or "over" in c))
                            or "nach expired" in c),
        ("mandate_inactive", lambda: ec in ("MANDATE_INACTIVE", "E_MANDATE_CANCELLED",
                                            "MANDATE_REVOKED")
                             or ("mandate" in c and ("revoked" in c or "inactive" in c
                                                     or "cancelled" in c or "canceled" in c
                                                     or "paused" in c))
                             or ("revoked" in c and _is_mandate(method, c))),
        # 4. Account closed needs ACCOUNT context: "closed loop wallet" is not
        #    a closed account and must stay recoverable.
        ("account_closed", lambda: ec in ("ACCOUNT_CLOSED", "DORMANT_ACCOUNT")
                            or (("closed" in c or "dormant" in c or "frozen" in c)
                                and ("account" in c or "customer" in c))),
        # 5. Card expiry, guarded so it cannot swallow mandate expiry.
        ("expired_card", lambda: ec in ("EXPIRED_CARD", "CARD_EXPIRED")
                         or (("expiry" in c or "expired" in c) and _is_card(method, c)
                             and "mandate" not in c)),
        ("invalid_cvv", lambda: ec in ("INVALID_CVV", "CVV_INCORRECT", "AUTH_FAILED",
                                       "AUTHENTICATION_FAILED")
                        or "cvv" in c or "authentication failed" in c or "3ds" in c
                        or "otp incorrect" in c),
        # 6. Limits: UPI limits are their own class; a card/bank limit is an
        #    issuer decline, NOT a UPI limit and NOT an unknown payload.
        ("upi_limit_exceeded", lambda: ("limit" in c and _is_upi(method, c))),
        (CATCH_ALL, lambda: "limit" in c and ("exceeded" in c or "crossed" in c
                                              or "reached" in c)),
        # 7. Timeouts only mean upi_timeout in a UPI context; a netbanking or
        #    card gateway timeout is a general transient decline.
        ("upi_timeout", lambda: (("timeout" in c or "timed out" in c or "expired" in c
                                  or "pending" in c) and _is_upi(method, c))),
        (CATCH_ALL, lambda: "timeout" in c or "timed out" in c or "gateway" in c),
        ("insufficient_funds", lambda: ec in ("INSUFFICIENT_FUNDS", "NSRF", "NSF",
                                              "LOW_BALANCE", "INSF")
                               or "insufficient" in c or "nsf" in c
                               or "low balance" in c or "balance is low" in c
                               or "not enough" in c),
        # 8. Explicit issuer declines are a confident catch-all, not "unknown".
        (CATCH_ALL, lambda: "declin" in c or "not approve" in c or "do not honour" in c
                            or "do not honor" in c or "issuer" in c or "bank" in c
                            or "not supported" in c or "not enabled" in c
                            or "not permitted" in c),
    ]


def classify_detailed(failure: dict[str, Any]) -> tuple[str, ReasonMeta, bool]:
    """Return (reason_key, ReasonMeta, matched).

    `matched` is False when no rule fired and the catch-all was assumed -- the
    caller must treat that as low confidence rather than as a retriable decline.
    """
    ec, c, method, _reason, _notes, _src = _fields(failure)
    for key, predicate in _rules(ec, c, method):
        if predicate():
            return key, get_reason(key), True
    return CATCH_ALL, get_reason(CATCH_ALL), False


def classify(failure: dict[str, Any]) -> tuple[str, ReasonMeta]:
    """Return (reason_key, ReasonMeta) for a failed-payment payload."""
    key, meta, _matched = classify_detailed(failure)
    return key, meta


def enrich(payload: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Attach retriability metadata to a raw payment.failed payload."""
    key, meta, matched = classify_detailed(payload)
    out = dict(payload)
    out["punar_reason"] = key
    out["punar_label"] = meta.label
    out["punar_retriability"] = meta.retriability
    out["punar_suggested_action"] = meta.suggested_action
    out["punar_requires_customer_action"] = meta.requires_customer_action
    out["punar_classified"] = matched
    out["punar_confidence"] = "matched_rule" if matched else "unrecognised_code"
    # An unrecognised decline code does not earn the catch-all's retry budget.
    unknown_budget = int(((policy or {}).get("retry", {}) or {})
                         .get("unknown_retry_limit", UNKNOWN_RETRY_LIMIT))
    out["punar_retry_limit"] = meta.retry_limit if matched else min(meta.retry_limit,
                                                                    unknown_budget)
    return out
