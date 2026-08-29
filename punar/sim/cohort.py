"""Synthetic failed-payment cohort generator.

Produces self-consistent cases: every case is seeded with a ground-truth
decline reason AND populated with raw Razorpay-style error fields that the
deterministic classifier maps back to that same reason. This keeps the offline
benchmark honest -- classification accuracy is not confounding recovery lift.
"""
import hashlib
import random
from typing import Any

from punar.core.taxonomy import COHORT_REASON_WEIGHTS, reason_labels

MERCHANTS = [
    "Acme SaaS", "BetaRead", "CloudKart", "DigiBima", "EduFlow", "FitTrack Pro",
    "Gharana Music", "HissaBook", "Invoicely", "JugnooRide", "KitabiDuniya",
    "LearnUp", "MediStore", "NiftyCart", "OrderStack", "PayLater Labs",
    "Quotient HR", "RentPe", "SastoDeal", "TaxSaver", "Udaan Travels",
    "VahanBazaar", "Workly", "YogaMandir", "Zivame Fit",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Diya", "Saisha", "Ananya", "Priya", "Meera", "Riya",
    "Neha", "Simran", "Kavya", "Aditi", "Rohan", "Karan", "Vikram", "Arjun",
]

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
          "Pune", "Ahmedabad", "Jaipur", "Kolkata", "Lucknow"]

# Reason -> plausible Razorpay-style fields so classify() recovers the reason.
REASON_FIELDS = {
    "insufficient_funds": [
        ("INSUFFICIENT_FUNDS", "insufficient balance in account"),
        ("NSRF", "non sufficient funds"),
        ("NSF", "funds insufficient for transaction"),
        ("LOW_BALANCE", "available balance is low"),
    ],
    "upi_timeout": [("TIMEOUT", "upi payment intent pending expiry"),
                    ("GATEWAY_TIMEOUT", "bank response timed out")],
    "expired_card": [("EXPIRED_CARD", "card expiry date has passed"),
                     ("CARD_EXPIRED", "card has expired")],
    "bank_decline_general": [("BANK_DECLINED", "issuer declined the transaction"),
                             ("GENERIC_DECLINE", "bank did not approve")],
    "mandate_inactive": [("MANDATE_INACTIVE", "e-mandate revoked by customer"),
                         ("E_MANDATE_CANCELLED", "nach mandate cancelled")],
    "upi_limit_exceeded": [("UPI_LIMIT", "upi transaction limit exceeded"),
                           ("LIMIT_EXCEEDED", "per day upi limit crossed")],
    "mandate_expired": [("MANDATE_EXPIRED", "nach/e-mandate tenure ended"),
                        ("MANDATE_TENURE_OVER", "e-mandate validity period over")],
    "invalid_cvv": [("INVALID_CVV", "cvv does not match"),
                    ("AUTH_FAILED", "3ds authentication failed")],
    "fraud_block": [("FRAUD", "blocked by risk engine"),
                    ("RISK_BLOCKED", "suspected fraudulent transaction")],
    "lost_stolen_card": [("CARD_LOST", "card reported lost or stolen"),
                         ("STOLEN_CARD", "card reported stolen")],
    "account_closed": [("ACCOUNT_CLOSED", "customer account closed"),
                       ("DORMANT_ACCOUNT", "account dormant and closed")],
}

METHOD_BY_REASON = {
    "insufficient_funds": ["card", "upi"],
    "upi_timeout": ["upi"],
    "expired_card": ["card"],
    "bank_decline_general": ["card"],
    "mandate_inactive": ["nach"],
    "upi_limit_exceeded": ["upi"],
    "mandate_expired": ["nach"],
    "invalid_cvv": ["card"],
    "fraud_block": ["card", "upi"],
    "lost_stolen_card": ["card"],
    "account_closed": ["card", "nach"],
}

LANG_WEIGHTS = {"en": 0.45, "hi": 0.25, "hinglish": 0.30}
AMOUNT_TIERS = [499, 999, 1499, 2499, 4999, 9999, 14999, 24999, 49999]
AMOUNT_W = [0.14, 0.18, 0.12, 0.16, 0.12, 0.08, 0.06, 0.08, 0.06]


def _weighted_choice(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    return rng.choices(items, weights=weights, k=1)[0]


def generate_cohort(size: int, seed: int,
                    include_unclassifiable: bool = False) -> list[dict[str, Any]]:
    """Generate a seeded list of failed-payment cases."""
    rng = random.Random(seed)
    reasons = reason_labels()
    weights = [COHORT_REASON_WEIGHTS[r] for r in reasons]
    cases: list[dict[str, Any]] = []

    # Probability mass of deliberately ambiguous payloads (tests the classifier
    # fallback path). Kept small so the benchmark stays interpretable.
    ambiguous_mass = 0.03 if include_unclassifiable else 0.0

    for i in range(size):
        rid = f"case-{seed}-{i:04d}"
        # Day-of-month skewed toward month-start (post-payday attempt spike).
        if rng.random() < 0.35:
            dom = rng.randint(1, 5)
        elif rng.random() < 0.6:
            dom = rng.randint(25, 28)
        else:
            dom = rng.randint(6, 24)
        hour = _weighted_choice(rng, list(range(7, 22)),
                                [3, 3, 4, 5, 6, 6, 5, 4, 4, 5, 5, 4, 4, 3, 2])
        merchant = rng.choice(MERCHANTS)
        first = rng.choice(FIRST_NAMES)
        city = rng.choice(CITIES)
        customer_id = f"cust-{hashlib.sha1(f'{seed}-{i}'.encode()).hexdigest()[:8]}"
        amount = _weighted_choice(rng, AMOUNT_TIERS, AMOUNT_W)
        lang = _weighted_choice(rng, list(LANG_WEIGHTS), list(LANG_WEIGHTS.values()))
        opted_out = rng.random() < 0.05
        paylink = f"https://razorpay.me/pay/{rid}"

        # Choose reason, reserving a tiny slice for unclassifiable noise.
        if rng.random() < ambiguous_mass:
            reason = "bank_decline_general"
            code, desc = "ISSUER_UNAVAILABLE", "issuer could not be reached"
        else:
            reason = _weighted_choice(rng, reasons, weights)
            code, desc = rng.choice(REASON_FIELDS[reason])
        method = rng.choice(METHOD_BY_REASON[reason])

        payload = {
            "case_id": rid,
            "customer_id": customer_id,
            "merchant_name": merchant,
            "amount_inr": float(amount),
            "reason": reason,  # simulator ground truth
            "method": method,
            "day_of_month": dom,
            "hour": hour,
            "language": lang,
            "opted_out": opted_out,
            "payment_link": paylink,
            "city": city,
            "customer_name": first,
            "notes": "",
            "error": {"code": code, "description": desc, "step": "capture"},
        }
        cases.append(payload)
    return cases
