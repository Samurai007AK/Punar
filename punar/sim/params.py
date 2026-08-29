"""Every modelling assumption in the Punar benchmark, in one place.

Nothing in the simulator may hard-code a number that encodes a belief about the
world. If a constant expresses "how often does X happen", it lives here, carries
a written justification, and is overridable from the CLI (``--set name=value``)
so a judge can re-run the whole benchmark under their own assumptions and see
how much of the result survives.

The parameter set is emitted alongside every result (``SimParams.to_dict``) so a
published number can always be traced back to the assumptions behind it.

PROVENANCE WARNING: these are *stated operator priors*, not measurements from a
real merchant book. They are ordered and shaped from published dunning practice
(WhatsApp out-reaches email in India; UPI daily limits reset on a clock; NACH
re-registration is high friction), but the absolute levels are assumptions. The
benchmark is therefore evidence about *policy design*, not a revenue forecast.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# ---------------------------------------------------------------------------
# Blocker mechanics -- the structural core of the world model.
#
# The simulator does NOT hold a (reason x intervention) success table. It holds
# a description of what is physically blocking each payment, and derives every
# intervention's success probability from that. This matters: the agent's
# decision priors (punar/core/select.py PRIORS) are a *separate* artefact, and
# the ordering of interventions the world rewards is an emergent consequence of
# the mechanics below rather than a copy of those priors.
#
# Each reason maps to (kind, daily_self_clear_hazard, customer_fix_difficulty,
# manual_resolution_rate).
#
#   kind                     what is actually wrong
#   ----                     ---------------------
#   liquidity                money is not in the account right now
#   transient                nothing is wrong; the attempt itself failed
#   limit                    a rolling cap was hit and resets on a clock
#   customer_fix             an instrument/authorisation must be changed by hand
#   hard                     the instrument is dead; no retry can ever work
#
#   daily_self_clear_hazard  P(blocker clears on its own on a given day) --
#                            salary credits, UPI daily-limit resets, issuer
#                            transients. Zero for anything needing human action.
#   customer_fix_difficulty  P(a *reached and willing* customer fails to
#                            complete the fix) -- friction of the remedy, not of
#                            the channel. Re-entering a CVV is easy; re-signing
#                            a NACH mandate is not.
#   manual_resolution_rate   P(a human ops agent resolves the case) -- the only
#                            path that does anything at all for `hard` blockers.
# ---------------------------------------------------------------------------
BlockerSpec = tuple[str, float, float, float]

DEFAULT_BLOCKERS: dict[str, BlockerSpec] = {
    # reason:                (kind,           self_clear, fix_difficulty, manual)
    "insufficient_funds":    ("liquidity",    0.085, 0.34, 0.05),
    "upi_timeout":           ("transient",    0.400, 0.10, 0.04),
    "bank_decline_general":  ("transient",    0.070, 0.30, 0.06),
    "upi_limit_exceeded":    ("limit",        0.550, 0.14, 0.05),
    "expired_card":          ("customer_fix", 0.00, 0.36, 0.10),
    "invalid_cvv":           ("customer_fix", 0.00, 0.20, 0.12),
    "mandate_inactive":      ("customer_fix", 0.00, 0.46, 0.22),
    "mandate_expired":       ("customer_fix", 0.00, 0.64, 0.26),
    "fraud_block":           ("hard",         0.02, 0.90, 0.13),
    "lost_stolen_card":      ("hard",         0.00, 0.88, 0.12),
    "account_closed":        ("hard",         0.00, 0.96, 0.08),
}

# Channel reach: P(a message on this channel is actually seen and acted on) for
# an average Indian retail customer. WhatsApp >> SMS > voice > email is the
# consistent ordering in Indian dunning practice; transactional email open rates
# sit far below messaging. `none` is the pseudo-channel used by silent retries
# and internal escalations -- no customer contact, so reach is not applicable
# and is fixed at 1.0.
DEFAULT_CHANNEL_REACH: dict[str, float] = {
    "whatsapp": 0.70,
    "sms": 0.52,
    "voice": 0.33,
    "email": 0.26,
    "none": 1.00,
}


@dataclass
class SimParams:
    """All tunable modelling assumptions. Every scalar field is CLI-overridable."""

    # --- structural tables -------------------------------------------------
    blockers: dict[str, BlockerSpec] = field(
        default_factory=lambda: dict(DEFAULT_BLOCKERS))
    channel_reach: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_REACH))

    # --- horizon -----------------------------------------------------------
    # Recurring-billing dunning cycles commonly run ~7 days before a
    # subscription is suspended; anything recovered later is out of scope.
    horizon_days: int = 7

    # --- customer latents (Beta shape parameters) --------------------------
    # Latent willingness to act on a dunning message. Mean ~0.57: most customers
    # of a *failed recurring payment* still want the service.
    intent_alpha: float = 4.0
    intent_beta: float = 3.0
    # Latent ability-to-pay right now. Symmetric: no assumption either way.
    liquidity_alpha: float = 2.5
    liquidity_beta: float = 2.5
    # Per-customer multiplicative spread on channel reach (some people live in
    # WhatsApp, some only read email). 0 = every customer is exactly average.
    channel_affinity_spread: float = 0.45

    # --- persuasion --------------------------------------------------------
    # P(a reached, willing customer with an easy fix completes it). Not 1.0:
    # people get distracted.
    persuasion_base: float = 0.82
    # Per decade of ticket size, the multiplicative drop in P(customer completes
    # payment). A Rs 25,000 charge gets more scrutiny than Rs 499.
    amount_friction: float = 0.16
    amount_friction_ref_inr: float = 1500.0
    # Persuasion multiplier when outreach is in the customer's own language
    # rather than generic English. Available to whichever arm bothers to
    # localise -- it is a property of the message, not of the arm.
    language_match_lift: float = 1.16
    # Reach multiplier for contact made outside the 08:00-19:00 IST window.
    off_window_reach_mult: float = 0.55

    # --- retries -----------------------------------------------------------
    # P(a retry attempt is captured | the blocker is genuinely cleared).
    # Below 1.0: issuer-side flakiness on re-presentment.
    retry_capture_rate: float = 0.88
    # A retry fired seconds after the failure sees almost none of the daily
    # self-clear hazard -- nothing has had time to change. This is the honest
    # version of "instant retries are bad": a *timing* effect available to any
    # arm, not a penalty attached to an arm's name.
    instant_retry_clear_mult: float = 0.10
    # Multiplier on the liquidity self-clear hazard for retries placed in a
    # payday-evening window. Timing knowledge is real and any arm may use it.
    payday_liquidity_boost: float = 2.1
    payday_window_hours: tuple[int, int] = (17, 21)
    payday_days_of_month: tuple[int, ...] = (1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31)

    # --- fatigue (ONE model, applied identically to every arm) -------------
    # Success multiplier per previous touch on the SAME channel. Repeating
    # yourself on one channel is tuned out fast.
    fatigue_same_channel: float = 0.62
    # Success multiplier per previous touch on ANY channel. Rotating channels
    # helps -- but it is not free, which is the point.
    fatigue_any_channel: float = 0.88

    # --- churn / annoyance -------------------------------------------------
    # Touches after which over-contact starts actively harming the case.
    churn_touch_threshold: int = 3
    # Multiplier on the organic self-cure hazard once a case has been
    # over-contacted. Spamming makes people less likely to come back on their own.
    churn_organic_mult: float = 0.70
    # Effect multiplier of contacting an opted-out customer. Zero: they have told
    # you to stop; the message converts nothing and only creates cost and
    # compliance exposure. Their organic self-cure is unaffected.
    optout_contact_effect: float = 0.0

    # --- money -------------------------------------------------------------
    # Modelled goodwill/support cost of one unwanted touch (a contact to an
    # opted-out customer, or outreach on a decline that can never be retried).
    # Order-of-magnitude estimate of a support-ticket deflection cost. Applied on
    # an IDENTICAL basis to every arm.
    annoyance_inr_per_unwanted_touch: float = 30.0

    # --- organic (the do-nothing control) ----------------------------------
    # How much of the organic self-cure hazard is gated on customer intent
    # (versus purely mechanical clearance of the blocker).
    organic_intent_weight: float = 0.55
    # P(a customer whose blocker has cleared re-attempts payment unprompted on
    # that day). This is what the do-nothing control measures.
    organic_retry_rate: float = 0.09

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _replace_tables(self, blockers: dict[str, BlockerSpec],
                        channel_reach: dict[str, float]) -> "SimParams":
        """Copy with the two structural tables swapped (used by world perturbation)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["blockers"] = dict(blockers)
        data["channel_reach"] = dict(channel_reach)
        return SimParams(**data)

    def with_overrides(self, overrides: dict[str, Any]) -> "SimParams":
        """Return a copy with scalar fields replaced (used by ``--set``)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        known = set(data)
        for k, v in (overrides or {}).items():
            if k not in known:
                raise KeyError(f"unknown parameter {k!r}; known: {sorted(known)}")
            cur = data[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data[k] = type(cur)(v) if isinstance(cur, (int, float)) else v
            elif isinstance(cur, bool):
                data[k] = str(v).lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                data[k] = int(float(v))
            elif isinstance(cur, float):
                data[k] = float(v)
            else:
                data[k] = v
        return SimParams(**data)


DEFAULTS = SimParams()

# Scalar parameters a judge is most likely to want to poke at.
TUNABLE = [f.name for f in fields(SimParams)
           if isinstance(getattr(DEFAULTS, f.name), (int, float))
           and not isinstance(getattr(DEFAULTS, f.name), bool)]


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set key=value`` pairs from the CLI."""
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Ticket-size distributions for the sensitivity analysis.
#
# PROVENANCE: none of these are measured from a real merchant book -- they are
# stated shapes, offered so a reader can see how much of the rupee headline is an
# artefact of the assumed ticket mix. `default` is the shape this repo has always
# used; the others bracket it. The recovery-RATE result should be near-invariant
# across all four; the rupee result should not be, and that is the point.
# ---------------------------------------------------------------------------
AMOUNT_DISTRIBUTIONS: dict[str, tuple[list[int], list[float]]] = {
    "default": ([499, 999, 1499, 2499, 4999, 9999, 14999, 24999, 49999],
                [0.14, 0.18, 0.12, 0.16, 0.12, 0.08, 0.06, 0.08, 0.06]),
    # Consumer subscription book: mostly small monthly plans.
    "subscription_small": ([149, 299, 499, 999, 1499, 2499, 4999],
                           [0.20, 0.24, 0.22, 0.16, 0.10, 0.05, 0.03]),
    # SaaS/B2B book: fewer, larger invoices.
    "b2b_large": ([2499, 4999, 9999, 19999, 49999, 99999],
                  [0.18, 0.22, 0.24, 0.18, 0.12, 0.06]),
    # Deliberately flat: no shape assumption at all.
    "uniform": ([499, 999, 1499, 2499, 4999, 9999, 14999, 24999, 49999],
                [1.0 / 9] * 9),
}
