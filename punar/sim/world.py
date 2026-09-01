"""The world model: ONE scoring function, shared by every comparator arm.

Design rule that this module exists to enforce
-----------------------------------------------
There is exactly one function that decides whether a touch worked --
:func:`touch_success_prob` -- and it does **not** know which policy produced the
touch. It has no ``arm``/``baseline`` parameter and no branch on one. Two arms
that emit the same intervention, on the same case, on the same day, at the same
hour, in the same language, with the same touch history, get the identical
probability. Every difference between arms therefore has to be earned by *what*
they choose to do and *when*.

Why the world is mechanistic rather than a rate table
-----------------------------------------------------
The obvious way to simulate this is a hand-written (reason x intervention)
success table. That is exactly what makes an agent benchmark circular: if the
simulator's table and the agent's decision priors are two spellings of the same
belief, the agent is graded by the function it optimises and the result shows
nothing.

So the world here holds no such table. It holds a description of *what is
physically blocking each payment* (:data:`punar.sim.params.DEFAULT_BLOCKERS`)
plus a handful of mechanism parameters -- channel reach, persuasion, ticket-size
friction, fatigue, capture rate. Which intervention is best for which decline
reason is never written down anywhere; it *falls out* of those mechanics. It can
therefore disagree with the agent's priors, and it does (run
``--check-circularity`` to see where).

Structure of a case in this world
---------------------------------
Each case gets latent customer state derived from its ``customer_id`` (intent,
liquidity, per-channel affinity) and a **clear-day**: the day on which whatever
was blocking the payment stops blocking it -- salary lands, a UPI daily limit
resets, an issuer transient passes, or the customer replaces a dead card of
their own accord. The clear-day is drawn once per case and shared by every arm,
so arms are compared on identical worlds (a paired design).

From there:

* a **retry** succeeds if the blocker is already clear and the presentment is
  captured -- which is why retry timing, not retry count, is what pays;
* a **contact** succeeds if it is seen (channel reach x contact window),
  understood (language), persuasive (intent, ticket size, how hard the remedy
  is) and *actionable* (for a money problem, a nudge only converts if there is
  money, or if the payment link lets them pay another way);
* an **escalation** routes to a human and is the only path that does anything at
  all for a dead instrument.

Organic self-cure -- customers who fix themselves with no prompting at all -- is
drawn from the same clear-day process and is what the do-nothing control
measures. Over-contact suppresses it, so an arm that spams can genuinely end up
behind doing nothing.
"""
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

from punar.sim.params import SimParams

# Which physical mechanism each intervention actually uses. This is structural
# (what the action *is*), not a belief about how well it works.
MECHANISM: dict[str, str] = {
    "silent_retry_aligned": "retry",
    "whatsapp_nudge_payment_link": "contact",
    "email_payment_link": "contact",
    "payment_link_sms": "contact",
    "voice_call": "contact",
    "promise_to_pay": "contact",      # a contact that also schedules a retry
    "escalate_manual": "manual",
}

# Kinds whose remedy is time/money rather than a customer action.
_MECHANICAL_KINDS = ("liquidity", "transient", "limit")

# Additional world parameters that are structural rather than tunable beliefs
# live on SimParams; these two are the exceptions worth naming inline because
# they only make sense together with the clear-day process.
SPONTANEOUS_FIX_HAZARD = 0.045
"""Daily P(a customer replaces a dead instrument with nobody asking them to),
before scaling by how hard the fix is and how engaged they are."""

NUDGE_UNBLOCK_RATE = 0.22
"""P(a payment link converts even though the original blocker has not cleared)
-- the customer pays from another instrument. Without this, nudging anyone with
a money problem would be exactly worthless, which is too strong a claim."""


def _rng(seed: int, *parts: Any) -> random.Random:
    key = "|".join([str(seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _unit(seed: int, *parts: Any) -> float:
    """A single deterministic U(0,1) draw keyed by (seed, *parts)."""
    return _rng(seed, *parts).random()


# ---------------------------------------------------------------------------
# latent customer state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Latents:
    intent: float                     # willingness to act on a nudge
    liquidity: float                  # ability to pay right now
    reach: dict[str, float]           # per-channel P(message is seen & acted on)


def latents_for(case: dict[str, Any], seed: int, params: SimParams) -> Latents:
    """Per-customer latent state. Keyed by customer_id, so the same customer
    behaves the same way in every arm."""
    cid = str(case.get("customer_id") or case.get("case_id"))
    r = _rng(seed, "latent", cid)
    intent = r.betavariate(params.intent_alpha, params.intent_beta)
    liquidity = r.betavariate(params.liquidity_alpha, params.liquidity_beta)
    reach: dict[str, float] = {}
    for ch in sorted(params.channel_reach):          # sorted: no dict-order dependence
        base = params.channel_reach[ch]
        if ch == "none":
            reach[ch] = 1.0
            continue
        a = _unit(seed, "affinity", cid, ch)
        reach[ch] = min(0.98, max(0.02,
                                  base * (1.0 + params.channel_affinity_spread * (2.0 * a - 1.0))))
    return Latents(intent=intent, liquidity=liquidity, reach=reach)


def blocker_for(reason: str, params: SimParams):
    return params.blockers.get(reason, params.blockers["bank_decline_general"])


def _is_payday(dom: int, params: SimParams) -> bool:
    return ((dom - 1) % 31) + 1 in params.payday_days_of_month


def _daily_clear_hazard(case: dict[str, Any], lat: Latents, day: int,
                        params: SimParams) -> float:
    """P(the blocker stops blocking on day `day` | it has not already)."""
    kind, self_clear, fix_difficulty, _manual = blocker_for(
        case.get("reason", "bank_decline_general"), params)
    if kind == "liquidity":
        dom = int(case.get("day_of_month", 15)) + day
        h = self_clear * (0.5 + lat.liquidity)
        if _is_payday(dom, params):
            h *= params.payday_liquidity_boost
    elif kind in ("transient", "limit"):
        h = self_clear
    else:  # customer_fix / hard -- only a spontaneous, unprompted remedy
        h = self_clear + SPONTANEOUS_FIX_HAZARD * (1.0 - fix_difficulty) * lat.intent
    return min(0.95, max(0.0, h))


def clear_day(case: dict[str, Any], lat: Latents, seed: int,
              params: SimParams) -> int | None:
    """Day (1..horizon) on which this case's blocker clears by itself, or None.

    Drawn once per case and shared by every arm: arms are compared on identical
    worlds. Day 0 is handled separately -- see :func:`_cleared_by`.
    """
    for d in range(1, params.horizon_days + 1):
        if _unit(seed, "clear", case.get("case_id"), d) < _daily_clear_hazard(case, lat, d, params):
            return d
    return None


def _cleared_by(state: "CaseWorld", day: int) -> bool:
    """Has the blocker cleared by `day`? Day 0 (an instant retry, seconds after
    the failure) sees almost none of the daily hazard -- nothing has changed yet."""
    if day <= 0:
        h = _daily_clear_hazard(state.case, state.latents, 1, state.params)
        return _unit(state.seed, "clear0", state.case.get("case_id")) < \
            h * state.params.instant_retry_clear_mult
    cd = state.clear_day
    return cd is not None and cd <= day


# ---------------------------------------------------------------------------
# per-case world state (mutated as an arm walks its touches)
# ---------------------------------------------------------------------------
@dataclass
class CaseWorld:
    case: dict[str, Any]
    latents: Latents
    params: SimParams
    seed: int
    clear_day: int | None = None
    touches_total: int = 0
    touches_by_key: dict[str, int] = field(default_factory=dict)
    extra_days: list[int] = field(default_factory=list)
    """Landing day of each touch, in order. Arms that do not control their own
    clock (the agent) record here so organic self-cure can be lined up."""

    @classmethod
    def build(cls, case: dict[str, Any], seed: int, params: SimParams) -> "CaseWorld":
        lat = latents_for(case, seed, params)
        return cls(case=case, latents=lat, params=params, seed=seed,
                   clear_day=clear_day(case, lat, seed, params))


def fatigue_key(intervention: str, channel: str) -> str:
    """What a touch fatigues. Silent retries and manual escalations both map to
    the `none` channel but are completely different actions, so they must not
    fatigue each other."""
    mech = MECHANISM.get(intervention, "contact" if channel not in ("none", "") else "manual")
    if mech == "retry":
        return "retry"
    if mech == "manual":
        return "manual"
    return channel or "email"


# ---------------------------------------------------------------------------
# THE scoring function
# ---------------------------------------------------------------------------
def touch_success_prob(state: CaseWorld, intervention: str, channel: str,
                       day: int, hour: int, localized: bool) -> float:
    """P(this touch recovers the payment). Policy-agnostic by construction.

    Arguments describe the *touch*, never the arm that produced it:
      intervention/channel  what was done and over what medium
      day                   whole days since the failure (0 = instant)
      hour                  local IST hour the touch lands at
      localized             was the copy in the customer's own language
    """
    p = state.params
    case = state.case
    lat = state.latents
    kind, _self_clear, fix_difficulty, manual_rate = blocker_for(
        case.get("reason", "bank_decline_general"), p)
    mech = MECHANISM.get(intervention, "contact" if channel not in ("none", "") else "manual")

    key = fatigue_key(intervention, channel)
    prior_same = state.touches_by_key.get(key, 0)
    prior_any = state.touches_total
    fatigue = (p.fatigue_same_channel ** prior_same) * (p.fatigue_any_channel ** prior_any)

    if mech == "manual":
        # Internal escalation: a human works the case. No customer contact, so
        # no reach/window/language terms.
        raw = manual_rate * (0.6 + 0.4 * lat.intent)
        return max(0.0, min(0.95, raw * fatigue))

    if mech == "retry":
        # A retry is a presentment. It converts iff the blocker is genuinely
        # clear at that moment and the issuer captures it.
        if not _cleared_by(state, day):
            return 0.0
        return max(0.0, min(0.98, p.retry_capture_rate * fatigue))

    # --- contact ----------------------------------------------------------
    if case.get("opted_out"):
        # They asked you to stop. The message converts nothing; it only creates
        # cost and compliance exposure.
        return max(0.0, p.optout_contact_effect)

    reach = lat.reach.get(channel, lat.reach.get("email", 0.26))
    window_mult = 1.0 if 8 <= hour < 19 else p.off_window_reach_mult
    lang_mult = p.language_match_lift if localized else 1.0

    amount = max(1.0, float(case.get("amount_inr", 0)))
    friction = 1.0 - p.amount_friction * math.log10(amount / p.amount_friction_ref_inr)
    friction = max(0.4, min(1.3, friction))

    if kind in _MECHANICAL_KINDS:
        # Nudging someone with a money problem only converts if the money is
        # there -- or if the link lets them pay from something else.
        availability = 1.0 if _cleared_by(state, day) else NUDGE_UNBLOCK_RATE
    else:
        availability = 1.0   # the customer action *is* the remedy

    raw = (reach * window_mult * lang_mult * p.persuasion_base
           * (1.0 - fix_difficulty) * friction * lat.intent * availability)
    return max(0.0, min(0.95, raw * fatigue))


def apply_touch(state: CaseWorld, intervention: str, channel: str, day: int,
                hour: int, localized: bool, round_index: int) -> bool:
    """Score one touch and fold it into the case's touch history.

    The outcome draw is keyed by (seed, case_id, intervention, round_index) --
    the reproducibility contract this repo has always held.
    """
    prob = touch_success_prob(state, intervention, channel, day, hour, localized)
    u = _unit(state.seed, state.case.get("case_id"), intervention, round_index)
    key = fatigue_key(intervention, channel)
    state.touches_by_key[key] = state.touches_by_key.get(key, 0) + 1
    state.touches_total += 1
    return u < prob


# ---------------------------------------------------------------------------
# organic self-cure -- the do-nothing control, and the background for every arm
# ---------------------------------------------------------------------------
def organic_recovery_day(state: CaseWorld, touch_days: list[int]) -> int | None:
    """Day the customer recovers themselves, unprompted, or None.

    `touch_days` is this arm's contact schedule: once a case has been
    over-contacted, its organic self-cure hazard is damped, so an arm that spams
    can measurably destroy value that doing nothing would have captured.
    """
    p = state.params
    lat = state.latents
    base = p.organic_retry_rate * (
        p.organic_intent_weight * lat.intent + (1.0 - p.organic_intent_weight))
    ordered = sorted(touch_days)
    for d in range(1, p.horizon_days + 1):
        if state.clear_day is None or state.clear_day > d:
            continue
        touched_before = sum(1 for t in ordered if t <= d)
        hazard = base
        if touched_before > p.churn_touch_threshold:
            hazard *= p.churn_organic_mult
        if _unit(state.seed, "organic", state.case.get("case_id"), d) < hazard:
            return d
    return None


# ---------------------------------------------------------------------------
# diagnostics: what does the world actually reward, and does it agree with the
# agent's priors? (used by --check-circularity and --prior-mismatch)
# ---------------------------------------------------------------------------
def expected_rate(reason: str, intervention: str, params: SimParams,
                  n_samples: int = 400, day: int = 2, diag_seed: int = 7717) -> float:
    """Monte-Carlo E[P(success)] for one (reason, intervention) over synthetic
    customers. Purely a reporting aid -- the simulator never consults it."""
    from punar.core.select import CHANNEL_MAP
    ch = CHANNEL_MAP.get(intervention, "email")
    total = 0.0
    for i in range(n_samples):
        case = {"case_id": f"diag-{reason}-{i}", "customer_id": f"diagcust-{i}",
                "reason": reason, "amount_inr": 2499.0, "day_of_month": 10 + (i % 20),
                "hour": 11, "language": "en", "opted_out": False}
        st = CaseWorld.build(case, diag_seed + i, params)
        total += touch_success_prob(st, intervention, ch, day, 11, localized=True)
    return total / max(1, n_samples)


def world_ranking(params: SimParams, n_samples: int = 400) -> dict[str, list[tuple[str, float]]]:
    """For each reason, the interventions the world actually rewards, best first."""
    from punar.core.select import PRIORS
    out: dict[str, list[tuple[str, float]]] = {}
    for reason in sorted(params.blockers):
        cands = sorted(PRIORS.get(reason, {}))
        if not cands:
            continue
        scored = [(iv, expected_rate(reason, iv, params, n_samples)) for iv in cands]
        scored.sort(key=lambda x: (-x[1], x[0]))
        out[reason] = scored
    return out


def prior_ranking() -> dict[str, list[tuple[str, float]]]:
    """The agent's belief, best first -- prior mean alpha/(alpha+beta)."""
    from punar.core.select import PRIORS
    out: dict[str, list[tuple[str, float]]] = {}
    for reason in sorted(PRIORS):
        scored = [(iv, a / (a + b)) for iv, (a, b) in sorted(PRIORS[reason].items())]
        scored.sort(key=lambda x: (-x[1], x[0]))
        out[reason] = scored
    return out


def circularity_report(params: SimParams, n_samples: int = 400) -> dict[str, Any]:
    """How much does the agent's prior agree with the world's argmax?

    A high number here is the thing a reviewer should be suspicious of. It is
    reported, not hidden.
    """
    world = world_ranking(params, n_samples)
    prior = prior_ranking()
    rows = []
    agree = 0
    considered = 0
    for reason in sorted(set(world) & set(prior)):
        w, pr = world[reason], prior[reason]
        if not w or not pr:
            continue
        considered += 1
        match = w[0][0] == pr[0][0]
        agree += 1 if match else 0
        rows.append({"reason": reason, "world_best": w[0][0], "world_p": round(w[0][1], 4),
                     "prior_best": pr[0][0], "prior_mean": round(pr[0][1], 4),
                     "argmax_agrees": match})
    return {"rows": rows, "reasons_considered": considered, "argmax_agreements": agree,
            "argmax_agreement_rate": round(agree / considered, 4) if considered else 0.0}


def perturb_world(params: SimParams, divergence: float, seed: int = 991) -> SimParams:
    """Move the WORLD away from the agent's priors by `divergence` (0 = none).

    Used by the prior-mismatch experiment. The agent's priors are left exactly as
    shipped; the world's mechanism parameters -- channel reach, self-clear
    hazards, remedy difficulty, manual resolution -- are jittered
    multiplicatively. At high divergence the channel ordering itself flips, so
    the agent's shipped beliefs are simply wrong and it has to learn.
    """
    if divergence <= 0:
        return params
    r = _rng(seed, "perturb", round(divergence, 6))
    reach = {}
    for ch in sorted(params.channel_reach):
        if ch == "none":
            reach[ch] = 1.0
            continue
        reach[ch] = min(0.98, max(0.02,
                                  params.channel_reach[ch] * math.exp(divergence * 1.25 * r.gauss(0, 1))))
    blockers = {}
    for reason in sorted(params.blockers):
        kind, sc, fd, mr = params.blockers[reason]
        sc2 = min(0.95, max(0.0, sc * math.exp(divergence * 1.0 * r.gauss(0, 1))))
        fd2 = min(0.99, max(0.01, fd * math.exp(divergence * 0.6 * r.gauss(0, 1))))
        mr2 = min(0.90, max(0.0, mr * math.exp(divergence * 1.0 * r.gauss(0, 1))))
        blockers[reason] = (kind, sc2, fd2, mr2)
    return params.with_overrides({})._replace_tables(blockers, reach)
