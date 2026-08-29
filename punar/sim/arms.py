"""Comparator arms.

Every arm is a *policy*: given a case, it emits a schedule of touches. It never
decides whether a touch worked -- that is :mod:`punar.sim.world`'s job, and the
same function scores all of them. An arm's only levers are which interventions
it picks, on which channel, on which day, at which hour, and in which language.

The arms, weakest to strongest:

``do_nothing``
    No contact at all. Measures organic self-cure -- how many of these payments
    come back with zero intervention. Without this control, no other arm's
    absolute recovery rate means anything.

``naive``
    The textbook straw man, and it is labelled as one: two instant retries fired
    seconds after the failure, then two generic English emails, no suppression
    list, no opt-out check, no dead-instrument check. Below current merchant
    practice. Retained only so the delta against the *realistic* arm is legible.

``realistic``
    Current merchant practice. Scheduled issuer-aware retries rather than instant
    ones (Razorpay ships Smart Retries), a suppression list honouring opt-outs
    and obviously-dead instruments, a multi-channel WhatsApp -> email sequence in
    business hours, vernacular copy. This is the comparator that matters.

``taxonomy_only`` / ``taxonomy_guardrails``
    Ablation rungs: the decline taxonomy routing on its own, then the same
    routing with the guardrail gate wired in, both without the bandit.

``punar``
    The full system, run through the real agent state machine.
"""
import hashlib
import random
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from punar.core import agent as agent_mod
from punar.core.gate import allow_touch, channel_enabled
from punar.core.select import CHANNEL_MAP, PRIORS
from punar.core.taxonomy import NON_RETRYABLE, get_reason
from punar.sim.params import SimParams
from punar.sim.world import MECHANISM, CaseWorld, apply_touch, organic_recovery_day

# Retry variants. All three are the same *mechanism* (a card/UPI presentment);
# they differ only in WHEN they land, which is the whole point -- timing is a
# policy choice any arm may make, not a property of who is making it.
MECHANISM.setdefault("instant_retry", "retry")
MECHANISM.setdefault("scheduled_retry", "retry")
RETRY_CHANNEL = {"instant_retry": "none", "scheduled_retry": "none",
                 "silent_retry_aligned": "none"}

# Cost basis. Identical for every arm: channel costs come from policy.json, and
# these two cover the actions policy.json has no channel for.
MANUAL_REVIEW_COST_INR = 45.0
"""Loaded cost of one human ops touch on an escalated case."""
RETRY_ATTEMPT_COST_INR = 0.0
"""Gateway cost of a failed re-presentment. Zero on Razorpay today; named so it
can be set non-zero for schemes that charge for declines."""

BUSINESS_HOUR = 11          # default landing hour for scheduled outreach (IST)
PAYDAY_EVENING_HOUR = 19    # landing hour for payday-aligned retries

# A merchant without a decline taxonomy still hard-suppresses the two codes the
# card networks require you to stop retrying.
NETWORK_SUPPRESSED = ("account_closed", "lost_stolen_card")


# ---------------------------------------------------------------------------
# touch plumbing
# ---------------------------------------------------------------------------
def _touch(round_index: int, intervention: str, channel: str, day: int, hour: int,
           localized: bool, ok: bool, cost: float) -> dict[str, Any]:
    return {"round": round_index, "intervention": intervention, "channel": channel,
            "day": day, "hour": hour, "localized": localized, "ok": ok,
            "cost_inr": round(cost, 4)}


def _reaches_customer(touch: dict[str, Any]) -> bool:
    """True when a touch actually consumed the customer's attention.

    Keyed off the same MECHANISM map that decides cost, so contact accounting
    and cost accounting can never disagree: a silent PSP re-presentment and an
    internal escalation are real recovery work, but neither reaches the
    customer and neither can annoy them.
    """
    if "contacts_customer" in touch:
        return bool(touch["contacts_customer"])
    iv, ch = touch.get("intervention", ""), touch.get("channel") or ""
    mech = MECHANISM.get(iv, "contact" if ch not in ("none", "") else "manual")
    return mech == "contact"


def touch_cost(intervention: str, channel: str, policy: dict[str, Any]) -> float:
    """One cost basis for every arm."""
    mech = MECHANISM.get(intervention, "contact" if channel not in ("none", "") else "manual")
    if mech == "retry":
        return RETRY_ATTEMPT_COST_INR
    if mech == "manual":
        return MANUAL_REVIEW_COST_INR
    return float(policy.get("channels", {}).get(channel, {}).get("cost_inr", 0.0))


def _localized(case: dict[str, Any], message_language: str) -> bool:
    """Did the message land in the customer's own language?"""
    return str(case.get("language") or "en") == message_language


def _next_payday_day(case: dict[str, Any], from_day: int, params: SimParams) -> int:
    """First day >= from_day whose calendar date falls in a payday window."""
    dom0 = int(case.get("day_of_month", 15))
    for d in range(max(from_day, 1), params.horizon_days + 1):
        if ((dom0 + d - 1) % 31) + 1 in params.payday_days_of_month:
            return d
    return min(max(from_day, 1), params.horizon_days)


def _landing(case: dict[str, Any], intervention: str, day_hint: int,
             params: SimParams) -> tuple[int, int]:
    """(day, hour) a given intervention actually lands on.

    This is part of the *definition of the action*, not of the arm: any policy
    that selects ``silent_retry_aligned`` gets payday-evening placement, and any
    policy that selects ``instant_retry`` gets day 0.
    """
    if intervention == "instant_retry":
        return 0, int(case.get("hour", 10))
    if intervention == "silent_retry_aligned":
        return _next_payday_day(case, max(day_hint, 1), params), PAYDAY_EVENING_HOUR
    if intervention == "scheduled_retry":
        return max(day_hint, 1), BUSINESS_HOUR
    return max(day_hint, 0), BUSINESS_HOUR


# ---------------------------------------------------------------------------
# scripted arms
# ---------------------------------------------------------------------------
def _run_script(case: dict[str, Any], seed: int, params: SimParams,
                policy: dict[str, Any], script: list[dict[str, Any]],
                arm: str) -> dict[str, Any]:
    """Execute a fixed touch schedule. `script` entries are
    {intervention, day, hour?, message_language?}."""
    state = CaseWorld.build(case, seed, params)
    touches: list[dict[str, Any]] = []
    recovered_by = None
    for i, step in enumerate(script, 1):
        iv = step["intervention"]
        ch = RETRY_CHANNEL.get(iv) or CHANNEL_MAP.get(iv, "email")
        day, hour = _landing(case, iv, int(step.get("day", 0)), params)
        if "hour" in step:
            hour = int(step["hour"])
        if day > params.horizon_days:
            break
        loc = _localized(case, str(step.get("message_language", "en")))
        ok = apply_touch(state, iv, ch, day, hour, loc, i)
        touches.append(_touch(i, iv, ch, day, hour, loc, ok, touch_cost(iv, ch, policy)))
        if ok:
            recovered_by = iv
            break
    return _finalize(case, state, touches, arm, recovered_by,
                     exit_code=("recovered_by_" + recovered_by) if recovered_by else "not_recovered")


def do_nothing_script(case: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def naive_script(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Two instant retries, then two generic English emails. No suppression."""
    return [
        {"intervention": "instant_retry", "day": 0},
        {"intervention": "instant_retry", "day": 0},
        {"intervention": "email_payment_link", "day": 0, "hour": int(case.get("hour", 10)),
         "message_language": "en"},
        {"intervention": "email_payment_link", "day": 1, "hour": BUSINESS_HOUR,
         "message_language": "en"},
    ]


def realistic_script(case: dict[str, Any], localize: bool = True) -> list[dict[str, Any]]:
    """Current merchant practice: suppression list + scheduled retries +
    multi-channel dunning in business hours."""
    reason = case.get("reason", "bank_decline_general")
    if reason in NETWORK_SUPPRESSED:
        return []                                   # network mandate: stop retrying
    lang = str(case.get("language") or "en") if localize else "en"
    contactable = not case.get("opted_out")         # suppression list
    script: list[dict[str, Any]] = [
        {"intervention": "scheduled_retry", "day": 1},
    ]
    if contactable:
        script.append({"intervention": "whatsapp_nudge_payment_link", "day": 2,
                       "hour": BUSINESS_HOUR, "message_language": lang})
    script.append({"intervention": "scheduled_retry", "day": 3})
    if contactable:
        script.append({"intervention": "email_payment_link", "day": 5,
                       "hour": BUSINESS_HOUR, "message_language": "en"})
    return script


# ---------------------------------------------------------------------------
# ablation arms: taxonomy routing, with and without guardrails
# ---------------------------------------------------------------------------
def _taxonomy_order(reason: str) -> list[str]:
    """Interventions for a reason, ordered by prior mean. Deterministic: no
    bandit sampling, no exploration -- this is routing only."""
    table = PRIORS.get(reason) or PRIORS["bank_decline_general"]
    return [iv for iv, _ in sorted(table.items(), key=lambda kv: (-(kv[1][0] / (kv[1][0] + kv[1][1])), kv[0]))]


def _run_taxonomy(case: dict[str, Any], seed: int, params: SimParams,
                  policy: dict[str, Any], arm: str, guardrails: bool,
                  now) -> dict[str, Any]:
    from punar.core.classify import enrich
    enriched = enrich(dict(case))
    reason = enriched["punar_reason"]
    # Mirror the agent's own channel gate so the ablation ladder is
    # apples-to-apples with the full system.
    order = [iv for iv in _taxonomy_order(reason)
             if channel_enabled(policy, RETRY_CHANNEL.get(iv) or CHANNEL_MAP.get(iv, "email"))]
    max_touches = max(int(enriched.get("punar_retry_limit", 3)), 2) + 2

    state = CaseWorld.build(case, seed, params)
    touches: list[dict[str, Any]] = []
    gate_case = dict(enriched)
    gate_case["touches"] = []
    recovered_by, exit_code = None, "not_recovered"
    clock = now
    for i in range(1, max_touches + 1):
        if not order:
            exit_code = "no_enabled_channels"
            break
        iv = order[(i - 1) % len(order)]
        ch = RETRY_CHANNEL.get(iv) or CHANNEL_MAP.get(iv, "email")
        if guardrails:
            verdict = allow_touch(gate_case, policy, clock, planned_channel=ch)
            if not verdict.allowed and not (ch in ("none", "") and verdict.code == "customer_opted_out"):
                exit_code = "gate_" + verdict.code
                break
        day, hour = _landing(case, iv, i - 1, params)
        if day > params.horizon_days:
            exit_code = "horizon_reached"
            break
        loc = _localized(case, str(case.get("language") or "en"))
        ok = apply_touch(state, iv, ch, day, hour, loc, i)
        cost = touch_cost(iv, ch, policy)
        touches.append(_touch(i, iv, ch, day, hour, loc, ok, cost))
        gate_case["touches"] = gate_case["touches"] + [
            {"timestamp": clock.isoformat(), "date": clock.date().isoformat(), "cost_inr": cost}]
        clock = clock + timedelta(hours=26)
        if ok:
            recovered_by, exit_code = iv, "recovered_by_" + iv
            break
    return _finalize(case, state, touches, arm, recovered_by, exit_code)


# ---------------------------------------------------------------------------
# the full agent
# ---------------------------------------------------------------------------
def run_punar(case: dict[str, Any], seed: int, params: SimParams,
              policy: dict[str, Any], now, arm: str = "punar") -> dict[str, Any]:
    digest = int.from_bytes(
        hashlib.sha256(f"{seed}|{case['case_id']}".encode()).digest()[:8], "big")
    rng = random.Random(digest)
    state = CaseWorld.build(case, seed, params)
    counter = {"n": 0}

    def simulate(case_state: dict[str, Any], intervention: str, _now) -> bool:
        counter["n"] += 1
        i = counter["n"]
        # The agent advances its own clock 26h per touch; read the clock off the
        # touch it just recorded rather than the stale closure argument.
        hist = case_state.get("touches") or []
        ts = hist[-1].get("timestamp") if hist else None
        day_hint = i - 1
        hour = BUSINESS_HOUR
        if ts:
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(ts)
            day_hint = max(0, (dt.date() - now.date()).days)
            hour = dt.hour
        ch = RETRY_CHANNEL.get(intervention) or CHANNEL_MAP.get(intervention, "email")
        day, land_hour = _landing(case_state, intervention, day_hint, params)
        if intervention not in RETRY_CHANNEL:
            land_hour = hour
        if day > params.horizon_days:
            return False
        # Punar renders EN/Hindi/Hinglish copy per customer, so its outreach is
        # always in the customer's own language.
        ok = apply_touch(state, intervention, ch, day, land_hour, True, i)
        state.extra_days.append(day)
        return ok

    agent_state = agent_mod.run_agent(case, policy, rng, simulate, now=now)

    arm_log = agent_state.get("arm_log", [])
    touches: list[dict[str, Any]] = []
    days = state.extra_days
    for i, t in enumerate(agent_state.get("touch_history", []), 1):
        obs = arm_log[i - 1] if i - 1 < len(arm_log) else {}
        iv = t["intervention"]
        ch = RETRY_CHANNEL.get(iv) or CHANNEL_MAP.get(iv, t.get("channel") or "email")
        day = days[i - 1] if i - 1 < len(days) else i - 1
        touches.append(_touch(i, iv, ch, day, BUSINESS_HOUR, True,
                              bool(obs.get("success", False)),
                              touch_cost(iv, ch, policy)))
    recovered_by = agent_state.get("chosen") if agent_state.get("outcome") == "recovered" else None
    return _finalize(case, state, touches, arm, recovered_by,
                     agent_state.get("exit_code") or "not_recovered",
                     runner=agent_mod.runner_name())


# ---------------------------------------------------------------------------
# shared row finalisation: organic self-cure, costs, net revenue
# ---------------------------------------------------------------------------
def _finalize(case: dict[str, Any], state: CaseWorld, touches: list[dict[str, Any]],
              arm: str, recovered_by: str | None, exit_code: str,
              runner: str = "pure-python") -> dict[str, Any]:
    params = state.params
    meta = get_reason(case.get("reason", ""))
    touch_days = [int(t["day"]) for t in touches]
    organic_day = organic_recovery_day(state, touch_days)

    policy_day = None
    if recovered_by is not None and touches:
        policy_day = int(touches[-1]["day"])

    recovered_via = None
    recovery_day = None
    if policy_day is not None and (organic_day is None or policy_day <= organic_day):
        recovered_via, recovery_day = "policy", policy_day
    elif organic_day is not None:
        recovered_via, recovery_day = "organic", organic_day
        # Touches scheduled after the customer had already paid would not have
        # been sent; do not charge the arm for them.
        touches = [t for t in touches if int(t["day"]) <= organic_day]
        recovered_by = None

    recovered = recovered_via is not None
    # Whether a touch consumed the customer's attention is a property of the
    # channel in policy.json, not of its name -- silent PSP re-presentments and
    # internal escalations never reach the customer and cannot annoy them.
    contact_touches = [t for t in touches if _reaches_customer(t)]
    opt_out_contacts = len(contact_touches) if case.get("opted_out") else 0
    non_retriable_contacts = len(contact_touches) if meta.retriability == NON_RETRYABLE else 0
    unwanted = opt_out_contacts + (0 if case.get("opted_out") else non_retriable_contacts)

    spend = round(sum(float(t["cost_inr"]) for t in touches), 4)
    annoyance = round(unwanted * params.annoyance_inr_per_unwanted_touch, 2)
    gross = float(case.get("amount_inr", 0)) if recovered else 0.0

    return {
        "case_id": case["case_id"],
        "reason": case.get("reason"),
        "retriability": meta.retriability,
        "amount_inr": float(case.get("amount_inr", 0)),
        "opted_out": bool(case.get("opted_out")),
        "policy": arm,
        "recovered": recovered,
        "recovered_via": recovered_via,
        "recovery_day": recovery_day,
        "recovery_intervention": recovered_by,
        "exit_code": exit_code if recovered_via != "organic" else "recovered_organic",
        "touches": touches,
        "touch_count": len(touches),
        "cost_inr": spend,
        "channel_spend_inr": spend,
        "opt_out_violations": opt_out_contacts,
        "non_retriable_touches": non_retriable_contacts,
        "unwanted_touches": unwanted,
        "annoyance_cost_inr": annoyance,
        "gross_revenue_inr": round(gross, 2),
        "net_revenue_inr": round(gross - spend - annoyance, 2),
        "runner": runner,
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def _script_arm(builder: Callable[[dict[str, Any]], list[dict[str, Any]]], name: str):
    def run(case, seed, params, policy, now):
        return _run_script(case, seed, params, policy, builder(case), name)
    return run


ARMS: dict[str, Callable[..., dict[str, Any]]] = {
    "do_nothing": _script_arm(do_nothing_script, "do_nothing"),
    "naive": _script_arm(naive_script, "naive"),
    "realistic": _script_arm(realistic_script, "realistic"),
    "realistic_english": _script_arm(lambda c: realistic_script(c, localize=False),
                                     "realistic_english"),
    "taxonomy_only": lambda case, seed, params, policy, now: _run_taxonomy(
        case, seed, params, policy, "taxonomy_only", guardrails=False, now=now),
    "taxonomy_guardrails": lambda case, seed, params, policy, now: _run_taxonomy(
        case, seed, params, policy, "taxonomy_guardrails", guardrails=True, now=now),
    "punar": lambda case, seed, params, policy, now: run_punar(
        case, seed, params, policy, now),
}

ARM_LABELS = {
    "do_nothing": "Do nothing (organic self-cure control)",
    "naive": "Naive dunning (instant retries + generic email, no guardrails)",
    "realistic": "Realistic merchant baseline (scheduled retries + suppression + multi-channel)",
    "realistic_english": "Realistic baseline, English-only copy",
    "taxonomy_only": "Ablation: taxonomy routing only",
    "taxonomy_guardrails": "Ablation: taxonomy routing + guardrails",
    "punar": "Punar (taxonomy + guardrails + bandit)",
}

ABLATION_LADDER = ["do_nothing", "taxonomy_only", "taxonomy_guardrails", "punar"]
COMPARATORS = ["do_nothing", "naive", "realistic"]


def run_arm(name: str, case: dict[str, Any], seed: int, params: SimParams,
            policy: dict[str, Any], now) -> dict[str, Any]:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name](case, seed, params, policy, now)
