"""Recovery-agent state machine.

Pure-Python runner + optional LangGraph wiring. Both paths share the SAME node
functions and router logic, so the offline demo stays deterministic/reviewable
while production can run the compiled graph with checkpointing.

Control flow: diagnose -> guard -> plan -> act -> [engine observes] ->
observe -> decide -> {guard, plan, terminal}. Every transition is appended to
an append-only audit trail.
"""
from collections.abc import Callable
from datetime import datetime
from typing import Any

from punar.core.classify import enrich
from punar.core.copy import generate_copy
from punar.core.gate import (
    allow_touch,
    channel_enabled,
    contacts_customer,
    next_contact_window,
)
from punar.core.select import (
    ABSTAIN,
    CHANNEL_MAP,
    ESCALATION_INTERVENTIONS,
    SILENT_RETRY_INTERVENTIONS,
    default_candidates,
    rank_intervention,
    update_arm,
)

NODES = ["diagnose", "guard", "plan", "act", "observe", "decide", "terminal"]


# ---------------------------- state helpers ---------------------------------
def empty_state(case: dict[str, Any], policy: dict[str, Any],
                rng: Any, now: datetime) -> dict[str, Any]:
    return {
        "case": dict(case), "policy": policy, "rng": rng, "now": now,
        "step": "diagnose", "diagnosis": None, "candidates": [], "chosen": None,
        "plan_records": [], "pending_action": None, "touch_history": [],
        "bandit_arms": {}, "arm_log": [], "audit": [], "outcome": None,
        "blocked_actions": [], "escalations": [],
        "exit_code": None, "terminated": False, "next_step": "guard",
        "simulated_outcome": None, "generated_copy": None, "context": _build_context(case),
    }


def _build_context(case: dict[str, Any]) -> dict[str, Any]:
    return {"amount_inr": float(case.get("amount_inr", 0)),
            "reason": case.get("punar_reason") or case.get("reason"),
            "customer_id": case.get("customer_id"), "case_id": case.get("case_id"),
            "day_of_month": int(case.get("day_of_month", 15)),
            "hour": int(case.get("hour", 10)), "method": case.get("method", ""),
            "touches": list(case.get("touches", []))}


def _audit(state: dict[str, Any], node: str, detail: dict[str, Any]) -> None:
    state["audit"].append({"step": node, "at": state["now"].isoformat(),
                           "case_id": state["case"].get("case_id"), "detail": detail})


# --------------------------------- nodes ------------------------------------
def diagnose(state: dict[str, Any]) -> dict[str, Any]:
    enriched = enrich(state["case"])
    state["case"] = enriched
    state["context"]["reason"] = enriched["punar_reason"]
    state["diagnosis"] = {"reason": enriched["punar_reason"], "label": enriched["punar_label"],
                          "retriability": enriched["punar_retriability"],
                          "suggested_action": enriched["punar_suggested_action"],
                          "retry_limit": enriched["punar_retry_limit"]}
    _audit(state, "diagnose", state["diagnosis"])
    state["step"] = "guard"
    return state


def guard(state: dict[str, Any]) -> dict[str, Any]:
    """Filter reason-plausible interventions through channel + policy gates."""
    policy, now, case = state["policy"], state["now"], state["case"]
    reason = state["diagnosis"]["reason"]
    allowed, last_verdict = [], None
    for iv in default_candidates(reason):
        ch = CHANNEL_MAP.get(iv, "")
        if ch and not channel_enabled(policy, ch):
            last_verdict = {"intervention": iv, "allowed": False, "code": "channel_disabled"}
            continue
        # Non-contacting channels (silent retry, internal escalation) are gated
        # too -- allow_touch reads their per-channel `exempt_from` list, so the
        # exemptions live in policy.json rather than in a branch here.
        verdict = allow_touch(case, policy, now, planned_channel=ch)
        if verdict.allowed:
            allowed.append(iv)
        else:
            last_verdict = {"intervention": iv, "allowed": False, "code": verdict.code}
    state["candidates"] = allowed
    state["last_gate_verdict"] = last_verdict
    _audit(state, "guard", {"allowed": allowed, "verdict": last_verdict})
    state["step"] = "plan" if allowed else "decide"
    return state


def plan(state: dict[str, Any]) -> dict[str, Any]:
    rng, reason = state["rng"], state["diagnosis"]["reason"]
    # Pass the live arms back in so a posterior updated this episode survives
    # into the next round instead of being re-seeded from the prior.
    winner, arms, records = rank_intervention(state["candidates"], state["context"],
                                              state["policy"], rng, reason,
                                              arms=state.get("bandit_arms") or None)
    state["chosen"] = winner
    state["plan_records"] = records
    state["bandit_arms"] = {a.name: a for a in arms}
    _audit(state, "plan", {"chosen": winner, "scores": records})
    if winner == ABSTAIN:
        # Every available action destroys value: decline to act rather than
        # spend a touch to chase a case that is not worth chasing.
        state["outcome"] = "written_off"
        state["exit_code"] = "abstained_below_value_floor"
        state["terminated"] = True
        state["step"] = "terminal"
        return state
    state["step"] = "act"
    return state


def act(state: dict[str, Any]) -> dict[str, Any]:
    """Render the action and dispatch it -- unless the policy judge blocks it.

    A rejected message is NOT sent. It never becomes a touch, never counts
    toward any cap, and never reports itself as delivered; the case is routed
    to human escalation instead. That is what makes the pre-send judge a
    control rather than a log line.
    """
    now, policy = state["now"], state["policy"]
    iv, arm = state["chosen"], state["bandit_arms"][state["chosen"]]
    channel = arm.channel or ""
    reaches_customer = contacts_customer(policy, channel) if channel else False
    lang = str(state["case"].get("language") or "en")[:10]
    reason = state["diagnosis"]["reason"]
    amount = state["context"]["amount_inr"]
    merchant = str(state["case"].get("merchant_name") or "merchant")
    link = state["case"].get("payment_link") or f"https://razorpay.me/pay/{state['case'].get('case_id')}"

    copy_text, copy_ok = "", True
    violations: list[str] = []
    if reaches_customer:
        copy_text, copy_ok, violations = generate_copy(
            iv, reason, lang, merchant=merchant, amount=f"{amount:.0f}", link=link,
            policy=policy)
        if not copy_ok:
            # Localized template tripped the judge -> retry the vetted English
            # fallback before giving up on contacting this customer at all.
            copy_text, copy_ok, violations = generate_copy(
                iv, reason, "en", merchant=merchant, amount=f"{amount:.0f}", link=link,
                policy=policy)

    state["generated_copy"] = {"intervention": iv, "language": lang, "text": copy_text,
                               "judge_allowed": copy_ok, "violations": violations}

    if reaches_customer and not copy_ok:
        blocked = {"round": len(state["touch_history"]) + 1, "intervention": iv,
                   "channel": channel, "timestamp": now.isoformat(),
                   "copy": copy_text, "violations": list(violations),
                   "blocked_by": "policy_judge"}
        state["blocked_actions"].append(blocked)
        state["escalations"].append(dict(blocked, escalated_to="human_review"))
        state["pending_action"] = None
        state["simulated_outcome"] = None
        _audit(state, "act", {"action": iv, "channel": channel, "sent": False,
                              "copy_judge_allowed": False, "violations": violations,
                              "detail": "blocked pre-send by policy judge; escalated to human review"})
        state["outcome"] = "written_off"
        state["exit_code"] = "blocked_by_policy_judge"
        state["terminated"] = True
        state["step"] = "terminal"
        return state

    touch = {"round": len(state["touch_history"]) + 1, "intervention": iv,
             "channel": channel, "cost_inr": arm.cost_inr,
             "timestamp": now.isoformat(), "date": now.date().isoformat(),
             "copy": copy_text, "copy_judge_allowed": copy_ok,
             "contacts_customer": reaches_customer,
             # Honest dispatch state. A real channel provider overwrites this
             # with the delivery receipt; offline it stays 'simulated'.
             "delivered": True, "delivery_status": "simulated"}
    if iv in ESCALATION_INTERVENTIONS:
        touch["delivery_status"] = "queued_for_human_review"
        state["escalations"].append(dict(touch, escalated_to="human_review"))
    elif iv in SILENT_RETRY_INTERVENTIONS:
        touch["delivery_status"] = "reattempted_at_psp"

    state["touch_history"].append(touch)
    state["case"]["touches"] = state["case"].get("touches", []) + [touch]
    # Keep the ranker's annoyance term honest: it reads context["touches"],
    # which must grow as the episode spends the customer's attention.
    state["context"]["touches"] = list(state["case"]["touches"])
    state["pending_action"] = touch
    _audit(state, "act", {"action": iv, "channel": channel, "sent": True,
                          "contacts_customer": reaches_customer,
                          "cost_inr": arm.cost_inr, "copy_judge_allowed": copy_ok,
                          "violations": violations})
    state["step"] = "observe"
    return state


def observe(state: dict[str, Any]) -> dict[str, Any]:
    """Consume an injected outcome and update the selected bandit arm."""
    outcome = state.get("simulated_outcome")
    arm = state["bandit_arms"].get(state["chosen"])
    record = {}
    if arm is not None and isinstance(outcome, bool):
        ctx = dict(state["context"], touches=list(state["case"].get("touches", [])))
        record = update_arm(state["bandit_arms"], state["chosen"], outcome, ctx,
                            policy=state.get("policy"))
        state["arm_log"].append(record)
    if outcome is True:
        state["outcome"] = "recovered"
        state["exit_code"] = "recovered_by_" + str(state["chosen"])
        state["terminated"] = True
    elif outcome is False and not state["candidates"]:
        state["outcome"] = "written_off"
        state["exit_code"] = "exhausted_no_allowed_actions"
        state["terminated"] = True
    else:
        state["exit_code"] = "retry_later"
    _audit(state, "observe", {"simulated_outcome": outcome, "arm_update": record,
                              "outcome": state["outcome"]})
    state["step"] = "decide"
    return state


def decide(state: dict[str, Any]) -> dict[str, Any]:
    """Terminal if recovered/exhausted; else re-enter guard for the next round."""
    if state["terminated"]:
        state["step"] = "terminal"
        _audit(state, "decide", {"terminated": True, "outcome": state["outcome"]})
        return state
    diag = state.get("diagnosis") or {}
    touches = state.get("touch_history") or []
    policy = state.get("policy") or {}
    g = policy.get("guardrails", {})
    # The retry budget comes from the taxonomy for this decline reason; the
    # outreach budget caps how many of those attempts may reach the customer.
    retry_limit = int(diag.get("retry_limit", 3))
    max_actions = max(retry_limit, 1) + 1
    outreach_cap = int(g.get("max_outreach_touches", 3))
    outreach = [t for t in touches if t.get("contacts_customer", True)]
    if len(outreach) >= outreach_cap:
        state["outcome"] = "written_off"
        state["exit_code"] = "outreach_budget_exhausted"
        state["terminated"] = True
        state["step"] = "terminal"
        _audit(state, "decide", {"outreach": len(outreach), "outreach_cap": outreach_cap})
        return state
    if not state["candidates"]:
        state["outcome"] = "written_off"
        state["exit_code"] = "no_allowed_actions"
        state["terminated"] = True
        state["step"] = "terminal"
        _audit(state, "decide", {"reason": "no_guardrail_approved_interventions"})
        return state
    if len(touches) >= max_actions:
        state["outcome"] = "written_off"
        state["exit_code"] = "max_actions_reached"
        state["terminated"] = True
        state["step"] = "terminal"
        _audit(state, "decide", {"touches": len(touches), "max_actions": max_actions})
        return state
    state["next_step"] = "guard"
    state["step"] = "guard"
    _audit(state, "decide", {"reroute": "guard"})
    return state


def terminal(state: dict[str, Any]) -> dict[str, Any]:
    _audit(state, "terminal", {"outcome": state["outcome"], "exit_code": state["exit_code"],
                               "touches": len(state.get("touch_history", []))})
    return state


_NODE_FN = {
    "diagnose": diagnose, "guard": guard, "plan": plan, "act": act,
    "observe": observe, "decide": decide, "terminal": terminal,
}


class _Runner:
    """Pure-Python fixed-point runner replicating the LangGraph edge logic."""
    MAX_ITER = 128

    def run(self, initial: dict[str, Any]) -> dict[str, Any]:
        st = dict(initial)
        for _ in range(self.MAX_ITER):
            st = step(st)
            if st["step"] == "terminal":
                break
        return st


RUNNER = _Runner()


def step(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the current node and advance the program counter."""
    return _NODE_FN[state["step"]](state)


def _schedule_next_attempt(state: dict[str, Any]) -> datetime:
    """When should the next attempt run?

    Derived from the guardrails and the decline reason, never from a fixed
    constant: wait out the inter-touch gap, apply any per-reason backoff, then
    snap forward into the contact window. Because this respects the real gap
    and the real window, the daily touch cap and the gap rule stay live -- a
    fixed +26h hop would silently place every attempt on its own calendar day
    and neither guardrail could ever bind.
    """
    from datetime import timedelta
    policy = state.get("policy") or {}
    g = policy.get("guardrails", {})
    reason = (state.get("diagnosis") or {}).get("reason", "")
    last = state.get("pending_action") or {}

    gap_minutes = int(g.get("min_inter_touch_minutes", 60))
    retry_cfg = policy.get("retry", {}) or {}
    backoff = retry_cfg.get("backoff_hours", {}) or {}
    hours = float(backoff.get(reason, retry_cfg.get("default_backoff_hours", 24)))

    nxt = state["now"] + timedelta(minutes=gap_minutes) + timedelta(hours=hours)
    # A silent PSP re-presentment does not consume customer attention, so it is
    # not held behind the inter-touch gap -- but it still waits out any
    # reason-specific backoff (e.g. re-present after payday).
    if last and not last.get("contacts_customer", True):
        nxt = state["now"] + timedelta(hours=hours or 1.0)
    return next_contact_window(nxt, policy)


def run_agent(case: dict[str, Any], policy: dict[str, Any], rng: Any,
              simulate: Callable[[dict[str, Any], str, datetime], bool],
              now: datetime | None = None) -> dict[str, Any]:
    """Run the full recovery loop for one case through the state machine.

    `simulate(case, intervention, now)` deterministically returns whether the
    intervention succeeded -- this keeps the agent channel-agnostic so the same
    machine runs against stubs offline and real channels in production.
    """
    if now is None:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    initial = empty_state(case, policy, rng, now)
    return _resume_with_simulation(initial, simulate, now)


def _resume_with_simulation(state: dict[str, Any],
                            simulate: Callable[[dict[str, Any], str, datetime], bool],
                            now: datetime) -> dict[str, Any]:
    """Drive the machine, injecting simulator outcomes at each 'act' node."""
    for _ in range(RUNNER.MAX_ITER):
        if state["step"] == "terminal":
            break
        if state["step"] == "act":
            state = step(state)
            if state["step"] == "terminal":
                # act() declined to send (policy judge blocked the copy). No
                # message left the building, so there is no outcome to observe
                # and nothing to attribute to the arm.
                break
            iv = state["chosen"]
            outcome = simulate(state["case"], iv, now)
            state["simulated_outcome"] = outcome
            state = observe(state)      # updates arm, may terminate on success
            if state["step"] == "terminal":
                break
            state = decide(state)       # reroutes to guard or terminates
            state["now"] = _schedule_next_attempt(state)
            continue
        state = step(state)
    # Cap breach fallback: force a clean terminal state.
    if state["step"] != "terminal":
        state["outcome"] = "written_off"
        state["exit_code"] = "max_rounds_reached"
        state["terminated"] = True
        state["step"] = "terminal"
        _audit(state, "terminal", {"outcome": state["outcome"], "exit_code": state["exit_code"],
                                   "touches": len(state.get("touch_history", []))})
    return state


# --------------------------- runner provenance -------------------------------
# A LangGraph runner used to live here. It was removed rather than shipped,
# because it never worked: `graph.invoke()` was handed the *function* returned
# by the outcome-injection helper instead of a state dict, and that helper ran
# the whole loop internally anyway -- so the graph was a no-op wrapper around
# the pure-Python runner. `langgraph` is not a runtime dependency and the path
# had no test, so the failure was invisible.
#
# The state machine above is deliberately a plain fixed-point loop over pure
# node functions: it is deterministic, inspectable, and every transition is
# audited. Porting it to LangGraph is mechanical (the nodes and the router are
# already separated for exactly that) and worth doing when durable
# checkpointing across a multi-day retry schedule is needed -- see
# docs/architecture.md. Until then this reports what actually runs.

RUNNER_NAME = "pure-python"


def runner_name() -> str:
    """Which execution engine actually ran this agent."""
    return RUNNER_NAME
