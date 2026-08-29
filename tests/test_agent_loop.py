"""End-to-end agent loop on the state machine."""
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from punar.core.agent import run_agent, runner_name
from punar.core.gate import load_policy

POLICY = load_policy("punar/config/policy.json")
IST = ZoneInfo("Asia/Kolkata")


def _always_true(case, intervention, now):
    return True


def _always_false(case, intervention, now):
    return False


def test_always_recovers_when_simulator_says_yes():
    case = {"case_id": "t1", "amount_inr": 2499.0, "opted_out": False,
            "error": {"code": "INSUFFICIENT_FUNDS", "description": "x"}}
    st = run_agent(case, POLICY, random.Random(7), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    assert st["outcome"] == "recovered"
    assert len(st["touch_history"]) >= 1
    assert any(a["step"] == "act" for a in st["audit"])


def test_non_retriable_never_contacted():
    """A fraud block must reach a human, and must never reach the customer.

    Asserted as "zero CONTACTING touches" rather than "zero touches": the
    escalation is a real recorded action on a non-customer-facing channel, and
    a test that counts all touches would pass even if the non-retriable
    guardrail were deleted outright.
    """
    case = {"case_id": "t2", "amount_inr": 5000.0, "opted_out": False,
            "error": {"code": "FRAUD", "description": "blocked"}, "method": "card"}
    st = run_agent(case, POLICY, random.Random(7), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    contacting = [t for t in st["touch_history"] if t.get("contacts_customer")]
    assert contacting == []
    assert st["escalations"], "a non-retriable decline must be escalated to a human"
    assert all(t["channel"] == "internal_escalation" for t in st["touch_history"])


def test_non_retriable_outreach_is_blocked_by_the_guardrail():
    """The block must come from the gate, not from a disabled channel."""
    from punar.core.gate import allow_touch
    case = {"case_id": "t2b", "amount_inr": 5000.0, "punar_retriability": "non_retryable",
            "punar_label": "Fraud block", "touches": []}
    verdict = allow_touch(case, POLICY, datetime(2026, 8, 28, 10, tzinfo=IST),
                          planned_channel="whatsapp")
    assert not verdict.allowed
    assert verdict.code == "non_retriable_decline"


def test_opted_out_customer_never_contacted():
    case = {"case_id": "t3", "amount_inr": 1499.0, "opted_out": True,
            "error": {"code": "INSUFFICIENT_FUNDS", "description": "x"}}
    st = run_agent(case, POLICY, random.Random(7), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    contacting = [t for t in st["touch_history"] if t.get("contacts_customer")]
    assert contacting == []
    assert st["outcome"] == "written_off"


def test_audit_trail_is_complete():
    case = {"case_id": "t4", "amount_inr": 999.0, "opted_out": False,
            "error": {"code": "UPI_TIMEOUT", "description": "x"}, "method": "upi"}
    st = run_agent(case, POLICY, random.Random(42), _always_false,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    steps = [a["step"] for a in st["audit"]]
    assert steps[0] == "diagnose"
    assert "act" in steps and "observe" in steps and "decide" in steps
    assert st["terminated"] is True


def test_copy_is_policy_judged_before_sending():
    case = {"case_id": "t5", "amount_inr": 2499.0, "opted_out": False,
            "language": "hi", "error": {"code": "INSUFFICIENT_FUNDS", "description": "x"}}
    st = run_agent(case, POLICY, random.Random(1), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    copies = [a["detail"].get("copy_judge_allowed") for a in st["audit"] if a["step"] == "act"]
    assert all(c is True for c in copies) and len(copies) >= 1


def test_bandit_arm_is_updated_after_observation():
    case = {"case_id": "t6", "amount_inr": 2499.0, "opted_out": False,
            "error": {"code": "EXPIRED_CARD", "description": "x"}}
    st = run_agent(case, POLICY, random.Random(99), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    assert len(st.get("arm_log", [])) >= 1
    rec = st["arm_log"][0]
    assert "arm" in rec and "success" in rec and "sampled_p" in rec


def test_runner_provenance_is_reported_honestly():
    """/health and every benchmark row report which engine actually ran.

    Replaces a tautological `isinstance(..., bool)` assertion on a LangGraph
    availability flag whose runner was broken and has since been removed.
    """
    assert runner_name() == "pure-python"


def test_policy_judge_blocks_the_send_and_escalates(monkeypatch):
    """A rejected message must not be sent, counted, or reported as delivered.

    This is the compliance claim the project makes on its front page. The
    original implementation logged the rejection and then appended the touch
    with delivered=True anyway.
    """
    import punar.core.copy as cp
    monkeypatch.setitem(cp.TEMPLATES, "whatsapp_nudge_payment_link",
                        {"en": "Pay now defaulter or we will take legal action. {optout}"})
    case = {"case_id": "t_block", "amount_inr": 5000.0, "language": "en",
            "error": {"code": "INSUFFICIENT_FUNDS", "description": "low balance"},
            "method": "upi"}
    st = run_agent(case, POLICY, random.Random(3), _always_true,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))

    assert st["touch_history"] == [], "a blocked message must never become a touch"
    assert len(st["blocked_actions"]) == 1
    assert st["exit_code"] == "blocked_by_policy_judge"
    assert st["escalations"], "a blocked send must be routed to a human"
    assert "legal_threat" in st["blocked_actions"][0]["violations"]


def test_silent_retry_is_reachable_and_does_not_contact_the_customer():
    """silent_retry_aligned was unreachable: policy.json had no channel for it."""
    case = {"case_id": "t_silent", "amount_inr": 5000.0,
            "error": {"code": "INSUFFICIENT_FUNDS", "description": "low balance"},
            "method": "upi"}
    st = run_agent(case, POLICY, random.Random(3), _always_false,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    used = [t["intervention"] for t in st["touch_history"]]
    assert "silent_retry_aligned" in used
    silent = [t for t in st["touch_history"] if t["intervention"] == "silent_retry_aligned"]
    assert all(t["contacts_customer"] is False for t in silent)
    assert all(t["cost_inr"] == 0.0 for t in silent)


def test_outreach_budget_caps_customer_contact():
    """Silent retries are unlimited-ish; messages to the customer are not."""
    case = {"case_id": "t_cap", "amount_inr": 50000.0,
            "error": {"code": "INSUFFICIENT_FUNDS", "description": "low balance"},
            "method": "upi"}
    st = run_agent(case, POLICY, random.Random(11), _always_false,
                   now=datetime(2026, 8, 28, 10, tzinfo=IST))
    outreach = [t for t in st["touch_history"] if t.get("contacts_customer")]
    cap = POLICY["guardrails"]["max_outreach_touches"]
    assert len(outreach) <= cap
