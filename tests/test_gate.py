"""Guardrails: quiet hours, caps, opt-outs, non-retriable blocks."""
from datetime import datetime
from zoneinfo import ZoneInfo

from punar.core.gate import allow_touch, load_policy
from punar.core.taxonomy import NON_RETRYABLE, RETRIABLE

POLICY = load_policy("punar/config/policy.json")
IST = ZoneInfo("Asia/Kolkata")


def _case(**kw):
    c = {"case_id": "c1", "punar_retriability": RETRIABLE, "touches": [], "active": True}
    c.update(kw)
    return c


def test_allowed_in_window():
    v = allow_touch(_case(), POLICY, datetime(2026, 8, 28, 10, 0, tzinfo=IST))
    assert v.allowed is True


def test_blocked_outside_window_before():
    v = allow_touch(_case(), POLICY, datetime(2026, 8, 28, 6, 59, tzinfo=IST))
    assert not v.allowed and v.code == "outside_contact_window"


def test_blocked_outside_window_after():
    v = allow_touch(_case(), POLICY, datetime(2026, 8, 28, 19, 0, tzinfo=IST))
    assert not v.allowed and v.code == "outside_contact_window"


def test_opt_out_always_wins():
    for hour in (10, 14):
        v = allow_touch(_case(opted_out=True), POLICY, datetime(2026, 8, 28, hour, 0, tzinfo=IST))
        assert not v.allowed and v.code == "customer_opted_out"


def test_non_retriable_blocked():
    v = allow_touch(_case(punar_retriability=NON_RETRYABLE), POLICY,
                    datetime(2026, 8, 28, 10, 0, tzinfo=IST))
    assert not v.allowed and "non_retriable" in v.code


def test_daily_touch_cap():
    touches = [{"timestamp": "2026-08-28T09:00:00+05:30", "date": "2026-08-28"},
               {"timestamp": "2026-08-28T11:00:00+05:30", "date": "2026-08-28"},
               {"timestamp": "2026-08-28T13:00:00+05:30", "date": "2026-08-28"}]
    v = allow_touch(_case(touches=touches), POLICY, datetime(2026, 8, 28, 14, 0, tzinfo=IST))
    assert not v.allowed and v.code == "daily_touch_cap"


def test_inter_touch_gap():
    touches = [{"timestamp": "2026-08-28T10:00:00+05:30", "date": "2026-08-28"}]
    v = allow_touch(_case(touches=touches), POLICY, datetime(2026, 8, 28, 10, 30, tzinfo=IST))
    assert not v.allowed and v.code == "inter_touch_gap"
    v2 = allow_touch(_case(touches=touches), POLICY, datetime(2026, 8, 28, 12, 0, tzinfo=IST))
    assert v2.allowed


def test_gap_respects_timezone():
    # 04:00 UTC == 09:30 IST -> should be inside window and >60m after a 08:00 IST touch
    touches = [{"timestamp": "2026-08-28T08:00:00+05:30", "date": "2026-08-28"}]
    v = allow_touch(_case(touches=touches), POLICY, datetime(2026, 8, 28, 4, 30, tzinfo=ZoneInfo("UTC")))
    assert v.allowed, v.code


def test_daily_cost_cap():
    touches = [{"timestamp": "2026-08-28T09:00:00+05:30", "date": "2026-08-28", "cost_inr": 499.6}]
    v = allow_touch(_case(touches=touches), POLICY, datetime(2026, 8, 28, 10, 0, tzinfo=IST),
                    planned_channel="whatsapp")
    assert not v.allowed and v.code == "daily_cost_cap"
