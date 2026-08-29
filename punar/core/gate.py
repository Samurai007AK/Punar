"""Pre-action guardrails: quiet hours, touch caps, opt-outs, non-retriable blocks.

Every recovery action must pass this gate before it is executed; the verdict
is appended to the audit trail so an RBI / Fair-Practices reviewer can confirm
that no prohibited contact ever left the building.

Two classes of channel exist:

* CONTACTING channels (whatsapp/email/voice/sms) -- every guardrail applies.
* NON-CONTACTING channels (``silent_retry``, ``internal_escalation``) -- these
  never reach the customer, so the rules that exist to protect the customer's
  attention (quiet hours, daily touch cap, inter-touch gap) do not apply. Which
  rules a channel is exempt from is declared in policy.json under
  ``channels.<name>.exempt_from``, never hardcoded here, so a reviewer can read
  the exemptions off the policy file.

All timestamp arithmetic is done in the policy's timezone (IST by default).
Bucketing a touch by the first ten characters of its ISO string would bucket a
20:00 UTC touch into the previous IST day and let a fourth same-day touch
through a cap of three; every timestamp is therefore converted before use.
"""
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

REASON_ALLOWED = "allowed"

# Guardrail identifiers a channel may be exempted from in policy.json.
RULE_OPT_OUT = "opt_out"
RULE_NON_RETRIABLE = "non_retriable"
RULE_QUIET_HOURS = "quiet_hours"
RULE_DAILY_TOUCH_CAP = "daily_touch_cap"
RULE_INTER_TOUCH_GAP = "inter_touch_gap"
RULE_DAILY_COST_CAP = "daily_cost_cap"


class GateVerdict:
    def __init__(self, allowed: bool, code: str, detail: str = ""):
        self.allowed = allowed
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "code": self.code, "detail": self.detail}


def load_policy(path: str = "punar/config/policy.json") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def policy_timezone(policy: dict[str, Any]) -> ZoneInfo:
    cw = policy.get("guardrails", {}).get("contact_window", {})
    return ZoneInfo(cw.get("timezone", "Asia/Kolkata"))


def to_policy_tz(value: Any, tz: ZoneInfo) -> datetime | None:
    """Coerce a timestamp of unknown provenance into the policy timezone.

    Accepts datetimes and ISO-8601 strings. A NAIVE timestamp is interpreted as
    already being in the policy timezone -- Punar's own touches are written with
    an IST offset, so a naive value can only come from an external feed, and
    assuming policy-local is both documented and safer than assuming the host's
    system local time (which would shift the inter-touch gap by the host's UTC
    offset on any non-IST machine). Anything unparseable returns None instead of
    raising: a malformed touch record must never crash the guardrail.
    """
    if value is None:
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:                       # date-only records ("2026-08-28")
                dt = datetime.fromisoformat(text[:10])
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _touches_on_date(touches: list[dict[str, Any]], date_iso: str,
                     tz: ZoneInfo) -> list[dict[str, Any]]:
    """Touches that fall on `date_iso` IN THE POLICY TIMEZONE."""
    out = []
    for t in touches or []:
        if not isinstance(t, dict):
            continue
        local = to_policy_tz(t.get("timestamp") or t.get("date"), tz)
        if local is None:
            continue
        if local.date().isoformat() == date_iso:
            out.append(t)
    return out


def is_contacting_touch(touch: dict[str, Any]) -> bool:
    """A record only consumes attention budget if it reached the customer.

    Silent retries are recorded in the same history (they are real recovery
    attempts and they cost money to reconcile) but they do not count toward the
    daily touch cap or the inter-touch gap, which exist to protect the
    customer's attention. Records that predate the flag default to contacting.
    """
    return bool(touch.get("contacts_customer", True)) and touch.get("delivered", True) is not False


def channel_exemptions(policy: dict[str, Any], channel: str) -> frozenset:
    """Guardrails the given channel is exempt from, per policy.json."""
    cfg = policy.get("channels", {}).get(channel, {})
    return frozenset(cfg.get("exempt_from", []) or [])


def contacts_customer(policy: dict[str, Any], channel: str) -> bool:
    """True unless policy.json marks the channel as non-customer-contacting."""
    cfg = policy.get("channels", {}).get(channel)
    if cfg is None:
        return bool(channel)          # unknown channel -> assume it reaches a human
    return bool(cfg.get("contacts_customer", True))


def allow_touch(case: dict[str, Any], policy: dict[str, Any], now: datetime,
                planned_channel: str = "", planned_cost_inr: float = 0.0) -> GateVerdict:
    g = policy.get("guardrails", {})
    tz = policy_timezone(policy)
    now = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    exempt = channel_exemptions(policy, planned_channel) if planned_channel else frozenset()

    if not case.get("active", True):
        return GateVerdict(False, "case_inactive", "Case already resolved or written off")

    # 1. Opt-out. Non-negotiable for anything that reaches the customer; a
    #    back-office escalation is explicitly exempted in policy.json.
    if (case.get("opted_out") or case.get("dnd")) and RULE_OPT_OUT not in exempt:
        return GateVerdict(False, "customer_opted_out",
                           "Customer has opted out; no further contact permitted")

    # 2. Non-retriable declines are blocked from outreach AND from silent
    #    retries (never re-present a fraud-blocked or closed-account
    #    instrument); only the manual-escalation path is exempt.
    if g.get("block_non_retriable") and case.get("punar_retriability") == "non_retryable" \
            and RULE_NON_RETRIABLE not in exempt:
        return GateVerdict(False, "non_retriable_decline",
                           f"Decline classified '{case.get('punar_label', '')}' is not retryable")

    # 3. Quiet hours (IST business window by default)
    cw = g.get("contact_window", {})
    start_h = int(cw.get("start_hour", 8))
    end_h = int(cw.get("end_hour", 19))
    if RULE_QUIET_HOURS not in exempt and not (start_h <= now.hour < end_h):
        return GateVerdict(
            False, "outside_contact_window",
            f"Contact window is {start_h:02d}:00-{end_h:02d}:00 {tz.key}; "
            f"local time is {now.hour:02d}:{now.minute:02d}")

    # 4. Daily touch cap -- bucketed by the POLICY timezone's calendar day.
    today = now.date().isoformat()
    todays = _touches_on_date(case.get("touches", []), today, tz)
    contacting = [t for t in todays if is_contacting_touch(t)]
    cap = int(g.get("max_touches_per_day", 3))
    if RULE_DAILY_TOUCH_CAP not in exempt and len(contacting) >= cap:
        return GateVerdict(False, "daily_touch_cap",
                           f"Already touched {len(contacting)}/{cap} times today")

    # 5. Minimum inter-touch gap
    min_gap = int(g.get("min_inter_touch_minutes", 60))
    if RULE_INTER_TOUCH_GAP not in exempt and contacting:
        last_ts = None
        for t in contacting:
            ts = to_policy_tz(t.get("timestamp") or t.get("date"), tz)
            if ts is None:
                continue          # malformed record: skip, never crash
            if last_ts is None or ts > last_ts:
                last_ts = ts
        if last_ts is not None:
            gap = (now - last_ts).total_seconds() / 60.0
            if gap < min_gap:
                return GateVerdict(False, "inter_touch_gap",
                                   f"Only {gap:.0f}m since last touch; minimum gap is {min_gap}m")

    # 6. Daily cost cap
    daily_cap = float(g.get("daily_max_cost_inr", 500))
    spent = sum(_as_float(t.get("cost_inr")) for t in todays)
    planned = float(planned_cost_inr) if planned_cost_inr else _channel_cost(policy, planned_channel)
    if RULE_DAILY_COST_CAP not in exempt and spent + planned > daily_cap:
        return GateVerdict(False, "daily_cost_cap",
                           f"Daily spend INR {spent:.2f} + {planned:.2f} would exceed cap INR {daily_cap:.2f}")

    return GateVerdict(True, REASON_ALLOWED, "All guardrails satisfied")


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _channel_cost(policy: dict[str, Any], channel: str) -> float:
    ch = policy.get("channels", {}).get(channel, {})
    return _as_float(ch.get("cost_inr"))


def channel_enabled(policy: dict[str, Any], channel: str) -> bool:
    return bool(policy.get("channels", {}).get(channel, {}).get("enabled", False))


def next_contact_window(now: datetime, policy: dict[str, Any]) -> datetime:
    """Snap `now` forward to the next moment inside the contact window."""
    tz = policy_timezone(policy)
    local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    cw = policy.get("guardrails", {}).get("contact_window", {})
    start_h, end_h = int(cw.get("start_hour", 8)), int(cw.get("end_hour", 19))
    if start_h <= local.hour < end_h:
        return local
    target = local.replace(hour=start_h, minute=0, second=0, microsecond=0)
    if local.hour >= end_h:
        from datetime import timedelta
        target = target + timedelta(days=1)
    return target
