"""Multi-arm benchmark harness.

This module used to score the baseline and Punar through two different branches
of the same function, which is the single worst thing a benchmark can do: it
assigns the lift instead of measuring it. There is now exactly one scoring
function -- :func:`punar.sim.world.touch_success_prob` -- and it takes no
argument identifying which policy produced the touch. Everything an arm gains it
gains by choosing a better action at a better time, on the same world, against
the same fatigue model, on the same cost basis.

What this module does:

* runs any named arm (:mod:`punar.sim.arms`) over a cohort;
* aggregates a run into metrics, including **net** revenue (gross recovered
  minus channel spend minus modelled annoyance cost) alongside gross;
* keeps the legacy ``compare_policies`` two-arm shape so existing callers keep
  working, while exposing every arm under ``result["arms"]``.

Reproducibility contract (unchanged, and load-bearing): every random draw is
derived by SHA-256 from (seed, case_id, purpose, index). Nothing calls
``datetime.now()`` on the benchmark path, nothing uses the unseeded global
``random``, and nothing depends on dict ordering. Results are therefore stable
across processes and across ``PYTHONHASHSEED`` values.
"""
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from punar.core.gate import load_policy
from punar.core.taxonomy import NON_RETRYABLE
from punar.sim import arms as arms_mod
from punar.sim.arms import ARM_LABELS, ARMS, COMPARATORS, run_arm
from punar.sim.params import SimParams

# Pinned clock: 10:00 IST, inside the 08:00-19:00 contact window. Never
# datetime.now() -- the benchmark must be reproducible on any day.
SIM_NOW = datetime(2026, 8, 28, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

DEFAULT_PARAMS = SimParams()

# Back-compat alias for callers that import the annoyance constant by its old
# name. The value now lives in punar.sim.params with its justification.
ANNOY_MULT = DEFAULT_PARAMS.annoyance_inr_per_unwanted_touch


# ---------------------------------------------------------------------------
# running arms
# ---------------------------------------------------------------------------
def run_cohort(cases: list[dict[str, Any]], arm: str, policy: dict[str, Any],
               seed: int, params: SimParams | None = None,
               now: datetime | None = None) -> list[dict[str, Any]]:
    """Run one arm over a cohort and return one row per case."""
    p = params or DEFAULT_PARAMS
    clock = now or SIM_NOW
    return [run_arm(arm, case, seed, p, policy, clock) for case in cases]


def run_all_arms(cases: list[dict[str, Any]], policy: dict[str, Any], seed: int,
                 arm_names: list[str] | None = None,
                 params: SimParams | None = None,
                 now: datetime | None = None) -> dict[str, list[dict[str, Any]]]:
    names = arm_names or (COMPARATORS + ["punar"])
    return {name: run_cohort(cases, name, policy, seed, params, now) for name in names}


def run_policy(cases: list[dict[str, Any]], policy_path: str, seed: int,
               use_agent: bool = True, params: SimParams | None = None,
               arm: str | None = None) -> list[dict[str, Any]]:
    """Legacy entry point. ``use_agent`` picks punar vs the naive comparator;
    prefer passing ``arm=`` explicitly."""
    policy = load_policy(policy_path)
    name = arm or ("punar" if use_agent else "naive")
    return run_cohort(cases, name, policy, seed, params)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one arm's rows.

    Gross revenue is what the arm recovered. Net revenue subtracts channel spend
    and the modelled annoyance cost of unwanted contact, on the same cost basis
    for every arm. Quote net.
    """
    total = len(rows) or 1
    recovered = [r for r in rows if r["recovered"]]
    touches_total = sum(r["touch_count"] for r in rows)
    organic = [r for r in recovered if r.get("recovered_via") == "organic"]
    spend = sum(float(r.get("channel_spend_inr", r.get("cost_inr", 0.0))) for r in rows)
    annoyance = sum(float(r.get("annoyance_cost_inr", 0.0)) for r in rows)
    gross = sum(float(r["amount_inr"]) for r in recovered)
    days = [r["recovery_day"] for r in recovered if r.get("recovery_day") is not None]
    return {
        "cases": len(rows),
        "recovery_rate": round(len(recovered) / total, 4),
        "recovered_cases": len(recovered),
        "organic_recoveries": len(organic),
        "policy_driven_recoveries": len(recovered) - len(organic),
        "revenue_recovered": round(gross, 2),
        "gross_revenue_inr": round(gross, 2),
        "channel_spend_inr": round(spend, 2),
        "annoyance_cost_inr": round(annoyance, 2),
        "net_revenue_inr": round(gross - spend - annoyance, 2),
        "touches_total": touches_total,
        "touches_per_case": round(touches_total / total, 3),
        "touches_per_recovery": round(touches_total / len(recovered), 2) if recovered else 0.0,
        "opt_out_violations": sum(int(r.get("opt_out_violations", 0)) for r in rows),
        "non_retriable_touches": sum(int(r.get("non_retriable_touches", 0)) for r in rows),
        "unwanted_touches": sum(int(r.get("unwanted_touches", 0)) for r in rows),
        "mean_days_to_recovery": round(sum(days) / len(days), 2) if days else 0.0,
        # legacy key names, kept so existing report/chart code keeps working
        "total_cost_inr": round(spend, 2),
        "false_positive_annoyance_cost_inr": round(annoyance, 2),
    }


def by_reason(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r["reason"], []).append(r)
    return {k: metrics(v) for k, v in sorted(buckets.items())}


def deltas(base: dict[str, Any], treat: dict[str, Any]) -> dict[str, Any]:
    """Treatment minus comparator, on every headline metric."""
    return {
        "recovery_rate_delta_pts": round((treat["recovery_rate"] - base["recovery_rate"]) * 100, 2),
        "extra_recovered": treat["recovered_cases"] - base["recovered_cases"],
        "extra_revenue_inr": round(treat["revenue_recovered"] - base["revenue_recovered"], 2),
        "net_revenue_delta_inr": round(treat["net_revenue_inr"] - base["net_revenue_inr"], 2),
        "touches_per_recovery_delta": round(
            treat["touches_per_recovery"] - base["touches_per_recovery"], 2),
        "opt_out_violations_delta": treat["opt_out_violations"] - base["opt_out_violations"],
        "cost_delta_inr": round(treat["total_cost_inr"] - base["total_cost_inr"], 2),
        "annoyance_cost_delta_inr": round(
            treat["false_positive_annoyance_cost_inr"] - base["false_positive_annoyance_cost_inr"], 2),
    }


# per-case vectors used by the bootstrap / permutation machinery
CASE_METRICS = {
    "recovered": lambda r: 1.0 if r["recovered"] else 0.0,
    "gross_revenue_inr": lambda r: float(r.get("gross_revenue_inr", 0.0)),
    "net_revenue_inr": lambda r: float(r.get("net_revenue_inr", 0.0)),
    "touches": lambda r: float(r["touch_count"]),
    "unwanted_touches": lambda r: float(r.get("unwanted_touches", 0)),
}


def case_vector(rows: list[dict[str, Any]], metric: str) -> list[float]:
    """Per-case values for `metric`, in cohort order, so two arms line up pairwise."""
    fn = CASE_METRICS[metric]
    return [fn(r) for r in rows]


# ---------------------------------------------------------------------------
# legacy two-arm comparison
# ---------------------------------------------------------------------------
def compare_policies(cases: list[dict[str, Any]], policy_path: str, seed: int,
                     params: SimParams | None = None,
                     baseline_arm: str = "naive",
                     arm_names: list[str] | None = None) -> dict[str, Any]:
    """Run every arm over one cohort.

    Keeps the historical two-key shape (``baseline`` / ``punar`` / ``deltas``)
    so existing callers do not break, but the honest answer is in ``arms`` --
    a single comparator can no longer stand in for the result.
    """
    policy = load_policy(policy_path)
    names = arm_names or sorted(set(COMPARATORS + [baseline_arm, "punar"]),
                                key=lambda n: (COMPARATORS + ["punar"]).index(n)
                                if n in COMPARATORS + ["punar"] else 99)
    rows = run_all_arms(cases, policy, seed, names, params)
    arm_metrics = {name: metrics(r) for name, r in rows.items()}

    b = arm_metrics[baseline_arm]
    p = arm_metrics["punar"]
    return {
        "baseline": b,
        "baseline_arm": baseline_arm,
        "punar": p,
        "deltas": deltas(b, p),
        "arms": arm_metrics,
        "arm_labels": {n: ARM_LABELS.get(n, n) for n in arm_metrics},
        "arm_rows": rows,
        "cases_baseline": rows[baseline_arm],
        "cases_punar": rows["punar"],
        "by_reason_punar": by_reason(rows["punar"]),
        "vs": {n: deltas(arm_metrics[n], p) for n in arm_metrics if n != "punar"},
        "params": (params or DEFAULT_PARAMS).to_dict(),
    }


__all__ = ["SIM_NOW", "ANNOY_MULT", "ARMS", "ARM_LABELS", "COMPARATORS",
           "run_cohort", "run_all_arms", "run_policy", "metrics", "by_reason",
           "deltas", "case_vector", "CASE_METRICS", "compare_policies",
           "NON_RETRYABLE", "arms_mod"]
