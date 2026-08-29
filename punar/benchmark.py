"""Canonical benchmark: every arm, over many seeds, with confidence intervals.

Design rules, all of them reactions to ways benchmarks mislead:

* **One world model, one scoring path.** Every arm -- do-nothing, naive
  dunning, a realistic merchant baseline, the ablations and Punar -- is scored
  by the same simulator in :mod:`punar.sim.world`. No arm gets a private
  penalty or bonus branch.
* **No point estimate without an interval.** Headline numbers are reported over
  ``--seeds`` independent cohorts with bootstrap confidence intervals and a
  paired permutation test, because a single seed says nothing about variance.
* **A do-nothing control.** Without it no arm's absolute recovery rate has a
  reference point: some failed payments self-cure with no intervention at all.
* **Ablations.** Taxonomy routing alone, then + guardrails, then + bandit, so
  the contribution of each component is visible rather than asserted.
* **Assumptions travel with the result.** Every modelling constant is emitted
  alongside the numbers (``--show-params``), so a reader can see exactly what
  was assumed rather than reverse-engineering it from the source.

Nothing here is calibrated on real Razorpay data. These are simulated outcomes
under stated priors; the arms are comparable to each other and to nothing else.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import Any, NamedTuple

from punar.core.bandit_store import BanditStore
from punar.core.gate import load_policy
from punar.core.select import set_bandit_store
from punar.sim.arms import ARM_LABELS, COMPARATORS
from punar.sim.cohort import generate_cohort
from punar.sim.engine import by_reason, metrics, run_all_arms
from punar.sim.params import SimParams
from punar.sim.stats import (
    Interval,
    bootstrap_ci,
    cohens_dz,
    paired_bootstrap_ci,
    paired_permutation_test,
)

DEFAULT_SEEDS = 30
DEFAULT_N_CASES = 250
ABLATION_ARMS = ["taxonomy_only", "taxonomy_guardrails", "punar"]

#: Headline metrics, and how to read each one off an arm's aggregate.
HEADLINE = {
    "recovery_rate": ("Recovery rate", "%", 1, 100.0),
    "gross_revenue_inr": ("Gross revenue recovered", " INR", 0, 1.0),
    "net_revenue_inr": ("Net revenue (gross - spend - annoyance)", " INR", 0, 1.0),
    "touches_per_recovery": ("Customer touches per recovery", "", 2, 1.0),
    "opt_out_violations": ("Contacts to opted-out customers", "", 0, 1.0),
    "non_retriable_touches": ("Contacts on non-retriable declines", "", 0, 1.0),
    "channel_spend_inr": ("Channel + ops spend", " INR", 0, 1.0),
    "organic_recoveries": ("Recoveries that needed no action (self-cure)", "", 1, 1.0),
}


def _arm_names(include_ablations: bool) -> list[str]:
    names = list(COMPARATORS) + ["punar"]
    if include_ablations:
        for name in ABLATION_ARMS:
            if name not in names:
                names.insert(len(names) - 1, name)
    return names


@contextmanager
def learning_store(seed: int) -> Iterator[BanditStore]:
    """A posterior store scoped to one cohort run.

    Learning must accumulate ACROSS the cases in a cohort -- that is the whole
    claim the bandit makes -- while staying reproducible: the store is created
    empty for each seed and discarded afterwards, so re-running a seed gives
    the identical trajectory rather than inheriting a previous run's
    posteriors.
    """
    handle, path = tempfile.mkstemp(prefix=f"punar_bandit_{seed}_", suffix=".db")
    os.close(handle)
    os.unlink(path)
    store = BanditStore(path)
    set_bandit_store(store)
    try:
        yield store
    finally:
        set_bandit_store(None)
        for suffix in ("", "-wal", "-shm"):
            with suppress(OSError):
                os.unlink(path + suffix)


class SeedResult(NamedTuple):
    """One cohort's outcome: per-arm aggregates, the raw rows, what was learned."""

    aggregates: dict[str, dict[str, Any]]
    rows: dict[str, list[dict[str, Any]]]
    posteriors: list[dict[str, Any]]


def run_seed(n_cases: int, seed: int, policy: dict[str, Any],
             params: SimParams, arm_names: Sequence[str],
             include_unclassifiable: bool = True,
             learn: bool = True) -> SeedResult:
    """Run every arm over one cohort and return per-arm aggregates."""
    cases = generate_cohort(n_cases, seed=seed,
                            include_unclassifiable=include_unclassifiable)
    if learn:
        with learning_store(seed) as store:
            rows = run_all_arms(cases, policy, seed, list(arm_names), params)
            posteriors = store.all_posteriors()
    else:
        rows = run_all_arms(cases, policy, seed, list(arm_names), params)
        posteriors = []
    return SeedResult({name: metrics(r) for name, r in rows.items()}, rows, posteriors)


def run_benchmark(n_cases: int = DEFAULT_N_CASES, seed: int = 42,
                  policy_path: str | None = None,
                  seeds: int = DEFAULT_SEEDS,
                  params: SimParams | None = None,
                  include_ablations: bool = True,
                  baseline_arm: str = "realistic",
                  include_unclassifiable: bool = True,
                  learn: bool = True,
                  progress: Callable[[int, int], None] | None = None,
                  ) -> dict[str, Any]:
    """Run the full benchmark across ``seeds`` cohorts starting at ``seed``.

    Returns a result dict whose every headline number carries a confidence
    interval, plus a paired significance test against ``baseline_arm``.
    """
    policy = load_policy(policy_path or "punar/config/policy.json")
    params = params or SimParams()
    names = _arm_names(include_ablations)
    seed_list = [seed + i for i in range(max(1, seeds))]

    # per-arm, per-metric: one observation per seed
    series: dict[str, dict[str, list[float]]] = {n: {m: [] for m in HEADLINE} for n in names}
    last_rows: dict[str, list[dict[str, Any]]] = {}
    last_posteriors: list[dict[str, Any]] = []

    for index, s in enumerate(seed_list, 1):
        outcome = run_seed(n_cases, s, policy, params, names,
                           include_unclassifiable, learn=learn)
        last_rows, last_posteriors = outcome.rows, outcome.posteriors
        for name in names:
            for metric in HEADLINE:
                series[name][metric].append(
                    float(outcome.aggregates[name].get(metric, 0.0) or 0.0))
        if progress:
            progress(index, len(seed_list))

    summary: dict[str, dict[str, Interval]] = {
        name: {metric: bootstrap_ci(values, seed=seed)
               for metric, values in per_metric.items()}
        for name, per_metric in series.items()
    }

    # Paired comparisons: the same cohorts under each arm, so pair by seed.
    comparisons: dict[str, dict[str, Any]] = {}
    punar_rates = series["punar"]["recovery_rate"]
    punar_net = series["punar"]["net_revenue_inr"]
    for name in names:
        if name == "punar":
            continue
        base_rates = series[name]["recovery_rate"]
        base_net = series[name]["net_revenue_inr"]
        # These helpers report mean(b) - mean(a), so the comparator is `a` and
        # Punar is `b`: a positive number always means Punar did better.
        lift = paired_bootstrap_ci(base_rates, punar_rates, seed=seed)
        net = paired_bootstrap_ci(base_net, punar_net, seed=seed)
        comparisons[name] = {
            "recovery_lift_pts": lift,
            "net_revenue_delta_inr": net,
            "test": paired_permutation_test(base_rates, punar_rates, seed=seed),
            "effect_size_dz": cohens_dz(base_rates, punar_rates),
        }

    return {
        "config": {
            "n_cases": n_cases, "seeds": len(seed_list),
            "seed_start": seed, "seed_end": seed_list[-1],
            "policy_path": policy_path or "punar/config/policy.json",
            "baseline_arm": baseline_arm,
            "include_ablations": include_ablations,
            "include_unclassifiable": include_unclassifiable,
            "learning_enabled": learn,
        },
        "arms": names,
        "labels": {n: ARM_LABELS.get(n, n) for n in names},
        "summary": summary,
        "series": series,
        "comparisons": comparisons,
        "by_reason_punar": by_reason(last_rows.get("punar", [])),
        "learned_posteriors": last_posteriors,
        "params": params.to_dict(),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def format_report(result: dict[str, Any], show_params: bool = False) -> str:
    """Human-readable report. Every number carries its interval."""
    cfg = result["config"]
    out: list[str] = []
    out.append("Punar benchmark")
    out.append(f"  cohort      : n={cfg['n_cases']} cases x {cfg['seeds']} seeds "
               f"({cfg['seed_start']}..{cfg['seed_end']})")
    out.append(f"  policy      : {cfg['policy_path']}")
    out.append(f"  comparator  : {cfg['baseline_arm']}")
    out.append("  intervals   : 95% bootstrap over seeds; outcomes are SIMULATED")
    out.append("")

    for metric, (label, unit, places, scale) in HEADLINE.items():
        out.append(f"{label}")
        for name in result["arms"]:
            interval = result["summary"][name][metric]
            out.append(f"  {name:<22} {interval.fmt(unit, places, scale)}")
        out.append("")

    out.append("Punar vs each comparator (paired over seeds)")
    for name, comparison in result["comparisons"].items():
        lift = comparison["recovery_lift_pts"]
        net = comparison["net_revenue_delta_inr"]
        p_value = comparison["test"]["p_value"]
        stars = "significant" if p_value < 0.05 else "NOT significant"
        out.append(f"  vs {name}")
        out.append(f"      recovery lift : {lift.fmt(' pts', 1, 100.0)}")
        out.append(f"      net revenue   : {net.fmt(' INR', 0, 1.0)}")
        out.append(f"      p={p_value:.4f} ({stars}), "
                   f"dz={comparison['effect_size_dz']:.2f}")
    out.append("")

    if show_params:
        out.append("Modelling assumptions (every constant, so none is hidden)")
        for key, value in sorted(result["params"].items()):
            out.append(f"  {key:<38} {value}")
        out.append("")

    out.append("These are simulated outcomes under stated priors, not measured "
               "Razorpay results. Arms are comparable to each other only.")
    return "\n".join(out)


def to_json_safe(result: dict[str, Any]) -> dict[str, Any]:
    """Convert Interval objects so the result can be json.dump()ed."""
    summary = {name: {metric: interval.to_dict() for metric, interval in per_metric.items()}
               for name, per_metric in result["summary"].items()}
    comparisons = {
        name: {
            "recovery_lift_pts": c["recovery_lift_pts"].to_dict(),
            "net_revenue_delta_inr": c["net_revenue_delta_inr"].to_dict(),
            "test": c["test"],
            "effect_size_dz": c["effect_size_dz"],
        }
        for name, c in result["comparisons"].items()
    }
    return {**{k: v for k, v in result.items() if k not in ("summary", "comparisons")},
            "summary": summary, "comparisons": comparisons}
