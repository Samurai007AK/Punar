"""Charts for the benchmark report.

Every chart obeys two rules that follow from how the numbers are produced:

1. **No point estimate without its interval.** Bars carry 95% bootstrap error
   bars computed over seeds. A bare bar would imply a precision the simulation
   does not have.
2. **Simulated is labelled simulated.** Each figure carries a provenance
   footer, so a screenshot lifted out of context still says what it is.

Matplotlib is an optional dependency: :func:`generate_charts` degrades to
returning an empty mapping rather than failing a report run.
"""
from __future__ import annotations

import os
from typing import Any

FOOTER = ("Simulated outcomes under stated priors -- not measured Razorpay "
          "results. Bars show 95% bootstrap CIs over seeds.")

# Colour-blind-safe, and deliberately muted for every arm except Punar so the
# comparison reads at a glance without relying on hue alone.
ARM_COLOURS = {
    "do_nothing": "#9aa0a6",
    "naive": "#c86d6d",
    "realistic": "#4b7bb5",
    "realistic_english": "#7fa3cc",
    "taxonomy_only": "#c9a227",
    "taxonomy_guardrails": "#8a9a3f",
    "punar": "#1f7a5a",
}


def _matplotlib():
    """Import matplotlib with a non-interactive backend, or return None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=144, bbox_inches="tight", facecolor="white")
    return path


def _finish(plt, fig, ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color="#555555", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.04, FOOTER, fontsize=7.5, color="#666666")


def _arm_bars(plt, result: dict[str, Any], metric: str, title: str,
              ylabel: str, out_dir: str, name: str, scale: float = 1.0,
              subtitle: str = "") -> str | None:
    arms: list[str] = result["arms"]
    summary = result["summary"]
    points = [summary[a][metric].point * scale for a in arms]
    lows = [max(0.0, points[i] - summary[a][metric].lo * scale) for i, a in enumerate(arms)]
    highs = [max(0.0, summary[a][metric].hi * scale - points[i]) for i, a in enumerate(arms)]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = [ARM_COLOURS.get(a, "#888888") for a in arms]
    bars = ax.bar(range(len(arms)), points, color=colours,
                  yerr=[lows, highs], capsize=4,
                  error_kw={"ecolor": "#333333", "elinewidth": 1.1})
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=8.5)
    ax.set_ylabel(ylabel)

    headroom = max(points + [1e-9]) * 0.18
    ax.set_ylim(0, max(points) + headroom + max(highs or [0]))
    for rect, value in zip(bars, points, strict=False):
        ax.annotate(f"{value:,.1f}" if scale != 1.0 or value < 100 else f"{value:,.0f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5, fontweight="bold")
    _finish(plt, fig, ax, title, subtitle)
    path = _save(fig, out_dir, name)
    plt.close(fig)
    return path


def chart_recovery(plt, result, out_dir: str) -> str | None:
    return _arm_bars(plt, result, "recovery_rate", "Recovery rate by arm",
                     "recovered (%)", out_dir, "recovery_rate", scale=100.0,
                     subtitle="Higher is better. 'do_nothing' is the organic "
                              "self-cure floor every other arm must beat.")


def chart_touches(plt, result, out_dir: str) -> str | None:
    return _arm_bars(plt, result, "touches_per_recovery",
                     "Customer touches spent per recovery",
                     "touches / recovery", out_dir, "touches_per_recovery",
                     subtitle="Lower is better: the same revenue for less of "
                              "the customer's attention.")


def chart_compliance(plt, result, out_dir: str) -> str | None:
    """Contacts that should never have happened. Punar's core claim."""
    arms: list[str] = result["arms"]
    summary = result["summary"]
    opt_out = [summary[a]["opt_out_violations"].point for a in arms]
    non_retriable = [summary[a]["non_retriable_touches"].point for a in arms]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.38
    xs = range(len(arms))
    ax.bar([x - width / 2 for x in xs], opt_out, width, label="contacted after opt-out",
           color="#b3352e")
    ax.bar([x + width / 2 for x in xs], non_retriable, width,
           label="contacted on a non-retriable decline", color="#e0a33e")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=8.5)
    ax.set_ylabel("contacts per cohort")
    ax.set_ylim(0, max(opt_out + non_retriable + [1]) * 1.25)
    for x, value in zip(xs, opt_out, strict=False):
        ax.annotate(f"{value:,.0f}", (x - width / 2, value), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    for x, value in zip(xs, non_retriable, strict=False):
        ax.annotate(f"{value:,.0f}", (x + width / 2, value), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _finish(plt, fig, ax, "Contacts that should never have been made",
            "Zero is the only acceptable value. Punar's gate is what makes it zero.")
    path = _save(fig, out_dir, "compliance_violations")
    plt.close(fig)
    return path


def chart_ablation(plt, result, out_dir: str) -> str | None:
    """What each component actually contributes, vs. the comparator."""
    comparisons = result.get("comparisons", {})
    order = [n for n in ("do_nothing", "naive", "realistic", "taxonomy_only",
                         "taxonomy_guardrails") if n in comparisons]
    if not order:
        return None
    points = [comparisons[n]["recovery_lift_pts"].point * 100 for n in order]
    lows = [points[i] - comparisons[n]["recovery_lift_pts"].lo * 100
            for i, n in enumerate(order)]
    highs = [comparisons[n]["recovery_lift_pts"].hi * 100 - points[i]
             for i, n in enumerate(order)]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = ["#1f7a5a" if p > 0 else "#b3352e" for p in points]
    ax.barh(range(len(order)), points, color=colours,
            xerr=[lows, highs], capsize=4,
            error_kw={"ecolor": "#333333", "elinewidth": 1.1})
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n.replace("_", " ") for n in order], fontsize=9)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Punar recovery rate minus that arm (percentage points)")
    span = max(abs(min(points)), abs(max(points))) + max(highs + lows + [1])
    ax.set_xlim(-span * 1.35, span * 1.35)
    for i, (point, name) in enumerate(zip(points, order, strict=False)):
        p_value = comparisons[name]["test"]["p_value"]
        mark = "" if p_value < 0.05 else "  (n.s.)"
        offset = 10 if point >= 0 else -10
        ax.annotate(f"{point:+.1f} pts{mark}", (point, i), textcoords="offset points",
                    xytext=(offset, 0), ha="left" if point >= 0 else "right",
                    va="center", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Where the lift actually comes from", fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    fig.text(0.01, -0.04, FOOTER, fontsize=7.5, color="#666666")
    path = _save(fig, out_dir, "ablation")
    plt.close(fig)
    return path


def chart_learning(plt, result, out_dir: str) -> str | None:
    """What the bandit learned: posterior mean per (reason, intervention).

    This is the chart that answers "show me what it learned" -- without a
    persisted posterior there would be nothing to plot.
    """
    posteriors = result.get("learned_posteriors") or []
    rows = [p for p in posteriors if (p.get("alpha", 0) + p.get("beta", 0)) >= 4]
    if not rows:
        return None
    rows.sort(key=lambda r: (r.get("reason", ""), r.get("intervention", "")))
    rows = rows[:18]

    labels = [f"{r['reason']}\n{r['intervention']}" for r in rows]
    means = [r["alpha"] / (r["alpha"] + r["beta"]) for r in rows]
    observations = [int(r.get("updates") or (r["alpha"] + r["beta"] - 2)) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    bars = ax.bar(range(len(rows)), means, color="#1f7a5a")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("learned success probability")
    ax.set_ylim(0, 1.12)
    for rect, count in zip(bars, observations, strict=False):
        ax.annotate(f"n={count}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5)
    _finish(plt, fig, ax, "What the bandit learned from observed outcomes",
            "Beta posterior mean per (decline reason, intervention). "
            "Seeded from priors, moved by evidence.")
    path = _save(fig, out_dir, "bandit_posteriors")
    plt.close(fig)
    return path


def generate_charts(result: dict[str, Any], out_dir: str = "outputs") -> dict[str, str]:
    """Render every chart. Returns {name: path}; empty if matplotlib is absent."""
    plt = _matplotlib()
    if plt is None:
        return {}
    charts = {
        "recovery_rate": chart_recovery(plt, result, out_dir),
        "touches_per_recovery": chart_touches(plt, result, out_dir),
        "compliance_violations": chart_compliance(plt, result, out_dir),
        "ablation": chart_ablation(plt, result, out_dir),
        "bandit_posteriors": chart_learning(plt, result, out_dir),
    }
    return {name: path for name, path in charts.items() if path}
