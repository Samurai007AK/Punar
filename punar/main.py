"""Punar command line.

    punar bench                 # run the benchmark, print the report
    punar report --out DIR      # benchmark + charts + a written report on disk
    punar recover --case FILE   # run one failed payment through the agent
    punar audit verify          # verify the audit hash chain
    punar learned               # show what the bandit has learned so far

Everything here runs offline against the simulator. No network calls, no
Razorpay credentials, no real messages -- outputs are labelled accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any

from punar.benchmark import (
    DEFAULT_N_CASES,
    DEFAULT_SEEDS,
    format_report,
    run_benchmark,
    to_json_safe,
)
from punar.charts import generate_charts
from punar.core.agent import run_agent
from punar.core.gate import load_policy
from punar.core.select import read_posteriors
from punar.sim.params import SimParams

DEFAULT_POLICY = "punar/config/policy.json"


# ---------------------------------------------------------------------------
# bench / report
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> dict[str, Any]:
    def progress(done: int, total: int) -> None:
        if not args.quiet and total > 1:
            print(f"\r  seed {done}/{total}", end="", file=sys.stderr, flush=True)

    result = run_benchmark(n_cases=args.n_cases, seed=args.seed, seeds=args.seeds,
                           policy_path=args.policy, params=SimParams(),
                           include_ablations=not args.no_ablations,
                           baseline_arm=args.baseline, progress=progress)
    if not args.quiet and args.seeds > 1:
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr)
    return result


def cmd_bench(args: argparse.Namespace) -> int:
    print(format_report(_run(args), show_params=args.show_params))
    return 0


def _markdown_report(result: dict[str, Any], charts: dict[str, str]) -> str:
    """A written report a reviewer can read without running anything."""
    cfg = result["config"]
    summary, comparisons = result["summary"], result["comparisons"]
    arms = result["arms"]
    lines: list[str] = []

    lines.append("# Punar benchmark report")
    lines.append("")
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                 f"n={cfg['n_cases']} cases x {cfg['seeds']} seeds "
                 f"(seeds {cfg['seed_start']}..{cfg['seed_end']})")
    lines.append("")
    lines.append("> **These are simulated outcomes under stated priors, not measured "
                 "Razorpay results.** Every arm is scored by the same world model in "
                 "`punar/sim/world.py`; no arm has a private scoring branch. All "
                 "intervals are 95% bootstrap CIs over seeds.")
    lines.append("")

    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | what it does |")
    lines.append("|---|---|")
    for name in arms:
        lines.append(f"| `{name}` | {result['labels'].get(name, name)} |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| arm | recovery rate | touches / recovery | opt-out contacts | "
                 "non-retriable contacts | net revenue (INR) |")
    lines.append("|---|---|---|---|---|---|")
    for name in arms:
        s = summary[name]
        lines.append(
            f"| `{name}` | {s['recovery_rate'].fmt('%', 1, 100.0)} | "
            f"{s['touches_per_recovery'].fmt('', 2)} | "
            f"{s['opt_out_violations'].point:,.0f} | "
            f"{s['non_retriable_touches'].point:,.0f} | "
            f"{s['net_revenue_inr'].point:,.0f} |")
    lines.append("")

    lines.append("## Punar vs each comparator")
    lines.append("")
    lines.append("| comparator | recovery lift | p | effect size | net revenue delta |")
    lines.append("|---|---|---|---|---|")
    for name, c in comparisons.items():
        p_value = c["test"]["p_value"]
        verdict = "" if p_value < 0.05 else " (n.s.)"
        lines.append(
            f"| `{name}` | {c['recovery_lift_pts'].fmt(' pts', 1, 100.0)} | "
            f"{p_value:.4f}{verdict} | dz={c['effect_size_dz']:.2f} | "
            f"{c['net_revenue_delta_inr'].fmt(' INR', 0)} |")
    lines.append("")

    if charts:
        lines.append("## Charts")
        lines.append("")
        for name, path in charts.items():
            lines.append(f"![{name}]({os.path.basename(path)})")
            lines.append("")

    lines.append("## Modelling assumptions")
    lines.append("")
    lines.append("Every constant that shapes these numbers, so none of them is hidden "
                 "in the source:")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    for key, value in sorted(result["params"].items()):
        rendered = str(value)
        if len(rendered) > 110:
            rendered = rendered[:107] + "..."
        lines.append(f"| `{key}` | {rendered} |")
    lines.append("")
    lines.append("Reproduce with:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python scripts/benchmark.py --n-cases {cfg['n_cases']} "
                 f"--seed {cfg['seed_start']} --seeds {cfg['seeds']}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    result = _run(args)
    os.makedirs(args.out, exist_ok=True)

    charts = generate_charts(result, args.out)
    if not charts and not args.quiet:
        print("[punar] matplotlib unavailable; skipping charts", file=sys.stderr)

    json_path = os.path.join(args.out, "benchmark.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(result), handle, indent=2, default=str)

    report_path = os.path.join(args.out, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown_report(result, charts))

    text_path = os.path.join(args.out, "report.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(format_report(result, show_params=True))

    print(format_report(result, show_params=args.show_params))
    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")
    for path in charts.values():
        print(f"Wrote {path}")
    return 0


# ---------------------------------------------------------------------------
# single case
# ---------------------------------------------------------------------------
def cmd_recover(args: argparse.Namespace) -> int:
    """Run one failed payment end to end and print the full decision trail."""
    policy = load_policy(args.policy)
    if args.case:
        with open(args.case, encoding="utf-8") as handle:
            case = json.load(handle)
    else:
        case = {"case_id": "pay_demo_1", "amount_inr": 2499.0, "method": "upi",
                "customer_id": "cust_demo_1", "merchant_name": "Demo Merchant",
                "language": args.language, "opted_out": args.opted_out,
                "error_code": "BAD_REQUEST_ERROR",
                "error_description": args.error or "insufficient balance in the account",
                "error_reason": args.reason or "INSUFFICIENT_FUNDS"}

    outcomes = iter([o.lower() in ("1", "true", "yes", "y") for o in (args.outcomes or [])])

    def simulate(_case: dict[str, Any], _intervention: str, _now: datetime) -> bool:
        try:
            return next(outcomes)
        except StopIteration:
            return False

    state = run_agent(case, policy, random.Random(args.seed), simulate)

    print(f"case      : {case.get('case_id')}")
    print(f"diagnosis : {state['diagnosis']['reason']} "
          f"({state['diagnosis']['retriability']})")
    print(f"outcome   : {state['outcome']}  [{state['exit_code']}]")
    print("")
    print("decision trail")
    for entry in state["audit"]:
        detail = json.dumps(entry["detail"], default=str)
        if len(detail) > 160:
            detail = detail[:157] + "..."
        print(f"  {entry['at']}  {entry['step']:<9} {detail}")
    if state["touch_history"]:
        print("")
        print("actions taken")
        for touch in state["touch_history"]:
            reach = "customer" if touch.get("contacts_customer") else "internal"
            print(f"  #{touch['round']} {touch['intervention']:<28} "
                  f"{touch['channel']:<20} [{reach}] {touch.get('delivery_status', '')}")
    if state["blocked_actions"]:
        print("")
        print("BLOCKED pre-send by the policy judge")
        for blocked in state["blocked_actions"]:
            print(f"  {blocked['intervention']}: {', '.join(blocked['violations'])}")
    return 0


# ---------------------------------------------------------------------------
# audit / learned
# ---------------------------------------------------------------------------
def cmd_audit(args: argparse.Namespace) -> int:
    from punar.audit import AuditStore
    with AuditStore(args.db) as store:
        result = store.verify_chain()
        print(json.dumps(result.to_dict(), indent=2))
        print(f"\nrows: {store.count_rows()}  cases: {store.count_cases()}")
    return 0 if result.ok else 1


def cmd_learned(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    rows = read_posteriors(policy)
    if not rows:
        print("No learned posteriors yet. Persistence is opt-in: set "
              "bandit.persist_posteriors=true in policy.json, or run the API.")
        return 0
    rows.sort(key=lambda r: (r.get("reason", ""), -(r.get("alpha", 0))))
    print(f"{'reason':<24} {'intervention':<30} {'mean':>6} {'n':>5}")
    for row in rows:
        alpha, beta = float(row["alpha"]), float(row["beta"])
        mean = alpha / (alpha + beta)
        print(f"{row['reason']:<24} {row['intervention']:<30} {mean:>6.3f} "
              f"{int(row.get('updates') or alpha + beta - 2):>5}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="punar",
        description="Punar -- autonomous recovery agent for failed Razorpay payments")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_bench_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                       help="independent cohorts to average over")
        p.add_argument("--policy", default=DEFAULT_POLICY)
        p.add_argument("--baseline", default="realistic",
                       choices=["do_nothing", "naive", "realistic"])
        p.add_argument("--no-ablations", action="store_true")
        p.add_argument("--show-params", action="store_true")
        p.add_argument("--quiet", action="store_true")

    bench = sub.add_parser("bench", help="run the benchmark and print the report")
    add_bench_args(bench)
    bench.set_defaults(func=cmd_bench)

    report = sub.add_parser("report", help="benchmark + charts + written report on disk")
    add_bench_args(report)
    report.add_argument("--out", default="outputs/demo_run")
    report.set_defaults(func=cmd_report)

    recover = sub.add_parser("recover", help="run one failed payment through the agent")
    recover.add_argument("--case", help="JSON file holding one case")
    recover.add_argument("--policy", default=DEFAULT_POLICY)
    recover.add_argument("--seed", type=int, default=42)
    recover.add_argument("--language", default="en", choices=["en", "hi", "hinglish"])
    recover.add_argument("--reason", help="Razorpay error_reason to simulate")
    recover.add_argument("--error", help="Razorpay error_description to simulate")
    recover.add_argument("--opted-out", action="store_true",
                         help="mark the customer as opted out")
    recover.add_argument("--outcomes", nargs="*",
                         help="scripted per-touch outcomes, e.g. false false true")
    recover.set_defaults(func=cmd_recover)

    audit = sub.add_parser("audit", help="verify the audit hash chain")
    audit.add_argument("--db", default="punar_audit.db")
    audit.set_defaults(func=cmd_audit)

    learned = sub.add_parser("learned", help="show what the bandit has learned")
    learned.add_argument("--policy", default=DEFAULT_POLICY)
    learned.set_defaults(func=cmd_learned)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
