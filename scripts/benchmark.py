#!/usr/bin/env python3
"""Run the canonical Punar benchmark.

Examples
--------
    python scripts/benchmark.py                       # 250 cases x 30 seeds
    python scripts/benchmark.py --seeds 1             # single cohort, fast
    python scripts/benchmark.py --show-params         # print every assumption
    python scripts/benchmark.py --json out/bench.json # machine-readable
"""
import argparse
import json
import os
import sys

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punar.benchmark import (DEFAULT_N_CASES, DEFAULT_SEEDS, format_report,
                             run_benchmark, to_json_safe)
from punar.sim.params import SimParams


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Punar recovery benchmark")
    parser.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES,
                        help="cases per cohort (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="first seed; cohorts use seed..seed+seeds-1")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="independent cohorts to average over (default: %(default)s)")
    parser.add_argument("--policy", default="punar/config/policy.json")
    parser.add_argument("--baseline", default="realistic",
                        choices=["do_nothing", "naive", "realistic"],
                        help="comparator for the headline lift (default: %(default)s)")
    parser.add_argument("--no-ablations", action="store_true",
                        help="skip the taxonomy/guardrails ablation arms")
    parser.add_argument("--show-params", action="store_true",
                        help="print every modelling constant with the results")
    parser.add_argument("--json", metavar="PATH", help="also write JSON results here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def progress(done: int, total: int) -> None:
        if not args.quiet and total > 1:
            print(f"\r  seed {done}/{total}", end="", file=sys.stderr, flush=True)

    result = run_benchmark(
        n_cases=args.n_cases, seed=args.seed, seeds=args.seeds,
        policy_path=args.policy, params=SimParams(),
        include_ablations=not args.no_ablations,
        baseline_arm=args.baseline, progress=progress)

    if not args.quiet and args.seeds > 1:
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr)

    print(format_report(result, show_params=args.show_params))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(to_json_safe(result), handle, indent=2, default=str)
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
