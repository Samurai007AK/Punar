#!/usr/bin/env python3
"""Rule-based vs LLM(-or-mock) reason-diagnosis ablation.

Compares the shipped rule-based classifier (punar.core.classify) against the
optional LLM variant (punar.core.classify_llm) over synthetic cohort
fixtures -- the same generator scripts/benchmark.py uses -- on:

* overall accuracy against the simulator's ground-truth reason;
* false-positive rate on non-retriable declines, specifically in the
  DANGEROUS direction: a ground-truth non-retriable decline (fraud_block,
  lost_stolen_card, mandate_expired, account_closed) that gets classified
  as something else. That miss means a customer is CONTACTED about a
  payment method that is fraudulent, stolen, or a closed/dormant account --
  the compliance failure this repo's whole guardrail story exists to
  prevent. A retriable decline wrongly called non-retriable only costs a
  missed retry (recoverable next run), which is why that direction is not
  what "false positive" means here;
* average latency for whichever path actually ran (wall clock; the real
  Groq path is unmeasured in this sandbox -- no network, no key);
* an estimated API cost per 1,000 cases, from openly guessed token counts
  and per-token prices (see the constants below -- this sandbox has no
  internet access to check a real price sheet).

Runnable with zero setup: `python scripts/llm_ablation.py` uses the mock LLM
classifier (punar.core.classify_llm.mock_classify_llm) whenever
USE_LLM_DIAGNOSIS=1 and GROQ_API_KEY are not both set, which is the default
in any environment without a Groq key. The report says plainly which path
ran; it never silently presents a mock number as a real API result.

Examples
--------
    python scripts/llm_ablation.py                  # 300 cases, mock LLM path
    python scripts/llm_ablation.py --n-cases 500
    USE_LLM_DIAGNOSIS=1 GROQ_API_KEY=... python scripts/llm_ablation.py  # real path
"""
import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punar.core.classify import classify  # noqa: E402
from punar.core.classify_llm import classify_llm_or_mock  # noqa: E402
from punar.core.taxonomy import NON_RETRYABLE, get_reason  # noqa: E402
from punar.sim.cohort import generate_cohort  # noqa: E402

DEFAULT_N_CASES = 300
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Cost model. Openly guessed, not a real Groq invoice -- this sandbox has no
# internet access to check current pricing. Assumes an 8B-class hosted model,
# a short fixed-instruction prompt plus three extracted fields (~130 input
# tokens) and a one-line JSON reply (~20 output tokens), at per-token rates
# in the ballpark of small hosted-model pricing as of this writing.
# ponytail: three guessed constants standing in for a price sheet. Swap for
# the real per-token rate from a GROQ_API_KEY invoice once one exists --
# nothing else in the cost line changes.
# ---------------------------------------------------------------------------
ASSUMED_PROMPT_TOKENS = 130
ASSUMED_COMPLETION_TOKENS = 20
ASSUMED_PRICE_PER_1K_PROMPT_TOKENS_USD = 0.00005     # guess: ~$0.05 / 1M input tokens
ASSUMED_PRICE_PER_1K_COMPLETION_TOKENS_USD = 0.00008  # guess: ~$0.08 / 1M output tokens


def _est_cost_per_1000_cases_usd() -> float:
    per_case = (ASSUMED_PROMPT_TOKENS / 1000) * ASSUMED_PRICE_PER_1K_PROMPT_TOKENS_USD \
        + (ASSUMED_COMPLETION_TOKENS / 1000) * ASSUMED_PRICE_PER_1K_COMPLETION_TOKENS_USD
    return per_case * 1000


def _non_retriable_fp(cases: list[dict[str, Any]], diagnoses: list[str]) -> tuple[float, int, int]:
    """(rate, fp_count, total) among ground-truth non-retriable cases where
    the diagnosis was NOT non-retriable -- the dangerous direction."""
    pairs = [(c, d) for c, d in zip(cases, diagnoses, strict=True)
             if get_reason(c["reason"]).retriability == NON_RETRYABLE]
    if not pairs:
        return 0.0, 0, 0
    fps = sum(1 for c, d in pairs if get_reason(d).retriability != NON_RETRYABLE)
    return fps / len(pairs), fps, len(pairs)


def run(n_cases: int, seed: int) -> dict[str, Any]:
    cases = generate_cohort(n_cases, seed=seed)
    truth = [c["reason"] for c in cases]

    rule_diag: list[str] = []
    llm_diag: list[str] = []
    was_real: list[bool] = []
    latencies_s: list[float] = []

    for case in cases:
        rule_key, _meta = classify(case)
        rule_diag.append(rule_key)

        t0 = time.perf_counter()
        llm_key, _confidence, real = classify_llm_or_mock(case)
        latencies_s.append(time.perf_counter() - t0)
        llm_diag.append(llm_key)
        was_real.append(real)

    rule_accuracy = sum(t == d for t, d in zip(truth, rule_diag, strict=True)) / len(cases)
    llm_accuracy = sum(t == d for t, d in zip(truth, llm_diag, strict=True)) / len(cases)
    rule_fp_rate, rule_fp, nr_total = _non_retriable_fp(cases, rule_diag)
    llm_fp_rate, llm_fp, _ = _non_retriable_fp(cases, llm_diag)

    all_real = bool(was_real) and all(was_real)
    any_real = any(was_real)
    llm_path = "real (Groq API)" if all_real else ("mixed (some calls fell back)" if any_real else "mock")
    avg_latency_ms = (sum(latencies_s) / len(latencies_s)) * 1000 if latencies_s else 0.0

    return {
        "n_cases": len(cases),
        "non_retriable_cases": nr_total,
        "rule_accuracy": rule_accuracy,
        "llm_accuracy": llm_accuracy,
        "rule_fp_rate": rule_fp_rate,
        "rule_fp": rule_fp,
        "llm_fp_rate": llm_fp_rate,
        "llm_fp": llm_fp,
        "llm_path": llm_path,
        "avg_latency_ms": avg_latency_ms,
        "measured_real_latency": all_real,
        "est_cost_per_1000_usd": _est_cost_per_1000_cases_usd(),
    }


def format_report(r: dict[str, Any]) -> str:
    lines = [
        f"Punar LLM-diagnosis ablation -- {r['n_cases']} cases "
        f"({r['non_retriable_cases']} ground-truth non-retriable)",
        f"LLM path this run: {r['llm_path']}",
        "",
        f"{'':12s}{'accuracy':>10s}{'non-retriable FP rate':>24s}",
        f"{'rule-based':12s}{r['rule_accuracy'] * 100:9.1f}%"
        f"{r['rule_fp_rate'] * 100:23.1f}%  ({r['rule_fp']}/{r['non_retriable_cases']})",
        f"{'llm':12s}{r['llm_accuracy'] * 100:9.1f}%"
        f"{r['llm_fp_rate'] * 100:23.1f}%  ({r['llm_fp']}/{r['non_retriable_cases']})",
        "",
    ]
    if r["measured_real_latency"]:
        lines.append(f"Avg latency (real Groq calls): {r['avg_latency_ms']:.1f} ms/case")
    else:
        lines.append(f"Avg latency (mock path): {r['avg_latency_ms']:.4f} ms/case "
                     "-- wall clock for the local mock function, NOT a real network call")
        lines.append("Real-path latency: unmeasured (no USE_LLM_DIAGNOSIS=1 + GROQ_API_KEY in this run)")
    lines.append(
        f"Estimated API cost per 1,000 cases: ${r['est_cost_per_1000_usd']:.4f} "
        f"(GUESS: {ASSUMED_PROMPT_TOKENS} prompt + {ASSUMED_COMPLETION_TOKENS} completion "
        f"tokens/case @ ${ASSUMED_PRICE_PER_1K_PROMPT_TOKENS_USD * 1000:.3f}/"
        f"${ASSUMED_PRICE_PER_1K_COMPLETION_TOKENS_USD * 1000:.3f} per 1K prompt/completion tokens)")
    lines += [
        "",
        "Why the FP direction above and not the other one: a non-retriable decline",
        "(fraud/stolen card/closed account/expired mandate) misclassified as retriable",
        "gets that customer CONTACTED about it -- the compliance-relevant failure.",
        "A retriable decline wrongly called non-retriable only costs a missed retry.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rule-based vs LLM(-or-mock) diagnosis ablation")
    parser.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES,
                        help="cases in the cohort (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    print(format_report(run(args.n_cases, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
