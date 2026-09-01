# Why the shipped classifier is not an LLM

## The eval

`scripts/llm_ablation.py`, 300 synthetic cases (seed 42, same cohort generator the
benchmark uses), 24 of them ground-truth non-retriable. No `GROQ_API_KEY` is set in
this sandbox, so the "llm" row below is the deterministic mock classifier in
`punar/core/classify_llm.py`, not a real model call. The report says this plainly
every time it runs.

| path | accuracy | non-retriable FP rate |
|---|---|---|
| rule-based | 100.0% | 0.0% (0/24) |
| llm (mock) | 92.3% | 4.2% (1/24) |

Non-retriable FP rate means: of the cases whose ground-truth reason is
non-retriable (fraud_block, lost_stolen_card, mandate_expired, account_closed), how
many did the classifier call something other than non-retriable. That is the
dangerous direction, not the reverse. A retriable decline wrongly called
non-retriable costs one missed silent retry, recoverable on the next attempt. A
non-retriable decline wrongly called retriable gets that customer CONTACTED about a
payment method that is fraudulent, stolen, or a closed account. This repo's
guardrails (`core/gate.py`) exist specifically to stop that contact, and they trust
the classifier's retriability verdict to do it.

The rule-based classifier scores 100% here by construction: the cohort generator
seeds each case with error fields drawn from the same string patterns the rules
match on, so this number says the rules are internally consistent, not that they
generalise to real Razorpay traffic never seen before. The mock LLM's 92.3% and
4.2% are equally not a claim about real model accuracy: it is a seeded random
perturbation of the rule-based answer, built to exercise the eval's plumbing with
no API key, not a simulation of how a real LLM would behave. Estimated API cost is
a guess too: $0.0081 per 1,000 cases, from assumed token counts (130 prompt + 20
completion) at assumed per-token prices, with no internet access in this sandbox to
check against a real price sheet.

## Why the default path stays rule-based

**Determinism.** Same input produces the same output, every time, on every
machine. A reviewer can trace any recovery decision back to the exact substring
that fired the rule. An LLM call is not guaranteed to return the same answer twice
even at temperature 0, across model versions, or across provider-side changes
nobody here controls.

**Zero marginal cost.** The rule-based path is string matching. It costs nothing
per case. The LLM path costs money per case, however small, on every one of the
thousands of `payment.failed` events this agent processes.

**Zero network dependency.** The rule-based path has no external call to fail,
rate-limit, time out, or go down. The LLM path depends on a third-party API being
reachable, authenticated, and within budget, none of which this agent controls.

**The load-bearing reason: the false-positive direction is known and fixed for
rules, and is not for an LLM.** The rule-based classifier's failure mode is a
closed, enumerable, testable set: the exact substrings and codes in
`core/classify.py`'s rule table, exercised by `tests/test_taxonomy.py` and
`tests/test_cohort_consistency.py`. Every rule that could call a decline
non-retriable, or fail to, is visible in the source and covered by a test. An
LLM's failure mode is not enumerable the same way: it can misclassify on a
phrasing nobody wrote a test for, and that failure surface can shift under a
model or prompt change with no code diff to review. This repo's whole guardrail
story depends on non-retriable declines never being contacted. A classifier whose
false-positive set is unknown is not a safe input to that guarantee.

## What this is, plainly

`punar/core/classify_llm.py` is exploratory. It is not imported by
`core/classify.py`, not on the default path, not required for the shipped agent to
work, and gated behind `USE_LLM_DIAGNOSIS=1` plus `GROQ_API_KEY` so it cannot run
by accident. It exists so a labelled-history-based classifier can be evaluated
later, against the same guardrail it would have to clear: never call a
non-retriable decline retriable.
