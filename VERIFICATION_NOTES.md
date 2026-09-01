# Verification notes: what broke, and how it was found

This file exists because "what broke during development and how you recovered from it"
is worth more than a clean-looking history. Everything below was a real defect in this
repository, found by auditing the code against its own claims. Each entry names the
defect, how it was detected, and the test that now prevents it.

---

## 1. The benchmark was measuring the author's thumb, not the policy

**The claim:** baseline 43.2% vs Punar 81.2%, "+38.0 pts", "+INR 925,905".

**The defect:** `_rate()` scored the two arms through *different branches*. Every
baseline touch took a flat −0.35 absolute probability penalty (`MISALIGN_PENALTY +
INSTANT_RETRY_PENALTY`) that Punar never paid, and `aligned` was hardcoded `False` for
all four baseline actions, so the penalty always applied. The same action on the same
case scored **0.030 for the baseline and 0.420 for Punar**. A 14× difference produced
by nothing but which arm was asking.

**How it was found:** re-scoring the identical naive baseline through Punar's own
branch. The headline collapsed from **+38.0 pts to +5.2 pts**. 86% of the claim was one
constant.

**Compounding it:** the fatigue counter was asymmetric. The baseline used one global
counter per case, so its fourth touch was multiplied by `0.75³ = 0.42`; Punar keyed the
counter by `(case_id, intervention)`, so because it rotated channels, 75% of its touches
were scored at `rnd=1` with no decay at all. Worth another 6.4 points.

**And the agent had the answer key.** The simulator's success table (`BASE_RATES`) and
the agent's decision priors (`PRIORS`) were two hand-written tables by the same author,
with **identical argmax on 11 of 11 reasons**. The agent was graded by the function it
optimised.

**The fix:** one world model (`punar/sim/world.py`), one scoring path, no per-arm
branches. A do-nothing control, a realistic merchant comparator, two ablation arms,
bootstrap CIs over 20+ seeds, and paired permutation tests. Every modelling constant
moved to `punar/sim/params.py` and is printed with `--show-params`.

**The honest result:** Punar +0.8 pts vs a realistic baseline (not significant), +21.9
pts vs naive dunning, +5.8 pts over its own taxonomy+guardrails ablation. Smaller
numbers, and they mean something.

---

## 2. Two of the three headline features never executed

**`silent_retry_aligned` and `escalate_manual` were unreachable.** `CHANNEL_MAP` mapped
both to channel `"none"`, and `policy.json` defined only `whatsapp/email/voice/sms`, so
`channel_enabled(policy, "none")` returned `False` and the guard skipped them on every
case. Payday-aligned silent retry, the feature the README led with, had never fired
once. Worse, `fraud_block` and `mandate_expired` had `escalate_manual` as their *only*
candidate, so they were written off silently instead of escalated to a human.

**The fix:** `silent_retry` and `internal_escalation` are real channels in
`policy.json`, declaring `contacts_customer: false` and an explicit `exempt_from` list.
The gate reads those exemptions rather than special-casing channel names.
Test: `test_silent_retry_is_reachable_and_does_not_contact_the_customer`.

**The bandit never learned.** `rank_intervention()` called `seed_arms()` on every
invocation, re-initialising every arm from a frozen `PRIORS` dict. `update_arm()`
wrote posteriors into objects discarded before the next decision, not just across cases
but *across rounds within one case*. Probe: 50 recorded wins drove α to 52.0; the very
next call returned α=2.0. Thompson sampling with no posterior is a seeded coin.

**The fix:** `punar/core/bandit_store.py` persists `(reason, intervention) → (α, β)`;
`seed_arms` loads it and falls back to priors only for unseen arms.
Test: `test_posterior_survives_across_separate_ranking_calls`.

---

## 3. The policy judge was a log line, not a control

`act()` rendered the copy, ran the judge, and on rejection retried in English, then
appended the touch with `"delivered": True` **regardless of the outcome**. A message the
judge had rejected was sent anyway.

The judge was also English-only on a product whose templates are Hindi and Hinglish.
These all passed clean: *"Bhugtan nahi kiya to hum kanooni karyavahi karenge"* (a legal
threat), *"Turant paisa bhejo warna gharwalon ko batayenge"* (threatening to tell the
customer's family, textbook RBI recovery harassment). A curly apostrophe defeated it:
`"won't"` was blocked, `"won’t"` was not. Meanwhile "toll-free helpline" tripped the
unapproved-discount rule.

**The fix:** on final rejection nothing is sent, no touch is recorded, and the case is
escalated to a human. Devanagari and romanized-Hindi patterns added; text is
NFKC-normalised and quotes folded before matching; false-positive patterns tightened.
Tests: `test_policy_judge_blocks_the_send_and_escalates`, plus non-English judge tests.

---

## 4. The audit trail was not append-only

`audit.py` opened with the docstring *"Append-only SQLite audit trail"*. Nine lines
later `upsert()` ran `UPDATE decisions SET data = ?`, and `upsert` was the **only**
write the API ever performed. Two writes for one case left one row, and the earlier
version was unrecoverable. `clear()` was an unrestricted `DELETE FROM decisions`
exposed as a public method, and the test suite asserted the overwrite behaviour as
correct. `ORDER BY created_at` sorted a caller-supplied column, so a webhook carrying
`created_at="9999-12-31"` could decide which revision was "latest".

**The fix:** every state change appends a new row; `BEFORE UPDATE` / `BEFORE DELETE`
triggers block in-place mutation even from a `sqlite3` shell; rows are hash-chained via
`prev_hash` and verified by `verify_chain()`; timestamps are server-assigned and
ordering is by monotonic row id; `clear()` raises; `prune()` is the only sanctioned
delete path and keeps the chain verifiable.
Tests: `test_verify_chain_detects_out_of_band_tampering`,
`test_verify_chain_detects_deletion_of_the_head_row`.

---

## 5. The service accepted unsigned webhooks and leaked PII

`verify_signature()` **failed open**: with no `RAZORPAY_WEBHOOK_SECRET` configured it
logged a warning and returned `True`. Verified live: a request with no signature and
one with a garbage signature both returned `202`. No endpoint had authentication, and
`GET /cases/{case_id}` returned `customer_id`, amount, and the full rendered message
body to anyone who could guess a Razorpay payment id.

`load_policy()` sat *outside* the handler's `try`, so a misconfigured deploy returned
`202 {"accepted": true}` and `/health: ok` while silently dropping every payment. A
non-object JSON body (`[1,2,3]`) and a string `amount` each produced an unhandled 500.
There was no idempotency at all, so Razorpay's own webhook retries would double-contact
the customer and break the documented touch cap.

**The fix:** fail closed with an explicit dev escape hatch; API-key auth with a separate
PII scope; startup config validation that refuses to boot; a real Pydantic model of the
Razorpay envelope; idempotency keyed on the event id; a durable job queue with leases,
retries and dead letters; body-size limits, rate limiting and security headers; real
`/health` and `/ready` dependency checks.
Tests: `tests/test_api_server.py` (23 tests).

---

## 6. Smaller, but load-bearing

- **`pip install .` produced a broken package.** `punar/config/policy.json` was not
  packaged, and every code path loads it. The installed wheel raised `FileNotFoundError`.
- **A hardcoded `timedelta(hours=26)`** advanced the clock between rounds, putting every
  touch on its own calendar day and thereby disabling the daily touch cap, the
  inter-touch gap and the retry budget simultaneously. Replaced with per-reason backoff
  from `policy.json` (`insufficient_funds` waits 72h for a payday credit;
  `upi_timeout` retries in 4h).
- **Timezone bugs in the gate:** the daily cap bucketed dates in the timestamp's own
  offset rather than IST, letting a fourth same-IST-day touch through a cap of 3; a
  touch with `date` but no `timestamp` crashed with `AttributeError`.
- **The classifier read the wrong payload shape.** It expected a nested
  `failure["error"]["code"]`, but a real Razorpay `payment.failed` entity carries flat
  `error_code` / `error_description` / `error_reason`, so every real webhook fell
  through to the catch-all. It also misrouted `"closed loop wallet not supported"` to
  `account_closed` (non-retryable), permanently writing off a recoverable payment.
- **The LangGraph runner had never run.** `graph.invoke()` was passed the *function*
  returned by the outcome-injection helper instead of a state dict, and that helper ran
  the whole loop internally, making the graph a no-op wrapper. `langgraph` was not
  installed and the path had no real test. **Removed** rather than shipped, see the
  note in `punar/core/agent.py`.
- **Tautological tests.** `assert a != b or True` cannot fail; a test named
  `test_annoyance_grows_sublinearly_with_touches` asserted only monotonicity over a
  function that is superlinear (`n**1.6`); `test_non_retriable_never_contacted` passed
  only because of the unreachable-channel bug above, and would have kept passing if the
  non-retriable guardrail were deleted outright. All replaced with real assertions.
- **`.gitignore` did not match the audit database.** It listed `audit.db`; the actual
  default filename is `punar_audit.db`, so a database full of customer PII would have
  been committed.

---

## Current state

```
84 tests passing
ruff: clean
benchmark: 250 cases x 20 seeds, reproducible, PYTHONHASHSEED-independent
```

Reproduce the headline numbers:

```bash
python scripts/benchmark.py --n-cases 250 --seed 42 --seeds 20
```
