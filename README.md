# Punar: autonomous recovery agent for failed Razorpay payments

**Punar** (Sanskrit: *"again", "reborn"*) turns failed Razorpay e-mandate / UPI / card
payments into successful ones, while staying inside RBI Fair-Practices and
digital-lending guardrails.

It classifies why a payment declined, decides whether contacting the customer is
permitted at all, picks a channel and a retry time, drafts compliant copy in the
customer's language, **runs that copy through a pre-send policy judge that can and does
block the send**, observes the result, and updates a per-reason Beta posterior that
survives the process.

Built for the Razorpay Buildathon, **Track 03, AI Revenue Recovery**:
*"measured money recovered across a batch, with compliant escalation, stopping rules,
and an audit trail."*

---

## The honest headline

Over **250 cases × 20 independent cohorts**, every arm scored by the *same* world model:

| arm | recovery rate | touches / recovery | contacted after opt-out | contacted on non-retriable | net revenue |
|---|---|---|---|---|---|
| do nothing (organic self-cure) | 15.4% `[14.5, 16.4]` | 0.00 | 0 | 0 | ₹333,012 |
| naive dunning | 16.4% `[15.4, 17.4]` | 24.55 | **27** | **40** | ₹361,023 |
| realistic merchant baseline | 37.5% `[36.4, 38.6]` | 8.41 | 0 | **29** | ₹799,983 |
| ablation: taxonomy only | 33.7% `[32.4, 35.0]` | 14.26 | **69** | 0 | ₹688,093 |
| ablation: + guardrails | 32.5% `[31.2, 33.8]` | 12.03 | 0 | 0 | ₹667,753 |
| **Punar** | **38.3%** `[37.1, 39.7]` | **7.79** | **0** | **0** | ₹758,088 |

Paired over seeds, Punar vs each comparator:

| comparator | recovery lift | p | verdict |
|---|---|---|---|
| do nothing | **+22.9 pts** `[+21.7, +24.1]` | 0.0002 | significant |
| naive dunning | **+21.9 pts** `[+20.6, +23.2]` | 0.0002 | significant |
| realistic merchant baseline | +0.8 pts `[−0.5, +2.1]` | 0.26 | **not significant** |
| ablation: taxonomy only | **+4.6 pts** `[+3.3, +5.9]` | 0.0002 | significant |
| ablation: + guardrails | **+5.8 pts** `[+4.6, +7.0]` | 0.0002 | significant |

**Read that third row honestly: Punar does not beat a well-run merchant baseline on
recovery rate.** It matches it (+0.8 pts, inside the confidence interval), and the
difference it actually makes is elsewhere:

- **It gets there on 7% fewer touches** (7.79 vs 8.41 per recovery), the same revenue
  for less of the customer's attention.
- **It makes zero contacts it should not have made.** The realistic baseline contacts
  customers on 29 non-retriable declines per cohort: fraud blocks, closed accounts,
  lost/stolen cards. Punar contacts none, because the gate blocks them and routes them
  to a human instead.
- **The ablation shows every component earns its place.** Taxonomy routing alone gets
  33.7%; adding guardrails *costs* 1.2 points of recovery (guardrails are not free) but
  removes all 69 opt-out violations; adding the learned bandit recovers 5.8 points back.
- Punar's net revenue is ₹42k *below* the realistic baseline, because escalating a
  non-retriable decline to a human costs ₹45 of ops time. That is a real cost of being
  compliant, and it is reported rather than hidden.

Reproduce all of it:

```bash
python scripts/benchmark.py --n-cases 250 --seed 42 --seeds 20
```

> **These are simulated outcomes under stated priors, not measured Razorpay results.**
> The simulator, the assumptions and their justifications live in `punar/sim/params.py`
> and are printed with `--show-params`. Arms are comparable to each other and to nothing
> else. Nothing in this repository has been calibrated against real merchant data.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                          # 84 tests
python scripts/benchmark.py --seeds 3      # fast benchmark
streamlit run app.py               # local console: one case, batch results, audit trail
```

### Watch it recover one payment

```bash
python -m punar.main recover --reason INSUFFICIENT_FUNDS --outcomes false false true
```

```
diagnosis : insufficient_funds (retryable)
outcome   : recovered  [recovered_by_silent_retry_aligned]

decision trail
  2026-08-29T05:40  diagnose  {"reason": "insufficient_funds", ...}
  2026-08-29T05:40  guard     {"allowed": ["silent_retry_aligned"],
                               "verdict": {"intervention": "promise_to_pay",
                                           "allowed": false,
                                           "code": "outside_contact_window"}}
  ...
actions taken
  #1 silent_retry_aligned         silent_retry   [internal] reattempted_at_psp
  #2 whatsapp_nudge_payment_link  whatsapp       [customer] simulated
  #3 silent_retry_aligned         silent_retry   [internal] reattempted_at_psp
```

Note the first guard verdict: it was 05:40 IST, outside the 08:00–19:00 contact window,
so outreach was refused and the next attempt was scheduled into the window.

### Watch it refuse

```bash
python -m punar.main recover --reason FRAUD --error "fraud block"        # never contacted
python -m punar.main recover --reason INSUFFICIENT_FUNDS --opted-out     # never contacted
```

### Run the service

```bash
cp .env.example .env                       # then fill in the secrets
export PUNAR_ALLOW_UNVERIFIED_WEBHOOKS=1   # local development ONLY
export PUNAR_ALLOW_UNAUTHENTICATED_READS=1 # local development ONLY
uvicorn punar.api.server:app --reload
```

| endpoint | purpose | auth |
|---|---|---|
| `POST /webhooks/razorpay` | Razorpay `payment.failed` receiver | HMAC signature |
| `GET /health`, `GET /ready` | real dependency checks (policy, audit chain, queue) | none |
| `GET /cases/{case_id}` | latest audit revision for a case | API key |
| `GET /cases/{case_id}/history` | every revision, the append-only trail | API key |
| `GET /jobs/{job_id}` | recovery-run status, including dead letters | API key |
| `GET /audit/verify` | verify the audit hash chain end to end | API key |
| `GET /bandit/posteriors` | what the agent has learned so far | API key |
| `GET /stats` | business counters with provider provenance | API key |
| `GET /metrics` | Prometheus exposition format | none |
| `GET /docs` | interactive OpenAPI | none |

---

## How it works

```
Razorpay payment.failed
        │
        ▼
  ┌──────────┐   11-class taxonomy from error_code + description + method,
  │ diagnose │   accepting both the flat and nested Razorpay shapes.
  └────┬─────┘   Unrecognised codes get a low-confidence flag, not a free retry.
       ▼
  ┌──────────┐   opt-out · non-retriable · contact window (IST) · daily touch cap ·
  │  guard   │   inter-touch gap · daily cost cap.  Every refusal is logged with a
  └────┬─────┘   typed code.  Non-contacting channels declare their exemptions in
       │         policy.json rather than in a special case in the code.
       ▼
  ┌──────────┐   Thompson sampling over a Beta posterior per (reason, intervention),
  │   plan   │   scored on expected value: p × amount − channel cost − annoyance,
  └────┬─────┘   where annoyance scales with the channel's intrusiveness.
       │         Below the value floor it selects `no_action` and stops.
       ▼
  ┌──────────┐   Render approved copy (en / hi / hinglish) → POLICY JUDGE.
  │   act    │   If the judge rejects, nothing is sent, no touch is recorded, and
  └────┬─────┘   the case is escalated to a human.
       ▼
  ┌──────────┐   Observe the outcome, update the Beta posterior, write it through
  │ observe  │   to the durable store so the learning survives the process.
  └────┬─────┘
       ▼
  ┌──────────┐   Recovered → stop.  Budget spent → stop.  Otherwise schedule the
  │  decide  │   next attempt from the reason's backoff and the contact window.
  └──────────┘
```

Every transition appends to a hash-chained audit trail.

| module | responsibility |
|---|---|
| `punar/core/taxonomy.py` | the 11 decline classes, their retriability and retry budgets |
| `punar/core/classify.py` | error code + text + method → reason, with a confidence flag |
| `punar/core/gate.py` | the six-check pre-action guardrail; returns typed verdicts |
| `punar/core/select.py` | Thompson-sampling arm selection with an abstention arm |
| `punar/core/bandit_store.py` | durable Beta posteriors, keyed by (reason, intervention) |
| `punar/core/copy.py` | multilingual templates + the pre-send policy judge |
| `punar/core/agent.py` | the state machine: diagnose, guard, plan, act, observe, decide |
| `punar/audit.py` | append-only, hash-chained, tamper-evident audit store |
| `punar/api/server.py` | FastAPI service: webhook, reads, health, metrics |
| `punar/api/config.py` | validated settings; refuses to start when misconfigured |
| `punar/api/jobs.py` | durable job queue with leases, retries and a dead-letter path |
| `punar/api/providers.py` | channel / consent / profile / outcome interfaces + stubs |
| `punar/sim/world.py` | the world model, one scoring path for every arm |
| `punar/sim/arms.py` | do-nothing, naive, realistic, ablations, Punar |
| `punar/sim/params.py` | every modelling constant, with its justification |
| `punar/sim/stats.py` | bootstrap CIs, paired permutation tests, effect sizes |
| `punar/config/policy.json` | the single source of truth for guardrails, channels, costs, backoffs |

---

## Compliance, and where it is enforced

| guarantee | enforced at | test |
|---|---|---|
| No contact outside 08:00–19:00 IST | `gate.py` rule 3 | `test_gate.py` |
| Max 3 customer touches/day, 3 outreach touches/case | `gate.py` rule 4, `agent.py` `decide` | `test_agent_loop.py` |
| Opt-out / DND honoured immediately | `gate.py` rule 1 | `test_agent_loop.py`, `test_api_server.py` |
| Non-retriable declines never contacted | `gate.py` rule 2 | `test_agent_loop.py` |
| Every message passes a pre-send judge | `agent.py` `act` | `test_agent_loop.py` |
| No legal threats / shaming / third-party disclosure, in **any** of en/hi/hinglish | `copy.py` | `test_copy.py` |
| Every decision reconstructable | `audit.py` | `test_audit_store.py` |

The policy judge is a control, not a log line: on rejection the message is **not sent**,
no touch is recorded, and the case is escalated. It matches Devanagari and romanized
Hindi as well as English, and normalises Unicode first. `"You won't…"` with a curly
apostrophe is caught exactly like the ASCII form.

The audit store is append-only **in fact**: `BEFORE UPDATE` / `BEFORE DELETE` triggers
make in-place mutation impossible even from a `sqlite3` shell, each row is chained by
`prev_hash`, and `GET /audit/verify` detects edits, deletions and reorderings.

```bash
python -m punar.main audit --db punar_audit.db
```

---

## What is real and what is not

A reviewer should not have to guess which parts are load-bearing.

**Real:** the taxonomy, the guardrail engine, the policy judge, the bandit and its
persistence, the audit chain, the job queue, webhook signature verification, the
API surface, and the simulator with its statistics.

**Stubbed, and labelled as stubbed at runtime:**

| gap | today | production path |
|---|---|---|
| Message delivery | `StubChannelSender` records what *would* be sent and returns `delivered=False` | WhatsApp Business / SMS gateway / SES behind the `ChannelSender` protocol |
| Outcome observation | `StubOutcomeObserver`, a seeded coin flip | correlate `payment.captured` webhooks back to the preceding touch |
| Customer contact details | `StubCustomerProfileLookup` returns none | Razorpay Customer API or the merchant's CRM |
| Consent / DND registry | `StubConsentLookup` reads a configured list | the merchant's consent ledger + DNCR |
| Payment link creation | a constructed `razorpay.me` URL | Razorpay Payment Links API |
| Scheduling across days | in-process job queue | Celery / Temporal with a delayed-execution store |

No response ever reports a simulated send as delivered. `/health`, `/stats` and every
audit record carry `providers.*.simulated`, so provenance travels with the number.

The decline classifier is rule-based, not an LLM. An optional LLM variant and the
measurements behind that choice live in [`docs/why_not_llm.md`](docs/why_not_llm.md);
it is off by default and not required for anything here to run.

---

## Tests

```bash
pytest -q                        # 84 tests
pytest tests/test_gate.py -q     # guardrail boundaries
pytest tests/test_copy.py -q     # policy judge, all three languages
pytest tests/test_audit_store.py -q   # append-only + tamper detection
pytest tests/test_api_server.py -q    # auth, validation, idempotency
```

The suite asserts guarantees, not implementation details: that a rejected message is
never sent, that a redelivered webhook never drives a second recovery run, that a
tampered audit row is detected, that an opted-out customer is never contacted, and that
a learned posterior survives across separate ranking calls.

---

## Reproducibility

Every random draw is derived by SHA-256 from `(seed, case_id, intervention, round)`.
The benchmark clock is pinned. Nothing on the benchmark path calls `datetime.now()`,
uses the unseeded global `random`, or depends on dict ordering. Results are identical
across processes and across `PYTHONHASHSEED` values.

## Docker

```bash
docker build -t punar .
docker run -p 8000:8000 -v punar-data:/data \
  -e RAZORPAY_WEBHOOK_SECRET=... -e PUNAR_API_KEYS=... punar
```

## License

MIT: see [LICENSE](LICENSE).
