"""Cost-aware intervention ranker: contextual Thompson-Sampling bandit.

Each candidate action carries a Beta(alpha, beta) arm that is sampled to
balance exploration and exploitation (Chapelle & Li, 2011 style). The arm with
the highest expected value -- P(recover|sample) x amount - channel cost -
per-channel annoyance penalty -- is selected, and if no arm clears the
expected-value floor the ranker ABSTAINS rather than spending a touch on a
case that cannot pay for it.

Posteriors are durable. Arms are seeded from the persistent store first
(`punar.core.bandit_store`) and fall back to the rule-derived PRIORS only for a
(reason, intervention) pair that has never been observed, so an update survives
the round, the case and the process. Every (arm, decision-time sample, cost,
outcome) tuple is logged for later causal-uplift (X-learner) training -- the
sample that is logged is the one the DECISION was made on, not a fresh draw.
"""
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from punar.core.bandit_store import BanditStore, store_for_policy

# Returned as the winner when no arm clears the expected-value floor.
ABSTAIN = "no_action"


def _det_rng(key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
INTERVENTIONS = [
    "silent_retry_aligned",          # silent retry aligned to customer payday evening
    "whatsapp_nudge_payment_link",   # polite WhatsApp nudge with one-tap link
    "email_payment_link",            # email with payment link + card-update option
    "voice_call",                    # human/AI voice call (high cost, high intent)
    "payment_link_sms",              # short SMS payment link (IN-only fallback)
    "promise_to_pay",                # schedule retry on promised date
    "escalate_manual",               # route to human review (no automated contact)
]

# Channel mapping keeps the ranker legible to finance reviewers.
# `silent_retry` and `internal_escalation` are REAL channels declared in
# policy.json: enabled, zero-cost and flagged non-customer-contacting. They
# were previously mapped to a channel named "none" that policy.json never
# defined, so both arms were filtered out of every case and could never run.
CHANNEL_MAP = {
    "silent_retry_aligned": "silent_retry",
    "whatsapp_nudge_payment_link": "whatsapp",
    "email_payment_link": "email",
    "voice_call": "voice",
    "payment_link_sms": "sms",
    "promise_to_pay": "whatsapp",
    "escalate_manual": "internal_escalation",
}

# Interventions that re-present the customer's instrument without contacting
# them. They consume the taxonomy's retry budget, not the daily touch cap.
SILENT_RETRY_INTERVENTIONS = frozenset({"silent_retry_aligned"})
ESCALATION_INTERVENTIONS = frozenset({"escalate_manual"})

# Fallback intrusiveness by channel when policy.json does not declare one.
DEFAULT_INTRUSIVENESS = {"silent_retry": 0.0, "internal_escalation": 0.0,
                         "email": 0.5, "sms": 0.8, "whatsapp": 1.0, "voice": 2.0}

# Literature/rule-derived priors by decline reason and intervention.
# alpha/(alpha+beta) is the prior mean recovery probability.
PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "insufficient_funds": {
        "silent_retry_aligned": (4.0, 2.0),
        "whatsapp_nudge_payment_link": (3.0, 3.0),
        "email_payment_link": (2.0, 4.0),
        "voice_call": (3.0, 3.0),
        "promise_to_pay": (3.0, 3.0),
    },
    "upi_timeout": {
        "silent_retry_aligned": (5.0, 2.0),
        "whatsapp_nudge_payment_link": (3.0, 2.0),
        "email_payment_link": (2.0, 3.0),
        "voice_call": (2.0, 2.0),
    },
    "expired_card": {
        "email_payment_link": (3.0, 3.0),
        "whatsapp_nudge_payment_link": (3.0, 3.0),
        "voice_call": (2.0, 3.0),
    },
    "bank_decline_general": {
        "silent_retry_aligned": (3.0, 3.0),
        "whatsapp_nudge_payment_link": (2.0, 3.0),
        "email_payment_link": (2.0, 3.0),
    },
    "mandate_inactive": {
        "whatsapp_nudge_payment_link": (2.0, 3.0),
        "email_payment_link": (2.0, 3.0),
        "voice_call": (2.0, 4.0),
        "escalate_manual": (1.0, 2.0),
    },
    "upi_limit_exceeded": {
        "whatsapp_nudge_payment_link": (3.0, 3.0),
        "email_payment_link": (2.0, 3.0),
        "promise_to_pay": (3.0, 3.0),
    },
    "mandate_expired": {"escalate_manual": (1.0, 3.0)},
    "invalid_cvv": {"email_payment_link": (3.0, 2.0), "whatsapp_nudge_payment_link": (3.0, 2.0)},
    "fraud_block": {"escalate_manual": (1.0, 2.0)},
    "lost_stolen_card": {"escalate_manual": (1.0, 2.0)},
    "account_closed": {"escalate_manual": (1.0, 3.0)},
}


@dataclass
class Arm:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    channel: str = ""
    cost_inr: float = 0.0
    intrusiveness: float = 1.0
    reason: str = ""
    # Sample drawn at DECISION time; update_arm logs this value, not a fresh
    # draw that had nothing to do with the action that was taken.
    last_sampled_p: float | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(max(self.alpha, 1e-6), max(self.beta, 1e-6))

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def record(self, sampled_p: float | None, context: dict[str, Any], success: bool,
               prior_mean: float | None = None) -> dict[str, Any]:
        """Log the decision/outcome pair with prior and posterior labelled correctly."""
        entry = {
            "arm": self.name,
            "reason": self.reason,
            # The sample the ranker actually acted on (None if it never ranked).
            "sampled_p": round(sampled_p, 4) if sampled_p is not None else None,
            "prior_mean": round(prior_mean if prior_mean is not None else self.mean(), 4),
            "posterior_mean": round(self.mean(), 4),
            "alpha": self.alpha, "beta": self.beta,
            "context": context, "success": success,
        }
        self.log.append(entry)
        return entry


def default_candidates(reason: str) -> list[str]:
    """Interventions that make sense for a reason, before guardrails filter them."""
    return list(PRIORS.get(reason, PRIORS["bank_decline_general"]).keys())


# --------------------------- posterior-store wiring -------------------------
_STORE: BanditStore | None = None


def set_bandit_store(store: BanditStore | None) -> None:
    """Inject (or clear) the process-wide posterior store. Used by tests."""
    global _STORE
    _STORE = store


def get_bandit_store(policy: dict[str, Any] | None = None) -> BanditStore | None:
    """Active posterior store: the injected one, else whatever policy asks for."""
    if _STORE is not None:
        return _STORE
    return store_for_policy(policy)


def read_posteriors(policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """All learned posteriors, for convergence charts and audit exports."""
    store = get_bandit_store(policy)
    return store.all_posteriors() if store is not None else []


def _prior_for(reason: str, name: str, policy: dict[str, Any]) -> tuple[float, float]:
    reason_prior = PRIORS.get(reason, PRIORS["bank_decline_general"])
    cfg = policy.get("bandit", {}) or {}
    fallback = (float(cfg.get("default_prior_success", 1.0)),
                float(cfg.get("default_prior_failure", 1.0)))
    return reason_prior.get(name, fallback)


def seed_arms(candidates: list[str], reason: str, policy: dict[str, Any],
              store: BanditStore | None = None) -> dict[str, Arm]:
    """Initialize arms from the LEARNED posterior, falling back to priors.

    An arm the store has never seen is seeded from the rule-derived PRIORS (or
    the policy's configured default prior); an arm with history is seeded from
    its stored Beta posterior, which is what makes the bandit learn at all.
    A missing/unreadable store degrades to pure priors.
    """
    arms: dict[str, Arm] = {}
    ch_cfg = policy.get("channels", {})
    store = store if store is not None else get_bandit_store(policy)
    learned: dict[str, tuple[float, float]] = {}
    if store is not None:
        learned = store.get_many(reason, candidates)
    for name in candidates:
        alpha, beta = learned.get(name) or _prior_for(reason, name, policy)
        channel = CHANNEL_MAP.get(name, "email")
        cfg = ch_cfg.get(channel, {})
        cost = float(cfg.get("cost_inr", 0.0))
        intrusive = float(cfg.get("intrusiveness",
                                  DEFAULT_INTRUSIVENESS.get(channel, 1.0)))
        arms[name] = Arm(name=name, alpha=float(alpha), beta=float(beta), channel=channel,
                         cost_inr=cost, intrusiveness=intrusive, reason=reason)
    return arms


def _annoyance_penalty(touches: list[dict[str, Any]], weight: float = 25.0,
                       intrusiveness: float = 1.0) -> float:
    """Rising marginal annoyance cost, scaled by how intrusive the arm's channel is.

    Superlinear in the touch count (n**1.6): the fourth message annoys far more
    than the first. Scaling by channel intrusiveness is what makes the term
    actually change the argmax -- a flat penalty is a constant offset applied to
    every arm and can never reorder them.
    """
    return weight * (len(touches) ** 1.6) * float(intrusiveness)


def annoyance_weight(policy: dict[str, Any] | None = None) -> float:
    return float(((policy or {}).get("bandit", {}) or {}).get("annoyance_weight_inr", 25.0))


def ev_floor(policy: dict[str, Any] | None = None) -> float:
    """Expected value below which the agent declines to act at all."""
    return float(((policy or {}).get("bandit", {}) or {}).get("min_expected_value_inr", 0.0))


def expected_value(arm: Arm, context: dict[str, Any], sampled_p: float,
                   weight: float = 25.0) -> float:
    amount = float(context.get("amount_inr", 0))
    recovery_value = sampled_p * amount
    annoyance = _annoyance_penalty(context.get("touches", []), weight, arm.intrusiveness)
    return recovery_value - arm.cost_inr - annoyance


def rank_intervention(candidates: list[str], context: dict[str, Any],
                      policy: dict[str, Any], rng: random.Random,
                      reason: str,
                      arms: dict[str, Arm] | None = None,
                      ) -> tuple[str, list[Arm], list[dict[str, Any]]]:
    """Score arms by sampled EV and return (winner_name, arms, score_records).

    Pass `arms` to carry live posteriors across rounds of the same episode;
    omit it and arms are seeded from the posterior store / priors. The winner
    is `ABSTAIN` when the best expected value is below the policy floor.
    """
    weight = annoyance_weight(policy)
    floor = ev_floor(policy)
    live = dict(arms) if arms else {}
    missing = [c for c in candidates if c not in live]
    if missing:
        live.update(seed_arms(missing, reason, policy))
    ordered = [live[c] for c in candidates if c in live]

    records: list[dict[str, Any]] = []
    scored: list[tuple[float, str]] = []
    for arm in ordered:
        p = arm.sample(rng)
        arm.last_sampled_p = p                 # the sample the decision is made on
        ev = expected_value(arm, context, p, weight)
        scored.append((ev, arm.name))
        records.append({"intervention": arm.name, "channel": arm.channel,
                        "sampled_p": round(p, 4), "prior_mean": round(arm.mean(), 4),
                        "cost_inr": arm.cost_inr, "intrusiveness": arm.intrusiveness,
                        "expected_value": round(ev, 2)})
    if not scored:
        return ABSTAIN, ordered, records
    scored.sort(key=lambda x: x[0], reverse=True)
    best_ev, winner = scored[0]
    if best_ev < floor:
        # Abstention arm: acting would destroy value, so decline to act.
        records.append({"intervention": ABSTAIN, "channel": "", "sampled_p": None,
                        "prior_mean": None, "cost_inr": 0.0, "intrusiveness": 0.0,
                        "expected_value": round(floor, 2),
                        "detail": f"best expected value {best_ev:.2f} < floor {floor:.2f}"})
        return ABSTAIN, ordered, records
    return winner, ordered, records


def update_arm(arms: dict[str, Arm], intervention: str, success: bool,
               context: dict[str, Any],
               store: BanditStore | None = None,
               policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bayesian update of the selected arm from an observed outcome.

    Writes through to the durable posterior store so the update survives the
    round, the case and the process, and logs the DECISION-time sample joined
    to the outcome it produced (with prior and posterior means labelled
    distinctly) rather than a fresh draw taken after the update.
    """
    arm = arms.get(intervention)
    if arm is None:
        return {}
    sampled_p = arm.last_sampled_p             # what the ranker actually acted on
    prior_mean = arm.mean()                    # BEFORE the update
    if success:
        arm.alpha += 1.0
    else:
        arm.beta += 1.0
    store = store if store is not None else get_bandit_store(policy)
    if store is not None and arm.reason:
        alpha, beta = store.record_outcome(arm.reason, intervention, success,
                                           prior=(arm.alpha - (1.0 if success else 0.0),
                                                  arm.beta - (0.0 if success else 1.0)))
        arm.alpha, arm.beta = alpha, beta
    return arm.record(sampled_p, context, success, prior_mean=prior_mean)
