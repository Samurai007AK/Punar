"""Thompson-sampling ranker: deterministic sampling, Bayesian updates."""
import random

from punar.core.gate import load_policy
from punar.core.select import Arm, rank_intervention, seed_arms, update_arm

POLICY = load_policy("punar/config/policy.json")


def test_arm_stats():
    a = Arm("x", alpha=3.0, beta=1.0, cost_inr=0.5)
    assert a.mean() == 0.75
    s = a.sample(random.Random(42))
    assert 0.0 <= s <= 1.0


def test_ranking_deterministic_given_seed():
    ctx = {"amount_inr": 5000.0, "reason": "insufficient_funds", "touches": []}
    r1 = rank_intervention(["email_payment_link", "whatsapp_nudge_payment_link"],
                           ctx, POLICY, random.Random(7), "insufficient_funds")
    r2 = rank_intervention(["email_payment_link", "whatsapp_nudge_payment_link"],
                           ctx, POLICY, random.Random(7), "insufficient_funds")
    assert r1[0] == r2[0] and r1[2] == r2[2]


def test_thompson_sampling_explores_across_seeds():
    """A real assertion about exploration, not `assert x or True`.

    Thompson sampling must sometimes pick the arm that is not the prior
    favourite -- otherwise it is an argmax over a fixed table and can never
    discover that its prior is wrong. Over many seeds we expect BOTH arms to
    win at least once, and the prior favourite to win the majority.
    """
    ctx = {"amount_inr": 5000.0, "reason": "insufficient_funds", "touches": []}
    winners = [rank_intervention(["email_payment_link", "voice_call"], ctx, POLICY,
                                 random.Random(seed), "insufficient_funds")[0]
               for seed in range(200)]
    distinct = set(winners)
    assert len(distinct) == 2, f"no exploration at all: always picked {distinct}"
    for arm in ("email_payment_link", "voice_call"):
        assert winners.count(arm) >= 5, f"{arm} was effectively never explored"


def test_annoyance_penalty_can_reorder_arms():
    """A penalty identical across arms is a constant offset and cannot reorder.

    Scaling by channel intrusiveness is what makes the term do any work: as a
    customer's attention is spent, a cheap low-intrusion channel must be able
    to overtake an expensive intrusive one.
    """
    from punar.core.select import _annoyance_penalty
    fresh = _annoyance_penalty([{}, {}], intrusiveness=0.5)
    intrusive = _annoyance_penalty([{}, {}], intrusiveness=2.0)
    assert intrusive > fresh, "penalty does not vary with channel intrusiveness"


def test_arm_update_is_bayesian():
    arms = seed_arms(["email_payment_link"], "expired_card", POLICY)
    before = arms["email_payment_link"].alpha, arms["email_payment_link"].beta
    update_arm(arms, "email_payment_link", True, {"amount_inr": 100})
    after = arms["email_payment_link"].alpha, arms["email_payment_link"].beta
    assert after[0] == before[0] + 1 and after[1] == before[1]
    update_arm(arms, "email_payment_link", False, {"amount_inr": 100})
    final = arms["email_payment_link"].alpha, arms["email_payment_link"].beta
    assert final[1] == before[1] + 1


def test_annoyance_grows_superlinearly_with_touches():
    """Named for what the function does: weight * n**1.6.

    The fourth message annoys far more than the first, so the penalty must grow
    faster than linearly -- the old test asserted only monotonicity under a name
    claiming the opposite.
    """
    from punar.core.select import _annoyance_penalty
    one = _annoyance_penalty([{}])
    two = _annoyance_penalty([{}, {}])
    four = _annoyance_penalty([{}] * 4)
    assert two > one and four > two
    assert four > 4 * one, "penalty is not superlinear in the touch count"


def test_posterior_survives_across_separate_ranking_calls(tmp_path):
    """The bug this replaces: rank_intervention re-seeded arms from PRIORS on
    every call, so 50 recorded wins were discarded before the next decision."""
    import random as _random

    from punar.core.bandit_store import BanditStore
    from punar.core.select import default_candidates, rank_intervention, set_bandit_store

    store = BanditStore(str(tmp_path / "bandit.db"))
    set_bandit_store(store)
    try:
        for _ in range(30):
            store.record_outcome("insufficient_funds", "email_payment_link", True)
        cands = default_candidates("insufficient_funds")
        _, arms, _ = rank_intervention(cands, {"amount_inr": 5000, "touches": []},
                                       POLICY, _random.Random(1), "insufficient_funds")
        learned = {a.name: a for a in arms}["email_payment_link"]
        assert learned.alpha >= 31.0, "the learned posterior was not reloaded"
    finally:
        set_bandit_store(None)


def test_agent_declines_to_act_when_every_option_destroys_value():
    """No abstention arm existed: a zero-value case still got contacted."""
    import random as _random

    from punar.core.select import ABSTAIN, default_candidates, rank_intervention

    # Only the customer-contacting arms: a free silent PSP retry legitimately
    # has zero expected value and is never worth abstaining from.
    contacting = [c for c in default_candidates("insufficient_funds")
                  if c not in ("silent_retry_aligned", "escalate_manual")]
    ctx = {"amount_inr": 0.0, "touches": [{"channel": "whatsapp"}] * 20}
    winner, _, _ = rank_intervention(contacting, ctx, POLICY,
                                     _random.Random(5), "insufficient_funds")
    assert winner == ABSTAIN
