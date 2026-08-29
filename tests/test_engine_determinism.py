"""Benchmark harness: reproducible, and Punar consistently beats naive dunning."""
from punar.sim.cohort import generate_cohort
from punar.sim.engine import compare_policies


def test_same_seed_same_result():
    cases = generate_cohort(30, seed=12)
    r1 = compare_policies(cases, "punar/config/policy.json", seed=12)
    r2 = compare_policies(cases, "punar/config/policy.json", seed=12)
    assert r1["baseline"]["recovery_rate"] == r2["baseline"]["recovery_rate"]
    assert r1["punar"]["recovery_rate"] == r2["punar"]["recovery_rate"]
    assert r1["deltas"]["extra_revenue_inr"] == r2["deltas"]["extra_revenue_inr"]


def test_punar_beats_naive_baseline_across_seeds():
    for seed in range(6):
        cases = generate_cohort(40, seed=seed)
        r = compare_policies(cases, "punar/config/policy.json", seed=seed)
        d = r["deltas"]["recovery_rate_delta_pts"]
        assert d > 0, f"seed {seed}: lift was {d}"
        assert r["punar"]["touches_per_recovery"] < r["baseline"]["touches_per_recovery"]
        assert r["punar"]["false_positive_annoyance_cost_inr"] == 0.0
        assert r["punar"]["opt_out_violations"] == 0
        assert r["baseline"]["opt_out_violations"] > 0
