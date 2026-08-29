"""Cohort self-consistency: generated error fields must re-classify to the seeded reason."""
from punar.core.classify import classify
from punar.sim.cohort import generate_cohort


def test_generated_cases_classify_back_to_seeded_reason():
    for seed in (1, 7, 42, 99):
        cases = generate_cohort(60, seed=seed)
        mismatches = [(c["case_id"], c["reason"], classify(c)[0])
                      for c in cases if classify(c)[0] != c["reason"]]
        assert not mismatches, f"seed {seed}: {mismatches[:5]}"
