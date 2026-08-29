"""Uncertainty. No point estimate leaves this package without an interval.

Two levels of variation exist in this benchmark and they are reported
separately, because they answer different questions:

**Within-cohort (case-level bootstrap).** "If I had drawn a different sample of
the same size from the same population of failed payments, how much would the
answer move?" Resamples cases with replacement, keeping every arm's result for a
resampled case together, so the arms stay paired.

**Across-cohort (seed-level).** "How much does the answer depend on which cohort
I happened to generate?" Repeats the whole benchmark on `--runs` independent
seeds and reports the distribution over runs. This is the interval that matters
for a headline, and it is the one the old single-seed benchmark did not have.

Significance is tested with a **paired permutation test** on the per-case
differences: under the null that the two policies are interchangeable, the sign
of each case's difference is exchangeable, so flipping signs at random builds the
exact reference distribution. No normality assumption, and it respects the fact
that both arms see the identical case.
"""
import hashlib
import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 5000


def _rng(seed: int, tag: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{tag}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sd(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)


class Interval:
    """A point estimate that cannot be quoted without its interval."""

    __slots__ = ("point", "lo", "hi", "n", "sd", "method", "level")

    def __init__(self, point: float, lo: float, hi: float, n: int,
                 sd_: float = 0.0, method: str = "bootstrap", level: float = 0.95):
        self.point, self.lo, self.hi, self.n = point, lo, hi, n
        self.sd, self.method, self.level = sd_, method, level

    def to_dict(self) -> dict[str, Any]:
        return {"point": round(self.point, 4), "ci_lo": round(self.lo, 4),
                "ci_hi": round(self.hi, 4), "n": self.n, "sd": round(self.sd, 4),
                "method": self.method, "level": self.level}

    def fmt(self, unit: str = "", places: int = 1, scale: float = 1.0) -> str:
        f = f"{{:,.{places}f}}"
        return (f"{f.format(self.point * scale)}{unit} "
                f"[{f.format(self.lo * scale)}, {f.format(self.hi * scale)}]")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Interval({self.fmt()})"


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float] = mean,
                 n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 1234,
                 level: float = 0.95) -> Interval:
    """Percentile bootstrap CI for a statistic of one sample."""
    values = list(values)
    if not values:
        return Interval(0.0, 0.0, 0.0, 0, method="bootstrap", level=level)
    point = statistic(values)
    if len(values) == 1:
        return Interval(point, point, point, 1, method="bootstrap", level=level)
    r = _rng(seed, "bootstrap")
    n = len(values)
    draws = []
    for _ in range(n_boot):
        draws.append(statistic([values[r.randrange(n)] for _ in range(n)]))
    draws.sort()
    a = (1.0 - level) / 2.0
    return Interval(point, percentile(draws, a), percentile(draws, 1 - a), n,
                    sd(values), "percentile_bootstrap", level)


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 1234,
                        level: float = 0.95) -> Interval:
    """CI for mean(b) - mean(a) resampling *pairs*, so the arms stay paired."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_ci needs equal-length paired samples")
    if not a:
        return Interval(0.0, 0.0, 0.0, 0, method="paired_bootstrap", level=level)
    diffs = [bi - ai for ai, bi in zip(a, b, strict=False)]
    return bootstrap_ci(diffs, mean, n_boot, seed, level)


def paired_permutation_test(a: Sequence[float], b: Sequence[float],
                            n_perm: int = DEFAULT_PERMUTATIONS,
                            seed: int = 4321) -> dict[str, Any]:
    """Two-sided paired permutation (sign-flip) test on mean(b) - mean(a).

    Returns the observed difference, the p-value, and the number of
    permutations. p is computed with the +1 correction so it is never exactly 0.
    """
    if len(a) != len(b):
        raise ValueError("paired_permutation_test needs equal-length paired samples")
    diffs = [bi - ai for ai, bi in zip(a, b, strict=False)]
    n = len(diffs)
    if n == 0:
        return {"observed": 0.0, "p_value": 1.0, "n_permutations": 0, "n_pairs": 0}
    observed = mean(diffs)
    r = _rng(seed, "permutation")
    extreme = 0
    for _ in range(n_perm):
        total = 0.0
        for d in diffs:
            total += d if r.random() < 0.5 else -d
        if abs(total / n) >= abs(observed) - 1e-12:
            extreme += 1
    return {"observed": round(observed, 6),
            "p_value": round((extreme + 1) / (n_perm + 1), 6),
            "n_permutations": n_perm, "n_pairs": n}


def summarize(values: Sequence[float], seed: int = 1234, n_boot: int = DEFAULT_BOOTSTRAP,
              level: float = 0.95) -> Interval:
    """Mean with a bootstrap CI. The only sanctioned way to quote a number."""
    return bootstrap_ci(values, mean, n_boot, seed, level)


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size for the difference b - a. Reported next to p so a
    significant-but-tiny effect cannot masquerade as a big one."""
    diffs = [bi - ai for ai, bi in zip(a, b, strict=False)]
    s = sd(diffs)
    return round(mean(diffs) / s, 4) if s > 0 else 0.0
