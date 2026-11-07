"""Percentiles, defined once and computed exactly (imported by the load generator and the metrics).

Nearest-rank: the p-th percentile of n values is the smallest value with at least p% of the
values at or below it, so it is always a value that was really measured.
"""

from __future__ import annotations

import math
from fractions import Fraction

TAIL_SAMPLES = 10
PERCENTILES = (50, 95, 99)


def _exact(p: float) -> Fraction:
    # Percentiles are written as decimals (95, 99.9). In floating point 1 - 99.9/100 isn't
    # exactly 0.001, which once made p99.9 ask for 10,001 samples instead of 10,000; working
    # with the decimal as written avoids that.
    return Fraction(str(p))


def samples_needed(p: float) -> int:
    """The smallest n with at least TAIL_SAMPLES values above the p-th percentile."""
    if not 0 < p < 100:
        raise ValueError(f"percentile must be between 0 and 100, got {p}")
    return math.ceil(TAIL_SAMPLES / (1 - _exact(p) / 100))


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    if not values:
        raise ValueError("no values")
    if not 0 < p <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    ordered = sorted(values)
    rank = math.ceil(_exact(p) / 100 * len(ordered))  # exact, so 95% of 200 is 190, not 191
    return ordered[max(rank, 1) - 1]


def reliable_percentile(values: list[float], p: float) -> float | None:
    """The percentile, or None when there are too few values to support it."""
    return percentile(values, p) if len(values) >= samples_needed(p) else None
