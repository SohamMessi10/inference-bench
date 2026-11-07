"""Turning per-request records into numbers that can be trusted.

Percentiles use the nearest-rank definition: the p-th percentile of n values is the smallest
value with at least p% of the values at or below it. It is always one of the measured values,
never an interpolation, so "p99 = 31 ms" means a request really took 31 ms.

**How many samples a percentile needs.** A percentile is reported only when at least
`TAIL_SAMPLES` (10) measurements lie beyond it: n ≥ 20 for p50, 200 for p95, 1,000 for p99.
With fewer, a single unusual request decides the answer (from 100 requests, p99 is just the
slowest one), so the report says "needs N samples" instead of printing a number that looks
precise and isn't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from inferbench.loadgen import Run

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


@dataclass(frozen=True)
class RunMetrics:
    """Everything the report shows about one run."""

    name: str                  # the results file's name
    model: str
    kind: str                  # "real" or "synthetic"
    threads: int | None
    workers: int
    images: int
    requests: int
    ok: int
    errors: dict
    seconds: float
    ok_per_second: float
    images_per_second: float
    latency: dict              # {50: ms or None, 95: ..., 99: ...}
    mean_ms: float | None
    wait_ms: float | None      # mean time waiting in the server before the model ran
    model_ms: float | None     # mean time in the model
    other_ms: float | None     # the rest: HTTP, JSON, the event loop, both processes sharing the CPUs
    first_ms: float | None     # the run's first request, sent on its own
    accuracy: float | None
    machine: dict

    @property
    def error_rate(self) -> float:
        return (self.requests - self.ok) / self.requests if self.requests else 0.0


def _mean(values):
    return sum(values) / len(values) if values else None


def run_metrics(run: Run, name: str = "") -> RunMetrics:
    measured = run.measured()
    ok = [r for r in measured if r.ok]
    latencies = [r.latency_ms for r in ok]
    timed = [r for r in ok if r.queue_ms is not None and r.inference_ms is not None]
    answered = [r for r in ok if r.correct is not None]
    first = next((r for r in run.records if r.phase == "first"), None)
    seconds = run.measured_seconds
    errors: dict[str, int] = {}
    for r in measured:
        if not r.ok:
            errors[r.error] = errors.get(r.error, 0) + 1
    return RunMetrics(
        name=name,
        model=run.server.get("name", "?"),
        kind=run.server.get("kind", "?"),
        threads=run.server.get("server", {}).get("threads"),
        workers=run.config.concurrency,
        images=run.config.images,
        requests=len(measured),
        ok=len(ok),
        errors=errors,
        seconds=seconds,
        ok_per_second=len(ok) / seconds if seconds else 0.0,
        images_per_second=sum(r.images for r in ok) / seconds if seconds else 0.0,
        latency={p: reliable_percentile(latencies, p) for p in PERCENTILES} if latencies else dict.fromkeys(PERCENTILES),
        mean_ms=_mean(latencies),
        wait_ms=_mean([r.queue_ms for r in timed]),
        model_ms=_mean([r.inference_ms for r in timed]),
        other_ms=_mean([r.latency_ms - r.queue_ms - r.inference_ms for r in timed]),
        first_ms=first.latency_ms if first is not None and first.ok else None,
        accuracy=(sum(r.correct for r in answered) / sum(r.images for r in answered)) if answered else None,
        machine=run.machine,
    )

