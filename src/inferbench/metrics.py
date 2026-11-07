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

from dataclasses import dataclass

from inferbench.loadgen import Run

from inferbench.percentiles import PERCENTILES, reliable_percentile


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
    client_lag_p99: float | None = None  # how late the load generator's own event loop ran (ms)

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
        client_lag_p99=run.loop_lag.get("p99_ms") if run.loop_lag else None,
    )

