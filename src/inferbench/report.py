"""The comparison report: one row per run, plus everything needed to judge the numbers.

Printed as text for a terminal or as Markdown to paste into documentation. Every report
ends with the conditions the numbers depend on: how load was sent, the sample-size rule
behind the percentiles, and the machines the runs came from.
"""

from __future__ import annotations

from inferbench.metrics import RunMetrics
from inferbench.percentiles import PERCENTILES, samples_needed

COLUMNS = ["run", "model", "threads", "workers", "images", "ok", "failed", "req/s", "images/s",
           "p50 ms", "p95 ms", "p99 ms", "wait ms", "model ms", "other ms", "first ms", "client lag p99"]

# The load generator counts as overloaded when its event loop ran at least this late at p99
# AND later than this share of the median latency: then a noticeable part of "latency" is the
# client's own delay. (With "or", every run warned, even 1.5 ms of lag at one worker.)
CLIENT_LAG_MS = 5.0
CLIENT_LAG_SHARE = 0.10


def client_overloaded(m: RunMetrics) -> bool:
    if m.client_lag_p99 is None:
        return False
    median = m.latency[50] or m.mean_ms or 0.0
    return m.client_lag_p99 > CLIENT_LAG_MS and m.client_lag_p99 > CLIENT_LAG_SHARE * median


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _percentile(m: RunMetrics, p: int) -> str:
    value = m.latency[p]
    if value is not None:
        return f"{value:.2f}"
    return f"n<{samples_needed(p)}" if m.ok else "-"


def row(m: RunMetrics) -> list[str]:
    model = m.model + (" (synthetic)" if m.kind == "synthetic" else "")
    failed = str(m.requests - m.ok) if m.requests == m.ok else f"{m.requests - m.ok} ({m.error_rate:.1%})"
    return [m.name, model, str(m.threads) if m.threads else "?", str(m.workers), str(m.images), str(m.ok), failed,
            f"{m.ok_per_second:.1f}", f"{m.images_per_second:.1f}",
            *(_percentile(m, p) for p in PERCENTILES),
            _ms(m.wait_ms), _ms(m.model_ms), _ms(m.other_ms), _ms(m.first_ms), _ms(m.client_lag_p99)]


def notes(runs: list[RunMetrics]) -> list[str]:
    out = [
        "Closed loop: each worker waits for an answer before sending again, so under overload "
        "latency is understated (fewer requests are sent exactly when the server is slow).",
        f"Percentiles are nearest-rank, shown only with at least 10 samples beyond them "
        f"({', '.join(f'p{p} needs {samples_needed(p):,}' for p in PERCENTILES)}); 'n<…' marks too few.",
        "Latency counts successful requests only; failures are counted separately and never dropped.",
        "wait = in the server before the model ran; model = in the model; other = the rest of the "
        "round trip (HTTP, JSON, the event loop). first = the run's first request, sent on its own.",
        "client lag = how late the load generator's own event loop ran (p99). 'late' failures are answers "
        "that arrived after the client's time limit.",
    ]
    for m in runs:
        if client_overloaded(m):
            out.append(f"{m.name}: the load generator itself was overloaded (its event loop ran {m.client_lag_p99:.1f} ms "
                       f"late at p99), so latencies include the client's own delay. Use fewer workers, "
                       f"or run the client on another machine.")
    failing = [m for m in runs if m.errors]
    for m in failing:
        out.append(f"{m.name}: {m.requests - m.ok} of {m.requests} requests failed "
                   f"({', '.join(f'{k}: {v}' for k, v in sorted(m.errors.items()))}).")
    for m in runs:
        if m.kind == "synthetic":
            out.append(f"{m.name}: a synthetic model (it waits a fixed time); not a measure of real inference.")
    machines = {(m.machine.get("cpu", "?"), m.machine.get("cpus"), m.machine.get("python")) for m in runs}
    for cpu, cpus, python in sorted(machines, key=str):
        out.append(f"Machine: {cpu}, {cpus} logical CPUs, Python {python}; client and server shared it.")
    if len(machines) > 1:
        out.append("These runs come from different machines or Python versions: compare them with care.")
    return out


def as_text(runs: list[RunMetrics]) -> str:
    table = [COLUMNS] + [row(m) for m in runs]
    widths = [max(len(r[i]) for r in table) for i in range(len(COLUMNS))]
    lines = ["  ".join(cell.ljust(w) if i < 2 else cell.rjust(w) for i, (cell, w) in enumerate(zip(r, widths)))
             for r in table]
    return "\n".join(lines + [""] + [f"- {n}" for n in notes(runs)]) + "\n"


def as_markdown(runs: list[RunMetrics]) -> str:
    def md(cells):
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    lines = [md(COLUMNS), "|" + "|".join(["---"] * 2 + ["---:"] * (len(COLUMNS) - 2)) + "|"]
    lines += [md(row(m)) for m in runs]
    return "\n".join(lines + [""] + [f"- {n}" for n in notes(runs)]) + "\n"
