"""The batching experiment: the same load against the same model, with and without batching.

For each mode a fresh server is started with identical settings apart from batching, and the
same closed-loop load (same images, same order, same warmup and duration) is run at each
concurrency level. Every run's results file is kept, so the comparison can be checked or
reprocessed later, and the conclusion is computed from the numbers, whichever way they go.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inferbench.loadgen import LoadConfig, run_load, write_results
from inferbench.metrics import RunMetrics, run_metrics
from inferbench.procs import ServerProcess
from inferbench.report import as_markdown

MODES = ("off", "on")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "digits-large"
    concurrency: tuple[int, ...] = (1, 4, 16, 64)
    duration: float = 5.0
    warmup: float = 1.0
    images: int = 1
    threads: int = 4
    batch_wait_ms: float = 2.0
    batch_max_images: int = 64
    fixed_ms: float = 5.0          # synthetic model only
    per_item_ms: float = 0.5

    def server_args(self, mode: str) -> list[str]:
        args = ["--model", self.model, "--threads", str(self.threads)]
        if self.model == "synthetic":
            args += ["--fixed-ms", str(self.fixed_ms), "--per-item-ms", str(self.per_item_ms)]
        if mode == "on":
            args += ["--batch", "--batch-wait-ms", str(self.batch_wait_ms),
                     "--batch-max-images", str(self.batch_max_images)]
        return args


def run_experiment(config: ExperimentConfig, models_dir: Path, out_dir: Path | None = None,
                   progress=print) -> tuple[Path, dict[str, list[RunMetrics]]]:
    if not config.concurrency or any(not 1 <= c <= 256 for c in config.concurrency):
        raise ValueError("concurrency levels must be from 1 to 256")
    out_dir = out_dir or Path("results") / f"batching-{config.model}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[RunMetrics]] = {mode: [] for mode in MODES}
    for mode in MODES:
        with ServerProcess(*config.server_args(mode), models_dir=models_dir) as server:
            for workers in config.concurrency:
                progress(f"batching {mode}, {workers} workers ...")
                run = asyncio.run(run_load(LoadConfig(url=server.url, concurrency=workers, duration=config.duration,
                                                      warmup=config.warmup, images=config.images)))
                name = f"{mode}-c{workers}"
                write_results(run, out_dir / f"{name}.jsonl")
                results[mode].append(run_metrics(run, name))
    report = render(config, results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return out_dir, results


def _change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    if before == 0:
        return "n/a"
    return f"{(after / before - 1) * 100:+.0f}%"


def comparison_rows(results: dict[str, list[RunMetrics]]) -> list[dict]:
    rows = []
    for off, on in zip(results["off"], results["on"], strict=True):
        rows.append({
            "workers": off.workers,
            "throughput_off": off.ok_per_second, "throughput_on": on.ok_per_second,
            "throughput_change": _change(off.ok_per_second, on.ok_per_second),
            "p50_off": off.latency[50], "p50_on": on.latency[50], "p50_change": _change(off.latency[50], on.latency[50]),
            "p99_off": off.latency[99], "p99_on": on.latency[99], "p99_change": _change(off.latency[99], on.latency[99]),
            "failed": (off.requests - off.ok) + (on.requests - on.ok),
        })
    return rows


def _workers(n: int) -> str:
    return f"{n} worker" if n == 1 else f"{n} workers"


def conclusion(rows: list[dict]) -> list[str]:
    """Sentences computed from the numbers: what batching did, at which load."""
    lines = []
    for r in rows:
        if r["throughput_off"] == 0 or r["throughput_on"] == 0:
            lines.append(f"{_workers(r['workers'])}: a mode had no successful requests, so there is nothing to compare.")
            continue
        ratio = r["throughput_on"] / r["throughput_off"]
        if ratio >= 1.05:
            verdict = f"batching raised throughput {ratio:.2f}×"
        elif ratio <= 0.95:
            verdict = f"batching lowered throughput to {ratio:.2f}× of unbatched"
        else:
            verdict = "batching made no clear difference to throughput (within 5%)"
        p50 = f", median latency {r['p50_off']:.2f} → {r['p50_on']:.2f} ms" if r["p50_off"] and r["p50_on"] else ""
        lines.append(f"{_workers(r['workers'])}: {verdict}{p50}.")
    return lines


def render(config: ExperimentConfig, results: dict[str, list[RunMetrics]]) -> str:
    rows = comparison_rows(results)
    out = [
        f"# Batching experiment: {config.model}",
        "",
        f"Same model, same {config.threads} model threads, same load ({config.images} image(s) per request, "
        f"{config.warmup:g} s warmup, {config.duration:g} s measured per run); the only difference is batching "
        f"(up to {config.batch_max_images} images, waiting at most {config.batch_wait_ms:g} ms).",
        "",
        "| Workers | Requests/s off | Requests/s on | Change | p50 off | p50 on | p99 off | p99 on | Failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def ms(v):
        return "-" if v is None else f"{v:.2f}"

    for r in rows:
        out.append(f"| {r['workers']} | {r['throughput_off']:.1f} | {r['throughput_on']:.1f} | {r['throughput_change']} | "
                   f"{ms(r['p50_off'])} | {ms(r['p50_on'])} | {ms(r['p99_off'])} | {ms(r['p99_on'])} | {r['failed']} |")
    out += ["", "## What it shows", ""] + [f"- {line}" for line in conclusion(rows)]
    out += ["", "## Every run", "", as_markdown(results["off"] + results["on"])]
    return "\n".join(out)
