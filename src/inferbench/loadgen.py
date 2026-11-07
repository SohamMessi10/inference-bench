"""The load generator: concurrent clients that record every request.

    run = asyncio.run(run_load(LoadConfig(url="http://127.0.0.1:8000", concurrency=8, duration=10)))

How it sends: **closed loop**. Each of `concurrency` workers sends a request, waits for the
answer, and sends the next. It is simple and it never overloads the client, but it has a known
blind spot: when the server slows down, the workers slow down too, so fewer requests are sent
exactly when the server is struggling, and latency under overload is understated (this is
called coordinated omission). Every results file says it was measured this way.

What it records: one record per request, failures included. A request that times out or can't
connect is a result, not something to drop; leaving failures out makes a struggling server look
better than it is.

Phases, in order:
  first    one request on its own: what a client sees after the server has been idle
  warmup   requests whose timings are recorded but excluded from results: the first requests
           to a process are slower (connections being opened, caches and memory warming up)
  measure  the results

Inputs are real held-out digit images, so every answer can be checked: a server that answers
quickly but wrongly under load is not a fast server.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
import numpy as np

from inferbench import __version__
from inferbench.models import load_digits_split
from inferbench.percentiles import percentile
from inferbench.server import MAX_IMAGES

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
LAG_PROBE_S = 0.001
MAX_CONCURRENCY = 256
MAX_DURATION = 3600


class LoadError(ValueError):
    """The load test can't be run as configured."""


@dataclass(frozen=True)
class LoadConfig:
    url: str = "http://127.0.0.1:8000"
    concurrency: int = 4
    duration: float = 10.0       # seconds of measured load
    warmup: float = 2.0          # seconds of load before measuring, not counted
    images: int = 1              # images per request
    timeout: float = 10.0        # seconds before a request counts as timed out
    seed: int = 0                # which images are sent, in which order
    backoff_ms: float = 0.0      # after a 503 (server busy or late), wait this long before sending again

    def check(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme != "http" or parts.hostname not in LOCAL_HOSTS or parts.path not in ("", "/"):
            # A benchmark of your own machine, not a way to send load to someone else's server.
            raise LoadError(f"the url must be http://127.0.0.1:PORT or http://localhost:PORT, got {self.url!r}")
        for name, value, low, high in [("concurrency", self.concurrency, 1, MAX_CONCURRENCY),
                                       ("images", self.images, 1, MAX_IMAGES)]:
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise LoadError(f"{name} must be a whole number from {low} to {high}, got {value!r}")
        for name, value, low, high in [("duration", self.duration, 0.1, MAX_DURATION),
                                       ("warmup", self.warmup, 0, MAX_DURATION),
                                       ("timeout", self.timeout, 0.01, 600),
                                       ("backoff_ms", self.backoff_ms / 1000, 0, 60)]:
            if not low <= value <= high:
                raise LoadError(f"{name} must be between {low} and {high} seconds, got {value}")


@dataclass
class Record:
    """One request and what happened to it. Times are seconds since the run started."""

    phase: str                   # "first", "warmup", or "measure"
    worker: int
    sent: float
    latency_ms: float            # until the answer arrived, or until it failed
    status: int | None           # HTTP status, or None if there was no answer
    error: str = ""              # "timeout", "connect", "protocol", "http 503", ... ; empty if fine
    images: int = 0
    correct: int | None = None   # how many predictions were right; None if there was no answer
    queue_ms: float | None = None      # the server's own timings, when it answered
    inference_ms: float | None = None
    server_batch: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error


@dataclass
class Run:
    config: LoadConfig
    server: dict                 # what /v1/model said, so results always name the model
    machine: dict
    started_at: str
    records: list[Record] = field(default_factory=list)
    measured_seconds: float = 0.0
    # How late the client's own event loop ran during the measured phase (ms). When the load
    # generator is overloaded it measures its own delay too, so this says how much to trust it.
    loop_lag: dict = field(default_factory=dict)

    def measured(self) -> list[Record]:
        return [r for r in self.records if r.phase == "measure"]


def machine_info() -> dict:
    import sklearn

    info = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "cpus": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "httpx2": httpx2.__version__, "inferbench": __version__}
    try:
        if platform.system() == "Darwin":
            import subprocess
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                         text=True, timeout=5).stdout.strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except (OSError, ValueError):
        pass
    return info


class Inputs:
    """Real held-out images, sent in a fixed shuffled order so a run can be repeated exactly."""

    def __init__(self, images_per_request: int, seed: int):
        _, X_test, _, y_test = load_digits_split()
        order = np.random.default_rng(seed).permutation(len(X_test))
        self.images, self.labels = X_test[order], y_test[order]
        self.per_request = images_per_request
        self.next = 0

    def take(self):
        n, size = self.per_request, len(self.images)
        index = [(self.next + i) % size for i in range(n)]
        self.next = (self.next + n) % size
        return self.images[index], self.labels[index]


def classify_error(exc: Exception) -> str:
    if isinstance(exc, httpx2.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx2.ConnectError):
        return "connect"
    if isinstance(exc, httpx2.RemoteProtocolError):
        return "protocol"
    return type(exc).__name__


def expected_answers(server: dict, images: np.ndarray, labels: np.ndarray) -> list[int]:
    """What a correct server says: the true digit for a real model, the documented rule for synthetic."""
    if server.get("kind") == "synthetic":
        return [int(v) for v in images.astype(np.float32).sum(axis=1).astype(np.int64) % 10]
    return [int(v) for v in labels]


async def _send(client, inputs: Inputs, server: dict, phase: str, worker: int, clock0: float,
                timeout_s: float) -> Record:
    images, labels = inputs.take()
    body = json.dumps({"inputs": images.tolist()})
    sent = time.perf_counter()
    try:
        response = await client.post("/v1/predict", content=body, headers={"content-type": "application/json"})
    except httpx2.HTTPError as exc:
        return Record(phase, worker, sent - clock0, (time.perf_counter() - sent) * 1000, None,
                      classify_error(exc), len(images))
    latency_ms = (time.perf_counter() - sent) * 1000
    record = Record(phase, worker, sent - clock0, latency_ms, response.status_code, images=len(images))
    if response.status_code == 200 and latency_ms > timeout_s * 1000:
        # The answer came, but after the client's limit; a client that gives up at that point
        # can't use it. This happens when the client's own event loop is running late: the
        # answer sits ready in the socket and is read after the deadline without the timeout
        # firing. Counted as late, not as a success, and the run's loop lag shows why.
        record.error = "late"
        return record
    if response.status_code != 200:
        record.error = f"http {response.status_code}"
        try:  # the server's own short reason, such as "busy" or "deadline", when it gives one
            reason = response.json().get("reason")
        except (ValueError, AttributeError):
            reason = None
        if isinstance(reason, str) and reason.isalpha() and len(reason) <= 20:
            record.error += f" {reason}"
        return record
    try:
        data = response.json()
        predictions, timing = data["predictions"], data["timing"]
        record.correct = sum(p == e for p, e in zip(predictions, expected_answers(server, images, labels)))
        if len(predictions) != len(images):
            record.error = "wrong number of answers"
        record.queue_ms, record.inference_ms = timing["queue_ms"], timing["inference_ms"]
        record.server_batch = timing["batch_size"]
    except (ValueError, KeyError, TypeError):
        record.error = "unreadable answer"
    return record


async def run_load(config: LoadConfig) -> Run:
    """Run one load test and return every request it made."""
    config.check()
    inputs = Inputs(config.images, config.seed)
    limits = httpx2.Limits(max_connections=config.concurrency, max_keepalive_connections=config.concurrency)
    async with httpx2.AsyncClient(base_url=config.url, timeout=config.timeout, limits=limits) as client:
        try:
            info = (await client.get("/v1/model")).json()
        except (httpx2.HTTPError, ValueError) as exc:
            raise LoadError(f"no model server answered at {config.url} ({classify_error(exc)}); "
                            f"start one with: ibench serve") from None
        run = Run(config, info, machine_info(), datetime.now(timezone.utc).isoformat(timespec="seconds"))
        clock0 = time.perf_counter()
        run.records.append(await _send(client, inputs, info, "first", 0, clock0, config.timeout))

        warmup_end = time.perf_counter() + config.warmup
        measure_end = warmup_end + config.duration

        async def worker(n: int) -> None:
            while (now := time.perf_counter()) < measure_end:
                phase = "warmup" if now < warmup_end else "measure"
                record = await _send(client, inputs, info, phase, n, clock0, config.timeout)
                run.records.append(record)
                if record.status == 503 and config.backoff_ms:
                    # A real client backs off when told the server is busy. Without this, a
                    # refused worker resends at once and the client spins at full speed.
                    await asyncio.sleep(config.backoff_ms / 1000)

        lags: list[float] = []

        async def lag_probe() -> None:
            # Ask to wake in 1 ms, see how late the wakeup is: a busy event loop wakes late.
            while (now := time.perf_counter()) < measure_end:
                await asyncio.sleep(LAG_PROBE_S)
                if now >= warmup_end:
                    lags.append(max((time.perf_counter() - now - LAG_PROBE_S) * 1000, 0.0))

        await asyncio.gather(lag_probe(), *(worker(n) for n in range(config.concurrency)))
        if lags:
            run.loop_lag = {"p50_ms": round(percentile(lags, 50), 3), "p99_ms": round(percentile(lags, 99), 3),
                            "max_ms": round(max(lags), 3), "samples": len(lags)}
        # Measured time runs from the first measured request being sent to the last measured
        # answer, both taken from the records, so the summary always agrees with them. It runs
        # past the deadline: requests sent before it count, so the time they take must too.
        measured = run.measured()
        if measured:
            run.measured_seconds = (max(r.sent + r.latency_ms / 1000 for r in measured)
                                    - min(r.sent for r in measured))
    return run


# --- results files ----------------------------------------------------------------------

def summary(run: Run) -> dict:
    measured = run.measured()
    ok = [r for r in measured if r.ok]
    errors = Counter(r.error for r in measured if not r.ok)
    answered = [r for r in ok if r.correct is not None]
    return {
        "requests": len(measured),
        "ok": len(ok),
        "errors": dict(errors),
        # Unrounded: a results file must reproduce its own numbers when read back. (Rounding
        # this to milliseconds once made throughput recomputed from a file differ by 0.1/s.)
        "measured_seconds": run.measured_seconds,
        "ok_per_second": round(len(ok) / run.measured_seconds, 1) if run.measured_seconds else 0.0,
        "images_per_second": round(sum(r.images for r in ok) / run.measured_seconds, 1) if run.measured_seconds else 0.0,
        "mean_latency_ms": round(sum(r.latency_ms for r in ok) / len(ok), 3) if ok else None,
        "accuracy": round(sum(r.correct for r in answered) / sum(r.images for r in answered), 4) if answered else None,
        "warmup_requests": sum(r.phase == "warmup" for r in run.records),
        "loop_lag": run.loop_lag,
    }


def write_results(run: Run, path: Path) -> None:
    """JSON Lines: a header describing the run, one line per request, then a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "run", "started_at": run.started_at, "config": asdict(run.config),
              "method": "closed loop: each worker waits for an answer before sending again",
              "server": run.server, "machine": run.machine}
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for record in run.records:
            out.write(json.dumps({"type": "request", **asdict(record)}) + "\n")
        out.write(json.dumps({"type": "summary", **summary(run)}) + "\n")
    tmp.replace(path)


def read_results(path: Path) -> Run:
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "run":
        raise LoadError(f"{path} isn't an ibench results file")
    header = lines[0]
    run = Run(LoadConfig(**header["config"]), header["server"], header["machine"], header["started_at"])
    run.records = [Record(**{k: v for k, v in line.items() if k != "type"}) for line in lines if line["type"] == "request"]
    final = [line for line in lines if line["type"] == "summary"]
    run.measured_seconds = final[-1]["measured_seconds"] if final else 0.0
    run.loop_lag = final[-1].get("loop_lag", {}) if final else {}
    return run
