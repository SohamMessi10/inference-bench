"""Cold start: what the first client sees when a server has only just started.

For each trial a fresh server process is started, and three things are timed:

  startup     from starting the process until /health answers (imports, loading weights)
  first       the first prediction request
  steady      the median of the next `steady_requests` requests, sent one after another

A load test's "first" request is different: by then the server may have been running and
warm for a while. This is the number a user hits after a deploy or a restart.
"""

from __future__ import annotations

import json
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from inferbench.loadgen import machine_info
from inferbench.models import DIGITS_MODELS, load_digits_split


class ColdStartError(RuntimeError):
    """A server couldn't be started or didn't answer."""


@dataclass(frozen=True)
class Trial:
    startup_s: float
    first_ms: float
    steady_ms: float


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post(url: str, body: bytes, timeout: float) -> float:
    request = urllib.request.Request(url, data=body, method="POST", headers={"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
        if response.status != 200:
            raise ColdStartError(f"the server answered {response.status}")
    return (time.perf_counter() - started) * 1000


def one_trial(model: str, models_dir: Path, steady_requests: int = 20, timeout: float = 60.0,
              server_args: tuple[str, ...] = ()) -> Trial:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    body = json.dumps({"inputs": [load_digits_split()[1][0].tolist()]}).encode()
    # The server's error output goes to a temporary file, not a pipe: nobody reads a pipe while
    # the server runs, so a talkative server could fill it and freeze mid-measurement. (The
    # pipe was also never closed; running the tests with warnings as errors caught that.)
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    started = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "inferbench.cli", "serve", "--model", model,
                             "--models-dir", str(models_dir), "--port", str(port), *server_args],
                            stdout=subprocess.DEVNULL, stderr=log)
    try:
        while True:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=1).close()
                break
            except OSError:
                if proc.poll() is not None:
                    log.seek(0)
                    raise ColdStartError(f"the server exited: {log.read().strip()[-500:]}") from None
                if time.perf_counter() - started > timeout:
                    raise ColdStartError(f"the server didn't answer within {timeout:g} s") from None
                time.sleep(0.005)
        startup_s = time.perf_counter() - started
        first_ms = _post(f"{base}/v1/predict", body, timeout)
        steady = [_post(f"{base}/v1/predict", body, timeout) for _ in range(steady_requests)]
    except (urllib.error.URLError, OSError) as exc:
        raise ColdStartError(f"a request to the new server failed: {exc}") from None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()
    return Trial(startup_s, first_ms, statistics.median(steady))


def measure(model: str, models_dir: Path, trials: int = 5, steady_requests: int = 20) -> dict:
    if model not in (*DIGITS_MODELS, "synthetic"):
        raise ColdStartError(f"unknown model {model!r}")
    if not 1 <= trials <= 50 or not 1 <= steady_requests <= 1000:
        raise ColdStartError("trials must be 1 to 50 and steady requests 1 to 1000")
    results = [one_trial(model, models_dir, steady_requests) for _ in range(trials)]
    return {
        "type": "coldstart", "model": model, "trials": trials, "steady_requests": steady_requests,
        "results": [asdict(t) for t in results],
        "median": {"startup_s": statistics.median(t.startup_s for t in results),
                   "first_ms": statistics.median(t.first_ms for t in results),
                   "steady_ms": statistics.median(t.steady_ms for t in results)},
        "machine": machine_info(),
    }
