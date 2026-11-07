"""The server as it really runs: a separate uvicorn process, with concurrent clients.

The test client runs the app in-process and can hide threading problems; in the Reliable Job
Runner, a bug that failed 1 request in 20-40 under real concurrent load never showed through it.
So the server is also tested here the way the benchmark will use it.
"""

import json
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from inferbench.models import load_model


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(*args):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "inferbench.cli", "serve", "--port", str(port), *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1).close()
            return base, proc
        except OSError:
            assert proc.poll() is None, proc.stderr.read()
            assert time.monotonic() < deadline, "server never started"
            time.sleep(0.05)


def stop(proc):
    proc.terminate()
    return proc.communicate(timeout=10)[1]


def predict(base, images):
    request = urllib.request.Request(f"{base}/v1/predict", data=json.dumps({"inputs": images}).encode(),
                                     method="POST", headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, None


@pytest.fixture
def digits_server(models_dir):
    base, proc = start("--model", "digits-small", "--models-dir", str(models_dir))
    yield base, proc
    if proc.poll() is None:
        stop(proc)


def test_concurrent_clients_get_correct_answers(digits_server, models_dir, digits):
    base, proc = digits_server
    model = load_model("digits-small", models_dir)
    images = digits[1]

    def client(seed):
        rng = random.Random(seed)
        outcomes = []
        for _ in range(30):
            picked = images[rng.sample(range(len(images)), rng.randint(1, 16))]
            status, body = predict(base, picked.tolist())
            right = body is not None and body["predictions"] == model.predict(picked).tolist()
            outcomes.append((status, right))
        return outcomes

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = Counter(o for batch in pool.map(client, range(8)) for o in batch)
    errors = stop(proc)
    assert results == Counter({(200, True): 240}), f"{dict(results)}; server log:\n{errors[-2000:]}"


def test_serving_a_model_that_was_never_trained_fails_clearly(tmp_path):
    run = subprocess.run([sys.executable, "-m", "inferbench.cli", "serve", "--model", "digits-large",
                          "--models-dir", str(tmp_path), "--port", str(free_port())],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 1
    assert "train it first with: ibench train digits-large" in run.stderr


def test_requests_beyond_the_thread_limit_wait_and_the_server_says_so():
    # 2 model threads, each call takes 200 ms, 6 requests at once: two run straight away, two
    # wait about 200 ms, and two about 400 ms. queue_ms must show that wait.
    base, proc = start("--model", "synthetic", "--fixed-ms", "200", "--per-item-ms", "0", "--threads", "2")
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            replies = list(pool.map(lambda _: predict(base, [[1.0] * 64]), range(6)))
        assert [status for status, _ in replies] == [200] * 6
        waits = sorted(body["timing"]["queue_ms"] for _, body in replies)
        assert waits[1] < 100 and 150 < waits[3] < 350 and 350 < waits[5] < 600, waits
        with urllib.request.urlopen(f"{base}/v1/model", timeout=5) as response:
            assert json.load(response)["server"]["threads"] == 2
    finally:
        stop(proc)


def test_synthetic_server_reports_the_cost_it_was_given():
    base, proc = start("--model", "synthetic", "--fixed-ms", "20", "--per-item-ms", "2")
    try:
        status, body = predict(base, [[1.0] * 64] * 10)
        assert status == 200
        assert 40 <= body["timing"]["inference_ms"] < 80  # 20 + 2 × 10 ms
        with urllib.request.urlopen(f"{base}/v1/model", timeout=5) as response:
            assert json.load(response)["kind"] == "synthetic"
    finally:
        stop(proc)


def test_batching_server_under_concurrent_clients(models_dir, digits):
    base, proc = start("--model", "digits-small", "--models-dir", str(models_dir), "--batch",
                       "--batch-wait-ms", "5", "--threads", "2")
    model = load_model("digits-small", models_dir)
    images = digits[1]

    def client(seed):
        rng = random.Random(seed)
        outcomes = []
        for _ in range(25):
            picked = images[rng.sample(range(len(images)), rng.randint(1, 8))]
            status, body = predict(base, picked.tolist())
            right = body is not None and body["predictions"] == model.predict(picked).tolist()
            outcomes.append((status, right, body["timing"]["batch_requests"] if body else 0))
        return outcomes

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [o for batch in pool.map(client, range(16)) for o in batch]
    errors = stop(proc)
    assert Counter((s, r) for s, r, _ in results) == Counter({(200, True): 400}), errors[-2000:]
    assert max(n for _, _, n in results) > 1  # requests really were combined
