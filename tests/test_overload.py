"""Overload and failure: refusing work the server can't do in time, and failing cleanly."""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inferbench.batcher import BatchSettings
from inferbench.models import FlakyModel, InjectedFailure, ModelError, SyntheticModel, load_model
from inferbench.server import create_app
from tests.test_loadgen import Served

ONE_IMAGE = json.dumps({"inputs": [[1.0] * 64]}).encode()


def post(url, timeout=10):
    request = urllib.request.Request(f"{url}/v1/predict", data=ONE_IMAGE, method="POST",
                                     headers={"content-type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), (time.perf_counter() - started) * 1000, response.headers
    except urllib.error.HTTPError as exc:
        with exc:  # an HTTPError holds the open response; Python 3.14 warns if it isn't closed
            return exc.code, json.load(exc), (time.perf_counter() - started) * 1000, exc.headers


def stats(url):
    with urllib.request.urlopen(f"{url}/v1/stats", timeout=5) as response:
        return json.load(response)


def at_once(url, n):
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda _: post(url), range(n)))


def test_without_limits_every_request_waits_its_turn():
    with Served(create_app(SyntheticModel(100, 0), threads=1)) as served:
        replies = at_once(served.url, 4)
        assert [status for status, *_ in replies] == [200] * 4
        assert max(ms for *_, ms, _ in replies) > 350  # the last waited for three 100 ms calls
        assert stats(served.url)["answered"] == 4


def test_a_full_server_refuses_at_once_instead_of_queueing():
    # 1 model thread and room for 1 waiting: of 6 requests at once, 2 are admitted.
    with Served(create_app(SyntheticModel(150, 0), threads=1, max_waiting=1)) as served:
        replies = at_once(served.url, 6)
        s = stats(served.url)
    ok = [r for r in replies if r[0] == 200]
    busy = [r for r in replies if r[0] == 503]
    assert len(ok) == 2 and len(busy) == 4
    assert all(body["reason"] == "busy" and headers.get("Retry-After") == "1" for _, body, _, headers in busy)
    assert all(ms < 100 for _, _, ms, _ in busy)  # refused immediately, not after waiting
    assert s["answered"] == 2 and s["refused_busy"] == 4 and s["peak_in_progress"] == 2


def test_max_waiting_zero_admits_only_what_can_run_now():
    with Served(create_app(SyntheticModel(150, 0), threads=2, max_waiting=0)) as served:
        statuses = sorted(r[0] for r in at_once(served.url, 5))
        # Once the burst is over the server has room again: finished requests free their place.
        again = sorted(r[0] for r in at_once(served.url, 2))
    assert statuses == [200, 200, 503, 503, 503]
    assert again == [200, 200]


def test_requests_that_waited_past_the_deadline_are_not_run():
    # 1 thread, 100 ms per call, 40 ms deadline: the first runs; the rest wait about 100 ms
    # or more for the thread, so by then they're past their deadline and are dropped unrun.
    with Served(create_app(SyntheticModel(100, 0), threads=1, deadline_ms=40)) as served:
        replies = at_once(served.url, 4)
        s = stats(served.url)
    statuses = sorted(r[0] for r in replies)
    assert statuses[0] == 200 and statuses.count(503) >= 2
    assert all(body["reason"] == "deadline" for status, body, *_ in replies if status == 503)
    assert s["answered"] + s["refused_deadline"] == 4 and s["refused_deadline"] >= 2


@pytest.mark.parametrize("max_images", [1, 2])
def test_deadlines_apply_to_batched_requests_too(max_images):
    # A batch takes 200 ms; batched requests that can't start within 50 ms are dropped. With
    # batches of 1, every late request is the first of its batch; with 2, some join a batch.
    # (Testing only batches of 2 once let a missing check on the first request go unnoticed.)
    app = create_app(SyntheticModel(200, 0), threads=1, batching=BatchSettings(max_images=max_images, max_wait_ms=1),
                     deadline_ms=50)
    with Served(app) as served:
        replies = at_once(served.url, 6)
        s = stats(served.url)
    assert any(r[0] == 200 for r in replies)
    late = [r for r in replies if r[0] == 503]
    assert late and all(body["reason"] == "deadline" for _, body, *_ in late)
    assert s["refused_deadline"] == len(late) and s["answered"] == 6 - len(late)


def test_a_failing_model_gives_500_without_internal_details():
    app = create_app(FlakyModel(SyntheticModel(0, 0), fail_rate=1.0))
    client = TestClient(app, base_url="http://localhost")
    response = client.post("/v1/predict", json={"inputs": [[1.0] * 64]})
    assert response.status_code == 500
    assert response.json() == {"detail": "the model failed on this request", "reason": "model"}
    assert "Traceback" not in response.text and "injected" not in response.text
    assert client.get("/v1/stats").json()["model_errors"] == 1
    assert "failing 100% of calls on purpose" in client.get("/v1/model").json()["description"]


def test_a_failing_model_behind_a_batcher_fails_the_whole_batch_cleanly():
    app = create_app(FlakyModel(SyntheticModel(0, 0), fail_rate=1.0), batching=BatchSettings(max_wait_ms=1))
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post("/v1/predict", json={"inputs": [[1.0] * 64]})
    assert response.status_code == 500 and response.json()["reason"] == "model"


def test_injected_failures_are_repeatable_and_near_the_rate():
    def failures(seed):
        model = FlakyModel(SyntheticModel(0, 0), fail_rate=0.3, seed=seed)
        pattern = []
        for _ in range(1000):
            try:
                model.predict(np.ones((1, 64)))
                pattern.append(False)
            except InjectedFailure:
                pattern.append(True)
        return pattern

    a = failures(1)
    assert a == failures(1) and a != failures(2)
    assert 250 < sum(a) < 350  # 30% of 1,000, give or take; the seed makes it exact each run
    with pytest.raises(ModelError):
        FlakyModel(SyntheticModel(0, 0), fail_rate=1.5)


def test_the_load_generator_records_why_a_request_was_refused():
    import asyncio

    from inferbench.loadgen import LoadConfig, run_load, summary

    with Served(create_app(SyntheticModel(50, 0), threads=1, max_waiting=0)) as served:
        run = asyncio.run(run_load(LoadConfig(url=served.url, concurrency=4, duration=0.5, warmup=0)))
    errors = summary(run)["errors"]
    assert set(errors) == {"http 503 busy"} and summary(run)["ok"] > 0


@pytest.mark.parametrize("kwargs", [{"max_waiting": -1}, {"deadline_ms": 0}, {"deadline_ms": 700_000}])
def test_limits_are_checked(kwargs):
    with pytest.raises(ValueError):
        create_app(SyntheticModel(0, 0), **kwargs)


def test_settings_are_reported(models_dir):
    app = create_app(load_model("digits-small", models_dir), max_waiting=8, deadline_ms=250)
    info = TestClient(app, base_url="http://localhost").get("/v1/model").json()
    assert (info["server"]["max_waiting"], info["server"]["deadline_ms"]) == (8, 250)
