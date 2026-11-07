"""The load generator against real servers: well-behaved, slow, failing, wrong, and absent."""

import asyncio
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from inferbench.cli import EXIT_MODEL, EXIT_OK, main
from inferbench.loadgen import (
    Inputs, LoadConfig, LoadError, expected_answers, read_results, run_load, summary, write_results,
)
from inferbench.models import SyntheticModel
from inferbench.server import create_app


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Served:
    """Any app in a real uvicorn server on a free port, in a background thread."""

    def __init__(self, app):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            assert time.monotonic() < deadline, "server never started"
            time.sleep(0.02)
        return self

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)

    def __exit__(self, *exc):
        self.stop()


def synthetic(fixed_ms=10, per_item_ms=0, threads=4):
    return create_app(SyntheticModel(fixed_ms, per_item_ms), threads=threads)


def load(**config):
    return asyncio.run(run_load(LoadConfig(**config)))


def test_a_healthy_run_records_every_phase_and_checks_every_answer():
    with Served(synthetic(fixed_ms=10, threads=4)) as served:
        run = load(url=served.url, concurrency=4, duration=1.0, warmup=0.3)
    phases = [r.phase for r in run.records]
    assert phases[0] == "first" and phases.count("first") == 1
    assert "warmup" in phases and "measure" in phases
    measured = run.measured()
    assert all(r.ok and r.correct == r.images == 1 for r in measured)
    s = summary(run)
    assert s["errors"] == {} and s["accuracy"] == 1.0
    assert 0.95 <= s["measured_seconds"] < 1.3
    # 4 model threads × one 10 ms call each at a time: at most 400 requests/s, less with overhead.
    assert 150 < s["ok_per_second"] <= 400
    assert all(r.inference_ms >= 10 and r.latency_ms >= r.inference_ms for r in measured)
    assert run.server["kind"] == "synthetic" and run.machine["cpus"]


def test_closed_loop_never_has_more_requests_in_flight_than_workers():
    with Served(synthetic(fixed_ms=5)) as served:
        run = load(url=served.url, concurrency=3, duration=0.6, warmup=0)
    by_worker = {}
    for r in run.records[1:]:
        by_worker.setdefault(r.worker, []).append(r)
    assert sorted(by_worker) == [0, 1, 2]
    for records in by_worker.values():  # each worker waits for its answer before sending again
        for a, b in zip(records, records[1:]):
            assert b.sent >= a.sent + a.latency_ms / 1000 - 1e-6
    events = sorted([(r.sent, 1) for r in run.records[1:]] + [(r.sent + r.latency_ms / 1000, -1) for r in run.records[1:]],
                    key=lambda e: (e[0], e[1]))
    in_flight = peak = 0
    for _, change in events:
        in_flight += change
        peak = max(peak, in_flight)
    assert peak <= 3


def test_timeouts_are_recorded_not_dropped():
    with Served(synthetic(fixed_ms=300)) as served:
        run = load(url=served.url, concurrency=2, duration=0.5, warmup=0, timeout=0.1)
    measured = run.measured()
    assert measured and all(r.error == "timeout" and r.status is None and r.correct is None for r in measured)
    assert all(95 <= r.latency_ms < 300 for r in measured)
    s = summary(run)
    assert s["ok"] == 0 and s["errors"] == {"timeout": len(measured)} and s["mean_latency_ms"] is None


def test_wrong_answers_and_bad_responses_are_caught():
    app = FastAPI()
    state = {"n": 0}

    @app.get("/v1/model")
    def info():
        return {"name": "broken", "kind": "real"}

    @app.post("/v1/predict")
    def predict(body: dict):
        state["n"] += 1
        timing = {"queue_ms": 0.0, "inference_ms": 0.0, "batch_size": len(body["inputs"])}
        match state["n"] % 4:
            case 0:
                return JSONResponse({"detail": "overloaded"}, status_code=503)
            case 1:
                return {"predictions": [-1] * len(body["inputs"]), "timing": timing}  # always wrong
            case 2:
                return {"predictions": [], "timing": timing}  # too few answers
            case _:
                return JSONResponse({"unexpected": True})

    with Served(app) as served:
        run = load(url=served.url, concurrency=1, duration=0.3, warmup=0, images=2)
    errors = summary(run)["errors"]
    assert set(errors) == {"http 503", "wrong number of answers", "unreadable answer"}
    wrong = [r for r in run.measured() if r.ok]
    assert wrong and all(r.correct == 0 for r in wrong)
    assert summary(run)["accuracy"] == 0.0


def test_a_server_that_dies_mid_run_shows_up_as_failures():
    served = Served(synthetic(fixed_ms=5)).__enter__()
    threading.Timer(0.5, served.stop).start()
    run = load(url=served.url, concurrency=2, duration=1.5, warmup=0, timeout=1.0)
    s = summary(run)
    assert s["ok"] > 0 and s["requests"] > s["ok"]
    assert set(s["errors"]) <= {"connect", "protocol", "timeout", "ReadError", "WriteError"}, s["errors"]


def test_no_server_is_a_clear_error():
    with pytest.raises(LoadError, match="no model server answered.*start one with: ibench serve"):
        load(url=f"http://127.0.0.1:{free_port()}", duration=0.2, warmup=0)


@pytest.mark.parametrize("changes,message", [
    ({"url": "http://example.com:8000"}, "url must be"),
    ({"url": "https://127.0.0.1:8000"}, "url must be"),
    ({"url": "http://127.0.0.1:8000/v1/predict"}, "url must be"),
    ({"url": "http://10.0.0.1:8000"}, "url must be"),
    ({"concurrency": 0}, "concurrency"),
    ({"concurrency": 257}, "concurrency"),
    ({"concurrency": True}, "concurrency"),
    ({"images": 65}, "images"),
    ({"duration": 0}, "duration"),
    ({"warmup": -1}, "warmup"),
    ({"timeout": 0}, "timeout"),
])
def test_bad_configurations_are_refused(changes, message):
    with pytest.raises(LoadError, match=message):
        LoadConfig(**changes).check()


def test_inputs_are_real_images_in_a_repeatable_order():
    a, b, c = Inputs(3, seed=1), Inputs(3, seed=1), Inputs(3, seed=2)
    first = [a.take()[1].tolist() for _ in range(200)]
    assert first == [b.take()[1].tolist() for _ in range(200)]
    assert first != [c.take()[1].tolist() for _ in range(200)]
    assert len(Inputs(1, 0).images) == 450  # the held-out images: the server never trained on them


def test_expected_answers():
    images = np.array([[1.0] * 64, [2.0] * 64])
    assert expected_answers({"kind": "synthetic"}, images, np.array([7, 8])) == [4, 8]  # 64 % 10, 128 % 10
    assert expected_answers({"kind": "real"}, images, np.array([7, 8])) == [7, 8]


def test_results_file_round_trip(tmp_path):
    with Served(synthetic(fixed_ms=2)) as served:
        run = load(url=served.url, concurrency=2, duration=0.3, warmup=0.1)
    path = tmp_path / "out" / "run.jsonl"
    write_results(run, path)
    lines = path.read_text().splitlines()
    assert len(lines) == len(run.records) + 2
    again = read_results(path)
    assert again.config == run.config and again.server == run.server and again.records == run.records
    assert summary(again) == summary(run)
    assert '"method": "closed loop' in lines[0]
    assert not list(path.parent.glob("*.tmp"))


def test_read_results_refuses_other_files(tmp_path):
    other = tmp_path / "x.jsonl"
    other.write_text('{"hello": 1}\n')
    with pytest.raises(LoadError, match="isn't an ibench results file"):
        read_results(other)


def test_load_command(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with Served(synthetic(fixed_ms=2)) as served:
        assert main(["load", "--url", served.url, "--concurrency", "2", "--duration", "0.3", "--warmup", "0"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "synthetic (synthetic):" in out and "requests/s" in out and "correct" not in out
    files = list((tmp_path / "results").glob("synthetic-c2-*.jsonl"))
    assert len(files) == 1 and f"every request is in {files[0].relative_to(tmp_path)}" in out


def test_load_command_fails_when_every_request_fails(capsys, tmp_path):
    with Served(synthetic(fixed_ms=300)) as served:
        code = main(["load", "--url", served.url, "--duration", "0.3", "--warmup", "0", "--timeout", "0.05",
                     "-o", str(tmp_path / "r.jsonl")])
    assert code == EXIT_MODEL and "0 ok" in capsys.readouterr().out


def test_load_command_refuses_a_remote_url(capsys):
    assert main(["load", "--url", "http://example.com:8000"]) == EXIT_MODEL
    assert "url must be" in capsys.readouterr().err


def test_summary_numbers_can_be_recomputed_from_the_records():
    # Slow requests, so the last answers arrive well after the deadline, and a warmup that
    # must not be counted. Both once survived a mutation test with fast requests and no warmup.
    with Served(synthetic(fixed_ms=200, threads=2)) as served:
        run = load(url=served.url, concurrency=2, duration=0.5, warmup=0.3)
    measured = run.measured()
    assert any(r.phase == "warmup" for r in run.records)
    first_sent = min(r.sent for r in measured)
    last_answer = max(r.sent + r.latency_ms / 1000 for r in measured)
    assert run.measured_seconds == pytest.approx(last_answer - first_sent, abs=1e-6)
    # The last answers arrive about one 200 ms call after the last send, and that time counts.
    assert last_answer - max(r.sent for r in measured) >= 0.15
    s = summary(run)
    assert s["requests"] == len(measured)
    assert s["ok_per_second"] == round(sum(r.ok for r in measured) / run.measured_seconds, 1)
    assert s["images_per_second"] == round(sum(r.images for r in measured if r.ok) / run.measured_seconds, 1)
