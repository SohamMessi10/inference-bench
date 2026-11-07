"""The dynamic batcher: what goes in a batch, what each request gets back, and failures."""

import asyncio
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from inferbench.batcher import BatchSettings, DynamicBatcher
from inferbench.models import Model, ModelInfo, load_model
from inferbench.server import create_app


class Recorder(Model):
    """Answers with each row's first pixel, so every answer shows which input it came from."""

    info = ModelInfo("recorder", "synthetic", 64, "test model")

    def __init__(self, cost_s=0.0, fail=False):
        self.calls, self.cost_s, self.fail = [], cost_s, fail

    def predict(self, batch):
        self.calls.append(len(batch))
        if self.cost_s:
            time.sleep(self.cost_s)
        if self.fail:
            raise RuntimeError("model exploded")
        return batch[:, 0].astype(np.int64)


def images(*values, n=1):
    return np.array([[v] + [0.0] * 63 for v in values for _ in range(n)], dtype=np.float32)


async def with_batcher(model, settings, work):
    # A time limit, so a lost request or a swallowed error fails the test in seconds instead
    # of hanging it (both once hung the mutation test until its own timeout).
    batcher = DynamicBatcher(model, settings)
    await batcher.start()
    try:
        return await asyncio.wait_for(work(batcher), timeout=10)
    finally:
        await batcher.stop()


def test_requests_arriving_together_share_one_model_call():
    model = Recorder()

    async def work(batcher):
        return await asyncio.gather(*(batcher.predict(images(v)) for v in range(10)))

    results = asyncio.run(with_batcher(model, BatchSettings(max_images=64, max_wait_ms=50), work))
    assert model.calls == [10]
    assert [r.predictions for r in results] == [[v] for v in range(10)]  # each gets its own answer
    assert all(r.batch_requests == 10 and r.batch_images == 10 for r in results)


def test_a_batch_never_exceeds_its_size_and_a_request_is_never_split():
    model = Recorder()

    async def work(batcher):
        return await asyncio.gather(*(batcher.predict(images(v, n=30)) for v in range(5)))

    results = asyncio.run(with_batcher(model, BatchSettings(max_images=64, max_wait_ms=50), work))
    assert model.calls == [60, 60, 30]
    assert [r.predictions for r in results] == [[v] * 30 for v in range(5)]


def test_a_lone_request_waits_for_company_but_no_longer_than_the_limit():
    async def work(batcher):
        return await batcher.predict(images(1))

    result = asyncio.run(with_batcher(Recorder(), BatchSettings(max_wait_ms=40), work))
    assert 38 <= result.queue_ms < 80 and result.batch_requests == 1
    quick = asyncio.run(with_batcher(Recorder(), BatchSettings(max_wait_ms=0), work))
    assert quick.queue_ms < 10


def test_several_workers_run_batches_at_the_same_time():
    model = Recorder(cost_s=0.1)

    async def work(batcher):
        started = time.perf_counter()
        await asyncio.gather(*(batcher.predict(images(v, n=64)) for v in range(4)))
        return time.perf_counter() - started

    elapsed = asyncio.run(with_batcher(model, BatchSettings(max_images=64, max_wait_ms=0, workers=4), work))
    assert model.calls == [64] * 4 and elapsed < 0.3  # four 100 ms calls side by side, not 400 ms in a row


def test_a_failing_model_fails_every_request_in_its_batch():
    async def work(batcher):
        return await asyncio.gather(*(batcher.predict(images(v)) for v in range(3)), return_exceptions=True)

    results = asyncio.run(with_batcher(Recorder(fail=True), BatchSettings(max_wait_ms=30), work))
    assert all(isinstance(r, RuntimeError) and str(r) == "model exploded" for r in results)


def test_requests_too_big_for_any_batch_are_refused():
    async def work(batcher):
        await batcher.predict(images(1, n=10))

    with pytest.raises(ValueError, match="can't fit a batch of 8"):
        asyncio.run(with_batcher(Recorder(), BatchSettings(max_images=8), work))


@pytest.mark.parametrize("bad", [dict(max_images=0), dict(max_images=1025), dict(max_wait_ms=-1),
                                 dict(max_wait_ms=1001), dict(workers=0)])
def test_settings_are_checked(bad):
    with pytest.raises(ValueError):
        BatchSettings(**bad).check()


@settings(deadline=None, max_examples=50)
@given(st.lists(st.integers(1, 64), min_size=1, max_size=40), st.integers(1, 128), st.integers(1, 4))
def test_every_request_gets_exactly_its_own_answers(sizes, max_images, workers):
    sizes = [min(s, max_images) for s in sizes]
    model = Recorder()

    async def work(batcher):
        return await asyncio.gather(*(batcher.predict(images(i, n=n)) for i, n in enumerate(sizes)))

    results = asyncio.run(with_batcher(model, BatchSettings(max_images, max_wait_ms=1, workers=workers), work))
    assert [r.predictions for r in results] == [[i] * n for i, n in enumerate(sizes)]
    assert sum(model.calls) == sum(sizes) and max(model.calls) <= max_images


def test_server_with_batching_answers_correctly(models_dir, digits):
    model = load_model("digits-small", models_dir)
    app = create_app(model, threads=2, batching=BatchSettings(max_images=64, max_wait_ms=5))
    with TestClient(app, base_url="http://localhost") as client:  # runs startup, so the batcher starts
        info = client.get("/v1/model").json()
        assert info["server"]["batching"] == {"max_images": 64, "max_wait_ms": 5, "workers": 2}
        assert app.state.batcher.settings.workers == 2  # as many batch workers as model threads
        sample = digits[1][:20]
        body = client.post("/v1/predict", json={"inputs": sample.tolist()}).json()
        assert body["predictions"] == model.predict(sample).tolist()
        assert body["timing"]["batch_requests"] == 1 and body["timing"]["batch_size"] == 20
        assert client.post("/v1/predict", json={"inputs": []}).status_code == 422


def test_server_reports_the_time_a_request_waited_for_its_batch(models_dir, digits):
    app = create_app(load_model("digits-small", models_dir), threads=1, batching=BatchSettings(max_wait_ms=40))
    with TestClient(app, base_url="http://localhost") as client:
        timing = client.post("/v1/predict", json={"inputs": digits[1][:1].tolist()}).json()["timing"]
    assert timing["queue_ms"] >= 38  # alone, it waited the full 40 ms for company
