"""Dynamic batching: group requests that arrive close together into one model call.

    request ─┐
    request ─┼─► queue ─► batch worker: take the first waiting request, then keep collecting
    request ─┘             until the batch holds `max_images` images or `max_wait_ms` has passed
                           since that first request; run the model once; give each request
                           back its own answers

The point is to spread the fixed cost of a model call over many inputs. The price is that a
request may wait up to `max_wait_ms` for others to join it. Whether that trade pays off depends
on the model and the load, which is what the batching experiment measures.

`workers` batch workers run at once, each taking batches from the same queue. Set to the same
number as the unbatched server's threads, both modes get the same amount of parallelism, so a
comparison between them measures batching and nothing else.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import anyio.to_thread
import numpy as np

from inferbench.models import Model


@dataclass(frozen=True)
class BatchSettings:
    max_images: int = 64
    max_wait_ms: float = 2.0
    workers: int = 1

    def check(self) -> None:
        if not 1 <= self.max_images <= 1024:
            raise ValueError(f"max_images must be between 1 and 1024, got {self.max_images}")
        if not 0 <= self.max_wait_ms <= 1000:
            raise ValueError(f"max_wait_ms must be between 0 and 1000, got {self.max_wait_ms}")
        if not 1 <= self.workers <= 64:
            raise ValueError(f"workers must be between 1 and 64, got {self.workers}")


@dataclass
class _Item:
    images: np.ndarray
    arrived: float
    done: asyncio.Future


@dataclass(frozen=True)
class BatchResult:
    predictions: list[int]
    queue_ms: float      # from this request reaching the batcher until its batch started
    inference_ms: float  # the whole batch's model call
    batch_images: int    # images in the batch this request was part of
    batch_requests: int  # requests in that batch


class DynamicBatcher:
    def __init__(self, model: Model, settings: BatchSettings = BatchSettings()):
        settings.check()
        self.model, self.settings = model, settings
        self.queue: asyncio.Queue[_Item] = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.batches = 0  # how many model calls have run, for tests and reports

    async def start(self) -> None:
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.settings.workers)]

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():  # nobody will serve these now
            item = self.queue.get_nowait()
            if not item.done.done():
                item.done.set_exception(RuntimeError("the server is shutting down"))

    async def predict(self, images: np.ndarray) -> BatchResult:
        if len(images) > self.settings.max_images:
            raise ValueError(f"a request of {len(images)} images can't fit a batch of {self.settings.max_images}")
        item = _Item(images, time.perf_counter(), asyncio.get_running_loop().create_future())
        await self.queue.put(item)
        return await item.done

    async def _worker(self) -> None:
        carried: _Item | None = None  # a request that didn't fit the previous batch
        while True:
            first = carried or await self.queue.get()
            carried = None
            batch, images = [first], len(first.images)
            deadline = first.arrived + self.settings.max_wait_ms / 1000
            while images < self.settings.max_images:
                remaining = deadline - time.perf_counter()
                try:
                    item = self.queue.get_nowait() if remaining <= 0 else await asyncio.wait_for(self.queue.get(), remaining)
                except (asyncio.QueueEmpty, TimeoutError):
                    break
                if images + len(item.images) > self.settings.max_images:
                    carried = item  # starts the next batch instead of overfilling this one
                    break
                batch.append(item)
                images += len(item.images)
            await self._run(batch)

    async def _run(self, batch: list[_Item]) -> None:
        started = time.perf_counter()
        try:
            stacked = np.concatenate([item.images for item in batch])
            predictions = await anyio.to_thread.run_sync(self.model.predict, stacked)
        except Exception as exc:  # a failing model fails every request in the batch, with the reason
            for item in batch:
                if not item.done.done():
                    item.done.set_exception(exc)
            return
        finished = time.perf_counter()
        self.batches += 1
        start = 0
        for item in batch:
            end = start + len(item.images)
            if not item.done.done():  # the client may have gone away
                item.done.set_result(BatchResult(
                    [int(p) for p in predictions[start:end]],
                    queue_ms=(started - item.arrived) * 1000, inference_ms=(finished - started) * 1000,
                    batch_images=len(stacked), batch_requests=len(batch)))
            start = end
