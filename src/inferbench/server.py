"""The model server.

    POST /v1/predict   {"inputs": [[64 numbers from 0 to 16], ...]}   1 to 64 images
                       -> {"model", "predictions", "timing": {"queue_ms", "inference_ms", "batch_size"}}
    GET  /v1/model     what is being served, whether it is real or synthetic, and the server's settings
    GET  /v1/stats     what the server has done: answered, refused as busy or late, failed
    GET  /health       {"status": "ok"}

The server reports its own timings with every answer, so a client can split a request's
latency into time on the wire, time waiting in the server, and time in the model.

**Under overload** it can protect itself in two ways, both off by default:

- `max_waiting`: at most this many requests wait for the model at once. Beyond that, a request
  is refused at once with 503 `busy`, instead of joining a queue that only grows.
- `deadline_ms`: a request that has already waited this long when the model is free is
  refused with 503 `deadline` instead of being run, because its client has probably given up.

A failing model gives 500 `model`, with no internal details in the response.

It runs on this computer only: it listens on 127.0.0.1, rejects other Host headers (DNS
rebinding), grants no CORS, caps request bodies, and validates every input.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

import anyio.to_thread
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from inferbench.batcher import BatchSettings, DeadlineExceeded, DynamicBatcher
from inferbench.models import INPUT_SIZE, PIXEL_MAX, Model

log = logging.getLogger("inferbench.server")

LOCAL_HOSTS = ["localhost", "127.0.0.1"]
MAX_IMAGES = 64
MAX_BODY_BYTES = 256 * 1024
MAX_ERRORS_SHOWN = 20
DEFAULT_THREADS = 4
PREDICT_PATH = "/v1/predict"

Pixel = Annotated[float, Field(ge=0, le=PIXEL_MAX, allow_inf_nan=False)]
Image = Annotated[list[Pixel], Field(min_length=INPUT_SIZE, max_length=INPUT_SIZE)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    inputs: Annotated[list[Image], Field(min_length=1, max_length=MAX_IMAGES)]


class Stats:
    """Counters, updated from the event loop and from model threads, so behind a lock."""

    FIELDS = ("answered", "refused_busy", "refused_deadline", "model_errors", "peak_in_progress")

    def __init__(self):
        self.lock = threading.Lock()
        self.values = dict.fromkeys(self.FIELDS, 0)

    def add(self, field: str, n: int = 1) -> None:
        with self.lock:
            self.values[field] += n

    def peak(self, value: int) -> None:
        with self.lock:
            self.values["peak_in_progress"] = max(self.values["peak_in_progress"], value)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.values)


def refuse(reason: str, detail: str, status: int = 503) -> JSONResponse:
    headers = {"Retry-After": "1"} if status == 503 else {}
    return JSONResponse({"detail": detail, "reason": reason}, status_code=status, headers=headers)


def create_app(model: Model, *, threads: int = DEFAULT_THREADS, batching: BatchSettings | None = None,
               max_waiting: int | None = None, deadline_ms: float | None = None,
               allowed_hosts: list[str] | None = None) -> FastAPI:
    """`threads` is how many model calls can run at once; the rest wait their turn.

    FastAPI runs a plain `def` endpoint in a worker thread from anyio's pool, which allows 40
    at a time unless told otherwise. A benchmark has to state this number, so it is set here
    explicitly and reported by /v1/model, rather than left as a library default.

    With `batching`, requests are grouped by a DynamicBatcher with `threads` batch workers, so
    both modes allow the same number of model calls at once.
    """
    if not 1 <= threads <= 64:
        raise ValueError(f"threads must be between 1 and 64, got {threads}")
    if max_waiting is not None and not 0 <= max_waiting <= 100_000:
        raise ValueError(f"max_waiting must be between 0 and 100000, got {max_waiting}")
    if deadline_ms is not None and not 0 < deadline_ms <= 600_000:
        raise ValueError(f"deadline_ms must be between 0 and 600000, got {deadline_ms}")
    batcher = None
    if batching is not None:
        batching = BatchSettings(batching.max_images, batching.max_wait_ms, workers=threads)
        batcher = DynamicBatcher(model, batching)
    stats = Stats()
    # Requests admitted to /v1/predict and not yet answered: running, or waiting to run.
    # Only touched on the event loop, so it needs no lock.
    in_progress = 0
    capacity = None if max_waiting is None else threads + max_waiting

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = threads
        if batcher:
            await batcher.start()
        yield
        if batcher:
            await batcher.stop()

    app = FastAPI(title="Inference Bench model server", version="0.1.0", lifespan=lifespan,
                  description=f"Serving {model.info.name} ({model.info.kind}).")
    app.state.stats = stats
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        nonlocal in_progress
        request.state.received = time.perf_counter()
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if origin is not None and origin not in (f"http://{host}", f"https://{host}"):
                return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
            length = request.headers.get("content-length")
            if length is None or not length.isdigit():
                return JSONResponse({"detail": "a Content-Length header is required"}, status_code=411)
            if int(length) > MAX_BODY_BYTES:
                return JSONResponse({"detail": f"request body is larger than {MAX_BODY_BYTES} bytes"},
                                    status_code=413)
        if request.url.path != PREDICT_PATH or request.method != "POST":
            return await call_next(request)
        # Admission happens here, on the event loop, before the request can queue for a thread.
        if capacity is not None and in_progress >= capacity:
            stats.add("refused_busy")
            return refuse("busy", f"the server is busy: {in_progress} requests are already in progress")
        in_progress += 1
        stats.peak(in_progress)
        try:
            return await call_next(request)
        finally:
            in_progress -= 1

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 copies each bad value back into the response. A NaN pixel then
        # crashes the JSON encoder and the client gets a 500, and a bad 64-image request would
        # echo all 4,096 numbers. Say where each problem is and what it is, without the values.
        errors = [{"loc": [str(part) for part in error.get("loc", ())], "msg": error.get("msg", ""),
                   "type": error.get("type", "")} for error in exc.errors()]
        return JSONResponse({"detail": errors[:MAX_ERRORS_SHOWN], "errors": len(errors)}, status_code=422)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/model")
    def model_info() -> dict:
        server = {"threads": threads, "max_images": MAX_IMAGES, "batching": None,
                  "max_waiting": max_waiting, "deadline_ms": deadline_ms}
        if batching:
            server["batching"] = {"max_images": batching.max_images, "max_wait_ms": batching.max_wait_ms,
                                  "workers": batching.workers}
        return model.info.to_json() | {"server": server}

    @app.get("/v1/stats")
    def server_stats() -> dict:
        return stats.snapshot()

    def model_failed() -> JSONResponse:
        # Logged in full on the server; the client learns only that the model failed.
        log.exception("the model failed")
        stats.add("model_errors")
        return refuse("model", "the model failed on this request", status=500)

    def too_late() -> JSONResponse:
        stats.add("refused_deadline")
        return refuse("deadline", f"the request waited more than {deadline_ms:g} ms, so it was not run")

    if batcher:
        app.state.batcher = batcher

        # async: the request waits for its batch without holding a worker thread.
        @app.post(PREDICT_PATH)
        async def predict_batched(body: PredictRequest, request: Request):
            images = np.asarray(body.inputs, dtype=np.float32)
            try:
                result = await batcher.predict(images, arrived=request.state.received,
                                               deadline_s=None if deadline_ms is None else deadline_ms / 1000)
            except DeadlineExceeded:
                return too_late()
            except Exception:
                return model_failed()
            stats.add("answered")
            return {
                "model": model.info.name,
                "predictions": result.predictions,
                "timing": {
                    # Reading and validating the body, then waiting for others to join the batch
                    # and for a batch worker to be free. (The batcher measures from arrival.)
                    "queue_ms": round(result.queue_ms, 3),
                    "inference_ms": round(result.inference_ms, 3),
                    "batch_size": result.batch_images,
                    "batch_requests": result.batch_requests,
                },
            }

        return app

    # A plain `def`: FastAPI runs it in a worker thread, so a slow model doesn't stall the server.
    @app.post(PREDICT_PATH)
    def predict(body: PredictRequest, request: Request):
        batch = np.asarray(body.inputs, dtype=np.float32)
        started = time.perf_counter()
        if deadline_ms is not None and (started - request.state.received) * 1000 > deadline_ms:
            return too_late()
        try:
            predictions = model.predict(batch)
        except Exception:
            return model_failed()
        finished = time.perf_counter()
        stats.add("answered")
        return {
            "model": model.info.name,
            "predictions": [int(p) for p in predictions],
            "timing": {
                # From the request reaching the server to the model starting: reading and
                # validating the body, then waiting for a free worker thread.
                "queue_ms": round((started - request.state.received) * 1000, 3),
                "inference_ms": round((finished - started) * 1000, 3),
                "batch_size": len(batch),
            },
        }

    return app
