"""The model server.

    POST /v1/predict   {"inputs": [[64 numbers from 0 to 16], ...]}   1 to 64 images
                       -> {"model", "predictions", "timing": {"queue_ms", "inference_ms", "batch_size"}}
    GET  /v1/model     what is being served, and whether it is a real model or synthetic
    GET  /health       {"status": "ok"}

The server reports its own timings with every answer, so a client can split a request's
latency into time on the wire, time waiting in the server, and time in the model.

It runs on this computer only: it listens on 127.0.0.1, rejects other Host headers (DNS
rebinding), grants no CORS, caps request bodies, and validates every input.
"""

from __future__ import annotations

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

from inferbench.models import INPUT_SIZE, PIXEL_MAX, Model

LOCAL_HOSTS = ["localhost", "127.0.0.1"]
MAX_IMAGES = 64
MAX_BODY_BYTES = 256 * 1024
MAX_ERRORS_SHOWN = 20
DEFAULT_THREADS = 4

Pixel = Annotated[float, Field(ge=0, le=PIXEL_MAX, allow_inf_nan=False)]
Image = Annotated[list[Pixel], Field(min_length=INPUT_SIZE, max_length=INPUT_SIZE)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    inputs: Annotated[list[Image], Field(min_length=1, max_length=MAX_IMAGES)]


def create_app(model: Model, *, threads: int = DEFAULT_THREADS, allowed_hosts: list[str] | None = None) -> FastAPI:
    """`threads` is how many requests can run the model at once; the rest wait their turn.

    FastAPI runs a plain `def` endpoint in a worker thread from anyio's pool, which allows 40
    at a time unless told otherwise. A benchmark has to state this number, so it is set here
    explicitly and reported by /v1/model, rather than left as a library default.
    """
    if not 1 <= threads <= 64:
        raise ValueError(f"threads must be between 1 and 64, got {threads}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = threads
        yield

    app = FastAPI(title="Inference Bench model server", version="0.1.0", lifespan=lifespan,
                  description=f"Serving {model.info.name} ({model.info.kind}).")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)

    @app.middleware("http")
    async def guard(request: Request, call_next):
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
        return await call_next(request)

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
        return model.info.to_json() | {"server": {"threads": threads, "max_images": MAX_IMAGES}}

    # A plain `def`: FastAPI runs it in a worker thread, so a slow model doesn't stall the server.
    @app.post("/v1/predict")
    def predict(body: PredictRequest, request: Request) -> dict:
        batch = np.asarray(body.inputs, dtype=np.float32)
        started = time.perf_counter()
        predictions = model.predict(batch)
        finished = time.perf_counter()
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
