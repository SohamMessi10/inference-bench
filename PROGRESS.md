# Progress

## Status

Milestone 1 of 6 complete: the models and the server. The load generator comes next.
The full plan is in [`docs/design.md`](docs/design.md).

## Milestones

### Milestone 1: models and the server
- Checked before building: Hugging Face and the PyTorch download site are unreachable from the
  build machine, so the real models are trained locally on scikit-learn's bundled digits
  dataset. numpy 2.5 needs Python 3.12+, so the project supports 3.12–3.14. `httpx2` (Pydantic's
  maintained continuation of httpx) is the HTTP client. Wheels exist for every Python version
  on Linux x86-64 and Apple Silicon.
- `models.py`: `digits-small` (256×256 hidden units, 85,002 parameters) and `digits-large`
  (1024×1024, 1,126,410), trained with scikit-learn on a fixed stratified 75/25 split and served
  as numpy matrix products. Measured: 98.00% and 97.78% accuracy on the 450 held-out images.
  `synthetic` waits a configurable fixed cost plus a cost per input and labels itself
  "NOT A MODEL".
- `server.py`: `POST /v1/predict` (1–64 images of 64 pixels, 0–16), `GET /v1/model`,
  `GET /health`. Each answer reports the server's own `queue_ms` and `inference_ms`. Strict
  validation, 127.0.0.1 only, Host check, cross-site refusal, a 256 KiB body cap.
- `--threads` sets how many requests run the model at once (default 4) and is reported by
  `/v1/model`. Without it, FastAPI's thread pool would quietly allow 40 (anyio's default), and a
  benchmark would be measuring a number nobody chose.
- `cli.py`: `ibench train [MODEL ...]` and `ibench serve`.
- Tests (54): accuracy on held-out images; numpy predictions and probabilities identical to
  scikit-learn's; a batch gives the same answers as one image at a time (Hypothesis); weights
  save and load; validation of every kind of bad input; the request guards; and a real uvicorn
  process under 8 concurrent clients (240 requests, every answer checked), plus a test that
  requests beyond the thread limit wait and `queue_ms` shows it.
- Bugs found: (1) a NaN or Infinity pixel made the server answer 500 instead of 422, because
  FastAPI's default error response copies the bad value back and JSON can't encode NaN. The
  error response now says where each problem is without echoing values, which also stops a bad
  64-image request echoing 4,096 numbers. (2) A wrong expected value in my own test (a
  parameter count I added up wrongly), corrected.
- Mutation-checked: 16 deliberate bugs. Three survived at first: a ReLU on the output layer
  (it rarely changes the top answer, so the scikit-learn comparison now checks probabilities);
  queue time never measured, and the thread limit not applied (both now caught by the real-server
  queueing test). One was equivalent: allowing NaN, because NaN and ±Infinity fail the 0–16
  range check anyway; the explicit check stays for clarity.

## Commands verified

| Where | Python | Command | Result |
|---|---|---|---|
| Claude's Linux workspace | 3.12, 3.13, 3.14 | `python -m pytest`, dev and ci profiles | 54 passed (milestone 1) |
| Claude's Linux workspace | 3.13 | `ibench train`, `ibench serve --model digits-large`, one real request | 3 of 3 digits right |

## Next

Milestone 2: the load generator. N concurrent workers send requests for a set time after a
warmup, and every request is recorded (when it was sent, how long it took, status, the server's
own timings), failures included, in a results file with the run's configuration.

## Concepts to review
- Why a benchmark must say whether its "model" is real.
- Training with one library and serving with another: checking they compute the same thing.
- Batch invariance: why batching must not change answers, and why float32 sums can differ slightly.
- Thread pools in FastAPI: plain `def` endpoints, anyio's default of 40, and why to set it.
- Why a test client isn't enough: testing against a real server process.
