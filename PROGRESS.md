# Progress

## Status

Milestone 2 of 6 complete: the load generator. Metrics and the report come next.
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

### Milestone 2: the load generator
- `loadgen.py`: closed-loop load with `httpx2`'s async client: N workers, each sending a request
  and waiting for its answer. Phases: one `first` request on its own, `warmup` (recorded, not
  counted), then `measure`. Inputs are the 450 held-out digit images in a seeded order, and every
  answer is checked: against the true digit for a real model, against the documented rule for
  the synthetic one.
- One record per request, failures included: send time, latency, HTTP status, an error name
  (`timeout`, `connect`, `protocol`, `http 503`, `wrong number of answers`, `unreadable answer`),
  images, how many answers were right, and the server's own queue and inference times.
- Results files (JSON Lines): a header with the configuration, the method ("closed loop"), what
  `/v1/model` reported, and the machine (OS, CPU, cores, Python and library versions); one line
  per request; a summary. Written atomically; `read_results` reads them back exactly.
- `ibench load` (local URLs only: this measures your own machine, it isn't a tool for sending
  load elsewhere). Exit code 1 if every request failed.
- First real measurement, digits-large with 4 model threads, 4 s per run on the 2-CPU
  workspace: 1 worker 363 requests/s at 2.7 ms mean; 4 workers 598/s at 6.6 ms; 16 workers 609/s
  at 26.1 ms. Throughput stops growing at the thread count while latency keeps rising: the extra
  requests just queue. Answers stayed 97.7–97.8% correct, matching held-out accuracy.
- Tests (25 new) against real uvicorn servers: a healthy run, closed-loop behaviour (no worker
  sends before its answer arrives; never more requests in flight than workers), timeouts
  recorded, a deliberately broken server (503s, wrong answers, short answers, garbage), a server
  that dies mid-run, no server at all, config validation, repeatable inputs, the file round trip.
- Bugs found: (1) the summary saved measured time rounded to milliseconds, so throughput
  recomputed from a results file came out 0.1/s different; the exact value is now stored.
  (2) The measured window began from a timestamp about 67 µs before the first record's own send
  time; it is now computed from the records, so the two always agree.
- Mutation-checked: 11 deliberate bugs. Two survived at first (measured time stopping at the
  deadline, and throughput counting warmup requests); a test with slow requests and a warmup that
  recomputes both from the raw records now catches them. A wrong assumption in that new test (where the
  deadline falls) was corrected.

## Commands verified

| Where | Python | Command | Result |
|---|---|---|---|
| Claude's Linux workspace | 3.12, 3.13, 3.14 | `python -m pytest`, dev and ci profiles | 54 passed (milestone 1) |
| Claude's Linux workspace | 3.13 | `ibench train`, `ibench serve --model digits-large`, one real request | 3 of 3 digits right |
| Claude's Linux workspace | 3.12, 3.13, 3.14 | `python -m pytest`, dev and ci profiles | 79 passed (milestone 2) |
| Claude's Linux workspace | 3.13 | `ibench load` at 1, 4 and 16 workers against digits-large | all requests ok |

## Next

Milestone 3: metrics and the report. Percentiles (p50/p95/p99) with sample-size rules (p95
needs 200 successes, p99 1,000), latency split into waiting and model time, cold start after a
fresh server start, and a report comparing runs with the hardware and settings of each.

## Concepts to review
- Why a benchmark must say whether its "model" is real.
- Training with one library and serving with another: checking they compute the same thing.
- Batch invariance: why batching must not change answers, and why float32 sums can differ slightly.
- Thread pools in FastAPI: plain `def` endpoints, anyio's default of 40, and why to set it.
- Why a test client isn't enough: testing against a real server process.
- Closed-loop load and coordinated omission: why it understates latency under overload.
- Why failures must stay in the results, and why answers are checked under load.
- Saturation: throughput flattens at the server's capacity while latency keeps growing.
