# Inference Bench: design

## Problem

Teams serving machine-learning models have to answer practical questions: how many requests a
second the server can take, how slow the slowest requests get as load rises, and whether
grouping requests into batches helps or hurts. The answers are easy to get wrong. A benchmark
that waits for each response before sending the next one hides queueing delay. A p99 measured
from 50 requests means nothing. A "real model" that is really a sleep call says nothing about
real inference. This project is a small, honest benchmark tool that gets those details right
and shows its working.

**User:** developers building AI applications who want to understand serving behaviour on their
own machine before reaching for heavier tools.

**Value:** a reproducible experiment, run on a laptop, that measures a real tradeoff (latency
versus throughput under batching) and reports it with the sample counts, warmup, hardware and
failures needed to trust it.

## Core features (MVP)

1. A model server (FastAPI) with two kinds of model, always labelled:
   - **Real:** a small neural network that classifies handwritten digits, trained locally on
     the digits dataset that ships with scikit-learn (1,797 8×8 images; no download). Two sizes:
     `digits-small` (256×256 hidden units) and `digits-large` (1024×1024). Accuracy is measured
     on a held-out quarter of the data and reported with the model.
   - **Synthetic:** a stand-in with a configurable cost (a fixed cost per call plus a cost per
     item), for experiments where inference time must be controlled exactly. It is never
     presented as a real model.
2. A load generator that sends bounded concurrent requests and records every request's timing
   and outcome, failures included.
3. Metrics: throughput, p50/p95/p99 latency, error counts, and the server's own queue and
   inference time per request. Cold start and warmup are measured separately from steady state.
4. A report comparing runs, recording sample count, warmup, duration, hardware, concurrency,
   and request size. Percentiles that the sample size can't support are marked, not printed.
5. A batching experiment: the same model served with and without dynamic batching, across
   concurrency levels, with the measured tradeoff stated even if batching doesn't help.

**Optional, later:** an open-loop (fixed arrival rate) mode; cancellation and backpressure;
comparing model versions; streaming responses; running in containers.

## User flow

```
ibench train                          train both digits models, report test accuracy
ibench serve --model digits-large     start the server on 127.0.0.1:8000
ibench load --concurrency 1,4,16 ...  send load, write one results file per run
ibench report results/*.jsonl         a table and an HTML report comparing the runs
```

`ibench experiment batching` runs the whole batching experiment (start servers, warm up, load,
report) as one reproducible command.

## Architecture

```
 load generator (httpx2, asyncio)  ──HTTP──►  model server (FastAPI + uvicorn)
   N concurrent workers                          /v1/predict
   one record per request                         │  validate → (batcher) → model in a worker thread
   results/<run>.jsonl                            ▼
          │                                     Model: digits-small | digits-large | synthetic
          ▼
 report: percentiles, throughput, errors, hardware → table + HTML
```

There is no database: each run is a JSON Lines file of per-request records plus a header with
the run's configuration and hardware, which is easy to keep, diff, and reprocess.

## Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.12–3.14 | numpy 2.5 requires 3.12 |
| Model training | scikit-learn 1.9 `MLPClassifier` | a well-tested trainer; no download needed |
| Inference | numpy 2.5 | the trained weights run as plain matrix products, so the server doesn't need scikit-learn and the cost of each batch size is clear |
| Server | FastAPI 0.162, uvicorn 0.54 | standard Python serving stack |
| Load client | httpx2 2.16 (async) | the maintained continuation of httpx |
| Tests | pytest, Hypothesis | as in the other projects |

## Model

- **Data:** scikit-learn's digits, 1,797 images of 64 pixels with values 0–16. Split 75/25,
  stratified, with a fixed seed; pixels scaled to 0–1.
- **Training:** `MLPClassifier`, ReLU, Adam, fixed `random_state`. Measured in a first check:
  98.0% test accuracy for 256×256 (2.7 s to train), 97.8% for 1024×1024 (19.5 s).
- **Serving:** the weights are saved to `models/<name>.npz` with metadata (accuracy, split, seed,
  versions) and run with numpy, as float32. A test checks the numpy predictions equal
  scikit-learn's on the whole test set.
- **Why batching might matter:** measured on this machine, `digits-large` costs 216 µs for one
  image and 30 µs per image in a batch of 64. Per-request HTTP overhead is likely to be larger
  than that, so whether server-side batching helps is a genuine question for the experiment.

## API

| Request | Response |
|---|---|
| `POST /v1/predict` `{"inputs": [[64 numbers 0–16], ...]}` (1–64 images) | `{"model", "predictions": [digit, ...], "timing": {"queue_ms", "inference_ms", "batch_size"}}` |
| `GET /v1/model` | name, kind (`real` or `synthetic`), input shape, test accuracy, parameters |
| `GET /health` | `{"status": "ok"}` |

Errors: 422 invalid input (wrong length, out of range, not finite, too many images); 413 body
too large; 503 when the server's queue is full (later milestones). The server reports its own
timings so the client can split latency into network, queueing, and inference.

## Security

Runs on one machine: listens on 127.0.0.1 only, rejects other Host headers (DNS rebinding),
grants no CORS, caps request size, and validates every input strictly. No secrets or accounts.

## Measurement methodology (the important part)

- **Closed loop first:** N workers each send a request, wait for the answer, and send the next.
  It's simple but understates latency under overload, because a slow server slows the senders
  down too ("coordinated omission"). That is stated in every report, and an open-loop mode with
  a fixed arrival rate is an optional extension.
- **Warmup is excluded** from results and recorded; **cold start** (the first request after the
  server starts) is measured separately.
- **Every request is kept,** including failures and timeouts. Throughput counts successes, and
  the error rate is always shown next to it.
- **Percentiles need samples:** a percentile is shown only when at least 10 measurements lie
  beyond it: 20 successful requests for p50, 200 for p95, 1,000 for p99. Otherwise the report
  shows `n<200` (and so on) instead of a number. Percentiles are nearest-rank, so each one is a
  latency some request really had.
- **Hardware and software** are recorded with every run. Client and server share the machine,
  which is a limitation the report states.

## Testing strategy

Unit tests for the model (accuracy threshold, numpy versus scikit-learn), input validation, and
percentile maths (checked against a brute-force definition with Hypothesis); API tests through
the test client; and, from the job runner's lesson, tests against a real server under
concurrent load, because a test client can hide threading bugs.

## Deployment

A local tool, installed with pip; CI runs the tests on Linux and macOS. Results and trained
weights stay out of Git (`models/`, `results/`), apart from small example results kept for the
README.

## Milestones

1. Models and the server: training, the digits models and the synthetic model, `/v1/predict`,
   validation, tests including a real server. *(complete)*
2. Load generator: concurrent workers, per-request records, warmup, results files. *(complete)*
3. Metrics and report: percentiles with sample-size rules, cold start, hardware, comparison. *(complete)*
4. Batching experiment: a dynamic batcher in the server and the experiment across concurrency. *(complete:
   results in [`batching.md`](batching.md))*
5. Overload and failure: a bounded queue (503), client timeouts, injected server errors. *(complete:
   results in [`overload.md`](overload.md))*
6. Methodology write-up, README, resume material.
