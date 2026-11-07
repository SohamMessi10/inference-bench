# Inference Bench

[![tests](https://github.com/SohamMessi10/inference-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/SohamMessi10/inference-bench/actions/workflows/tests.yml)

A benchmark for model servers that shows its working. It serves a small neural network over
HTTP, sends it controlled load, and measures throughput, latency percentiles, and failures. It
also runs two experiments every AI platform team eventually faces: **does batching help**, and
**what happens when the server gets more work than it can do**.

It runs on a laptop, needs no GPU and no downloads, and every number it reports comes with the
sample count, settings, and machine needed to judge it.

## What it found

Measured on a 2-CPU Linux machine, client and server sharing it; the write-ups give every run.

**Batching** ([`docs/batching.md`](docs/batching.md)): grouping requests that arrive within 2 ms
into one model call.

| Concurrent clients | Throughput change from batching (6 runs) |
|---:|---:|
| 1 | −39% to −47% (each request waits for company that never comes) |
| 4 | −3% to −8% |
| 16 | +11% to +51% |
| 64 | +36% to +105% (median latency roughly halved) |

**Overload** ([`docs/overload.md`](docs/overload.md)): 16 clients that give up after 100 ms, against
a server that can answer about 100 requests a second.

| Server | Answered in time (3 runs) | Work that reached nobody |
|---|---:|---:|
| no limits | 0 per second | 99% |
| bounded queue (4 waiting) | 91–92 per second | 0% |

Without limits the server collapsed: busy the whole time, it didn't answer a single request in
time. A bounded queue, refusing what it can't serve in time, kept it near capacity.

**And one result withdrawn.** An early overload run claimed 4–17× more useful throughput with
protection. Successful requests had a 95 ms median under a 50 ms limit, which shouldn't be
possible: the load generator itself was overloaded, running 34 ms late. The tool now measures
its own lag in every run and warns when the client, not the server, is the bottleneck. The
story is in [`docs/overload.md`](docs/overload.md#a-mistake-this-experiment-caught).

## Try it

Requires Python 3.12 or newer.

```bash
git clone https://github.com/SohamMessi10/inference-bench.git
cd inference-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ibench train
ibench serve --model digits-large
```

`ibench train` trains two small neural networks that read 8×8 handwritten digits, using the
dataset that ships with scikit-learn, and reports their accuracy on 450 images they never saw.
Then, in a second terminal (with the same `.venv` activated):

```bash
ibench load --concurrency 1 -o results/c1.jsonl
ibench load --concurrency 4 -o results/c4.jsonl
ibench load --concurrency 16 -o results/c16.jsonl
ibench report results/c1.jsonl results/c4.jsonl results/c16.jsonl
```

This is a real report from the build machine (columns trimmed to fit):

```
run  workers  ok    failed  req/s  p50 ms  p95 ms  p99 ms  wait ms  model ms  other ms  client lag p99
c1         1  3634       0  363.4    2.57    3.38    4.01     0.67      0.28      1.71            1.36
c4         4  6889       0  688.6    5.24    8.83   11.69     2.06      0.45      3.21            4.22
c16       16  6633       0  662.4   23.47   33.02   39.20    11.55      1.39     11.08           14.04

- c16: the load generator itself was overloaded (its event loop ran 14.0 ms late at p99), so
  latencies include the client's own delay. Use fewer workers, or run the client on another machine.
```

Two things to read from it. Throughput stops growing at 4 workers, the server's thread count,
while latency keeps climbing: extra requests just wait in line. And the model itself is a small
part of each request (0.28 of 2.57 ms with one worker); most of the time is HTTP, JSON and
Python, which is why batching can help.

The experiments run everything (servers, warmup, load, report) in one command each:

```bash
ibench experiment batching
ibench experiment overload
ibench coldstart --model digits-large
```

## How it works

```
 load generator (httpx2, asyncio)  ──HTTP──►  model server (FastAPI + uvicorn)
   N concurrent workers                          admission (bounded queue) → /v1/predict
   one record per request                         │  validate → (batcher) → deadline → model thread
   its own event-loop lag                         ▼
   results/<run>.jsonl                          Model: digits-small | digits-large | synthetic
          │
          ▼
 report: percentiles, throughput, errors, client lag, hardware
```

- **Models.** `digits-small` (85,002 parameters, 98.0% accurate) and `digits-large` (1.1 million,
  97.8%) are trained with scikit-learn and served as plain numpy matrix products; a test checks
  they compute exactly what scikit-learn computes. `synthetic` waits a set time per call and says
  `NOT A MODEL`, for experiments that need a controlled cost.
- **Server.** `POST /v1/predict` takes 1–64 images and answers with predictions plus the server's
  own timings (time waiting, time in the model). Optional: `--batch` (dynamic batching),
  `--max-waiting` (refuse with 503 when the queue is full), `--deadline-ms` (refuse requests that
  waited too long), `--fail-rate` (make the model fail on purpose). `/v1/model` and `/v1/stats`
  report its settings and what it did.
- **Load generator.** Closed loop: each worker sends, waits for the answer, sends again. Every
  request is recorded, failures included, and every answer is checked against the true digit.
- **Report.** Nearest-rank percentiles, shown only when the sample supports them; latency split
  into waiting, model and the rest; the machine; and warnings when something limits the result.

## Methodology

The rules behind every number are in [`docs/methodology.md`](docs/methodology.md). The short version:

- Say what's measured: real or synthetic, and every setting, in every results file.
- Keep every request, including failures; check every answer; count answers after the client's
  limit as late.
- Show a percentile only with 10 samples beyond it (p99 needs 1,000 requests).
- Measure the load generator's own lag, and warn when it, not the server, is the bottleneck.
- Repeat experiments and report ranges: the same batching run gave +36% once and +105% another time.
- Change one thing per comparison (both batching modes allow the same model parallelism).

## Testing

144 tests on Python 3.12–3.14, in CI on Linux and macOS.

- **Against real server processes,** under concurrent clients, not only an in-process test
  client: an earlier project of mine had a threading bug that only showed up that way.
- **Property tests (Hypothesis):** percentiles against their definition; batching never mixes
  up answers between requests, for random request sizes, batch limits and worker counts; batched
  predictions equal one-at-a-time predictions.
- **Failure paths:** timeouts, refusals, a server that dies mid-run, a server that gives wrong or
  malformed answers, a model that fails, an event loop deliberately blocked to produce late answers.
- **Leaks fail the tests:** an unclosed file, pipe or response is an error, not a warning. This
  caught a pipe that could have frozen a server, and (only on Python 3.14) responses left open.
- **Mutation testing, by hand:** more than 60 deliberate bugs across the milestones, each of which
  must make a test fail. The ones that survived led to new tests.

## Security

The server is for one computer: it listens on 127.0.0.1 only, rejects other Host headers (DNS
rebinding), refuses cross-site requests, grants no CORS, caps request bodies at 256 KiB, validates
every input strictly, and never returns internal error details. The load generator only targets
localhost, so it can't be pointed at someone else's server.

## Limitations

- **One machine.** Client and server share its CPUs, which limits how far the high-concurrency
  numbers can be trusted; the report's client-lag column says when.
- **Closed-loop load** understates latency under overload. An open-loop mode would fix that.
- **Small models on a CPU.** The shapes of the results (where batching starts to pay, collapse
  without overload protection) generalise better than the exact numbers. A GPU, with its large
  fixed cost per call, would make batching matter more.
- **No retries in the load generator,** on purpose, so failures stay visible.

## Possible extensions

An open-loop load mode; serving an ONNX or PyTorch model; an HTML report with charts; comparing
model versions; running client and server on separate machines.

## More

- [`docs/methodology.md`](docs/methodology.md): how the numbers are made.
- [`docs/batching.md`](docs/batching.md), [`docs/overload.md`](docs/overload.md): the experiments.
- [`docs/design.md`](docs/design.md): architecture, API, decisions.
- [`PROGRESS.md`](PROGRESS.md): how it was built, milestone by milestone, with every bug found.

## License

[MIT](LICENSE)
