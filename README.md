# Inference Bench

Measure how concurrency and batching affect the latency and throughput of a small model server,
on your own computer, with the sample counts, warmup and hardware needed to trust the numbers.
Work in progress: see [`docs/design.md`](docs/design.md) for the plan and
[`PROGRESS.md`](PROGRESS.md) for what's done.

```bash
python -m pip install -e ".[dev]"
ibench train
ibench serve --model digits-large
```

Then, in a second terminal:

```bash
ibench load --concurrency 4 --duration 10
```

`ibench train` trains two small neural networks that read handwritten digits, using the dataset
that ships with scikit-learn, and reports their accuracy on images they never saw. `ibench serve`
serves one of them on http://127.0.0.1:8000 (interactive docs at `/docs`). A `synthetic` model,
which only waits for a set time and says so, is there for controlled experiments.

`ibench load` sends real held-out digit images from several concurrent workers, checks every
answer, and writes every request (failures included) to a file in `results/`.

```bash
ibench report results/*.jsonl
ibench coldstart --model digits-large
```

`ibench report` compares runs in one table: throughput, p50/p95/p99 latency (only when there are
enough requests to support them), and how each request's time splits between waiting, the model,
and everything else. `ibench coldstart` starts fresh servers and times their startup and first
request.

```bash
ibench experiment batching --model digits-large
```

runs the whole batching experiment: the same load with and without dynamic batching, at 1, 4, 16
and 64 workers, with a report. Over six runs ([`docs/batching.md`](docs/batching.md)), batching
cost 39–47% of throughput with one worker and helped by 11% to 105% at 16 and 64.

```bash
ibench experiment overload
```

overloads a server whose clients give up after 100 ms. Without limits it collapsed: busy the
whole time, it answered nothing in time and 99% of its work reached nobody. With a bounded queue
it answered 91–92 requests a second in time, near its capacity ([`docs/overload.md`](docs/overload.md)).
