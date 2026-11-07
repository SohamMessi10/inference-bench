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

`ibench train` trains two small neural networks that read handwritten digits, using the dataset
that ships with scikit-learn, and reports their accuracy on images they never saw. `ibench serve`
serves one of them on http://127.0.0.1:8000 (interactive docs at `/docs`). A `synthetic` model,
which only waits for a set time and says so, is there for controlled experiments.
