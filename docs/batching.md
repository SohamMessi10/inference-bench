# Does batching help? The experiment

`ibench experiment batching` starts the same model server twice, once without and once with
dynamic batching, and sends the same load to each at several concurrency levels. Everything else
is identical: the model, 4 model threads (4 batch workers when batching), the images and their
order, 1 s of warmup and 5 s of measured load per run. Batching groups requests that arrive
within 2 ms of the first one, up to 64 images.

## Setup

- **Machine:** Claude's Linux workspace, Intel Xeon @ 2.10 GHz, 2 logical CPUs, Python 3.13.16.
  The load generator and the server share those 2 CPUs.
- **Load:** closed loop, one image per request. Every request was checked; none failed.
- **Repeats:** the `digits-large` experiment was run 3 times, and the tables give the range. The
  synthetic experiment was run once.

## digits-large (a real model: 1.1 million parameters, 97.8% accurate)

| Workers | Requests/s without | Requests/s with | Change | Median latency without → with |
|---:|---:|---:|---:|---:|
| 1 | 299–338 | 179–196 | −39% to −42% | 2.8–3.2 → 5.0–5.3 ms |
| 4 | 541–565 | 517–550 | −3% to −8% | 6.5–6.7 → 6.8–7.1 ms |
| 16 | 528–573 | 745–795 | +32% to +51% | 27.1–28.7 → 19.3–20.0 ms |
| 64 | 493–554 | 857–1,043 | +74% to +105% | 112–123 → 59–68 ms |

## synthetic (not a model: waits 5 ms per call + 0.5 ms per image)

| Workers | Requests/s without | Requests/s with | Change | Median latency without → with |
|---:|---:|---:|---:|---:|
| 1 | 110.9 | 89.2 | −20% | 8.76 → 10.87 ms |
| 4 | 401.0 | 326.7 | −19% | 9.49 → 11.70 ms |
| 16 | 499.9 | 837.2 | +67% | 31.51 → 18.40 ms |
| 64 | 468.7 | 1,186.6 | +153% | 130.14 → 52.57 ms |

## What it shows

**Under light load, batching costs latency and gains nothing.** With one worker there is never
anyone to batch with, so every request waits the full 2 ms for company that doesn't come. Median
latency rises by about that 2 ms, and with a closed loop, slower answers mean fewer requests
sent: throughput falls by about 40%.

**Under heavy load, batching is better on both counts.** With 64 workers, unbatched requests
queue for one of the 4 model threads and each pays the whole per-call cost alone. Batched, one
model call serves many requests, so throughput rose 74–105% and median latency roughly halved.

**The crossover is between 4 and 16 workers here,** about where the 4 model threads are fully
busy. Below it, there are too few requests waiting to fill batches; above it, there is a queue
anyway, and batching empties it faster.

**The bigger the fixed cost per call, the more batching helps.** The synthetic model, with a
5 ms fixed cost per call, gained 2.5× at 64 workers; the real model, whose per-call cost is
mostly HTTP and Python overhead rather than arithmetic, gained 1.7–2.1×. Accelerators (GPUs)
have large fixed costs per call, which is why batching matters so much there.

**A better median doesn't guarantee a better tail.** At 16 workers, batching's p99 was better in
two runs and slightly worse in one (46.3 vs 44.4 ms). A request that just misses a batch waits
for the next one.

## Why the report gives ranges

The same experiment, run three times on the same machine, moved by up to 31 percentage points at
64 workers (+74% to +105%). The direction of each result was the same every time; the size was
not. One run would have overstated how precise any of these numbers is.

## Caveats

- Closed-loop load understates latency under overload, in both modes.
- Client and server share 2 CPUs, so at high concurrency the load generator competes with the
  server; a separate client machine would likely show higher throughput in both modes.
- The 2 ms wait and 64-image limit are one setting. A shorter wait would cost less at light load
  and gain less at heavy load.
- A different machine, an Apple Silicon laptop for example, will give different numbers. Run
  `ibench experiment batching` to measure yours.
