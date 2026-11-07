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
- **Repeats:** the `digits-large` experiment was run 6 times, in two sessions of 3, and the tables
  give the range. Runs 4–6 also measured the load generator's own event-loop lag. The synthetic
  experiment was run once.

## digits-large (a real model: 1.1 million parameters, 97.8% accurate)

| Workers | Requests/s without | Requests/s with | Change, runs 1–3 | Change, runs 4–6 | Median latency without → with |
|---:|---:|---:|---:|---:|---:|
| 1 | 299–361 | 179–196 | −39% to −42% | −47% | 2.6–3.2 → 5.0–5.3 ms |
| 4 | 541–685 | 517–660 | −3% to −8% | −4% to −6% | 5.2–6.7 → 5.6–7.1 ms |
| 16 | 528–655 | 725–795 | +32% to +51% | +11% to +19% | 23.2–28.7 → 19.3–20.7 ms |
| 64 | 493–673 | 857–1,043 | +74% to +105% | +36% to +56% | 93–123 → 59–68 ms |

**At 64 workers the client was overloaded,** so treat those numbers with care. In runs 4–6 the
load generator's event loop ran 14–29 ms late (p99) without batching and 61–70 ms late with it
(more answers per second means more work for the client too). Part of what was measured at 64
workers is the client's own delay, and it affects the two modes differently. The results at 1
to 16 workers are the more trustworthy ones.

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
sent: throughput fell by 39–47% in every run.

**Under heavy load, batching helps, by an amount this machine can't pin down precisely.** Every
run at 16 and 64 workers showed batching ahead on throughput and median latency, but the size
of the gain varied from +11% to +51% at 16 workers and from +36% to +105% at 64, between runs
and between sessions. One model call serving many requests is the reason it helps; client
saturation and run-to-run noise are why the size is uncertain.

**The crossover is between 4 and 16 workers here,** about where the 4 model threads are fully
busy. Below it, there are too few requests waiting to fill batches; above it, there is a queue
anyway, and batching empties it faster.

**The bigger the fixed cost per call, the more batching helps.** The synthetic model, with a
5 ms fixed cost per call, gained 2.5× at 64 workers. The real model's per-call cost is mostly
HTTP and Python overhead rather than arithmetic. Accelerators (GPUs) have large fixed costs per
call, which is why batching matters so much there.

**A better median doesn't guarantee a better tail.** At 16 workers, batching's p99 was worse in
3 of the 6 runs even though its median was better in all of them. A request that just misses a
batch waits for the next one.

## Why the report gives ranges

The same experiment on the same machine gave +105% in one run and +36% in another at 64 workers,
and the two sessions of three runs didn't even overlap at 16 workers. A single run would have
made one of those numbers look like the answer. The direction was the same every time; the size
was not.

## Caveats

- Closed-loop load understates latency under overload, in both modes.
- Client and server share 2 CPUs. At high concurrency the load generator competes with the
  server, and its measured event-loop lag shows it. A separate client machine, or a laptop with
  more cores, would give cleaner numbers at 64 workers.
- The 2 ms wait and 64-image limit are one setting. A shorter wait would cost less at light load
  and gain less at heavy load.
- A different machine will give different numbers. Run `ibench experiment batching` to measure
  yours, and look at the report's client-lag column before trusting the high-concurrency rows.
