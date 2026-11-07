# Methodology: how the numbers are made, and why they can be trusted

Benchmarks of servers go wrong in a handful of well-known ways. These are the rules this tool
follows to avoid them, and where each one is enforced.

## 1. Say what is being measured

- **Real or synthetic, always labelled.** The digits models are real neural networks, trained
  here and measured on 450 images they never saw (98.0% and 97.8% accurate). The synthetic model
  only waits a set time, and its description says `NOT A MODEL`. Every results file records
  which one was served (`/v1/model`).
- **Every setting that changes the answer is recorded,** in the header of every results file:
  model, model threads, batching, queue limit, deadline, workers, images per request, warmup,
  duration, client time limit, and the machine (OS, CPU, cores, Python and library versions).
- **The model threads are set explicitly.** FastAPI would otherwise allow 40 at once (anyio's
  default), and a benchmark would be measuring a number nobody chose.

## 2. Send load honestly, and say how

- **Closed loop.** Each worker sends a request, waits for the answer, then sends the next. This
  is simple and doesn't overload the client, but it has a known blind spot: when the server slows
  down, the workers slow down too, so fewer requests are sent exactly when the server is
  struggling, and latency under overload is understated ("coordinated omission"). Every report
  says the load was closed-loop. An open-loop mode is a possible extension.
- **Warmup is recorded but not counted.** The first requests to a process are slower; they're in
  the results file, marked `warmup`, and excluded from the numbers.
- **Cold start is measured separately** (`ibench coldstart`), by starting a fresh server for each
  trial and timing its startup and first request.
- **Clients that are refused back off** in the overload experiment (20 ms, the same in every
  mode), as real clients do, instead of resending at full speed.

## 3. Keep everything, including failures

- **Every request is recorded,** successful or not: timeouts, refusals with the server's reason
  (`http 503 busy`, `http 503 deadline`), model errors, wrong or unreadable answers.
- **Answers are checked under load.** Inputs are real held-out images, so a server that answers
  quickly but wrongly isn't counted as fast. Accuracy under load matched held-out accuracy.
- **An answer after the client's limit is `late`, not a success.** A client that gives up at
  that point can't use it.
- **Wasted work is measured,** in the overload experiment, by comparing the server's own count
  of answers (`/v1/stats`) with the answers clients received.

## 4. Report only what the samples support

- **Nearest-rank percentiles.** Each percentile is a latency some request really had, never an
  interpolation. They are computed with exact fractions (floating-point once made p99.9 ask for
  10,001 samples instead of 10,000).
- **A percentile needs 10 samples beyond it:** 20 successful requests for p50, 200 for p95,
  1,000 for p99. With fewer, a single request decides it, so the report shows `n<1000` instead.
- **Latency is split** into time waiting in the server, time in the model, and everything else
  (HTTP, JSON, the event loop), from timings the server reports with each answer.

## 5. Check the client isn't the bottleneck

- **The load generator measures its own event-loop lag** in every run: a probe asks to wake
  every millisecond and records how late it actually wakes. When the client is overloaded, part
  of every "latency" is its own delay, and timeouts fire late.
- **The report warns** when that lag (p99) is over 5 ms and over 10% of the median latency.
- This check exists because it was needed: an early overload experiment reported successes with
  a 95 ms median under a 50 ms limit. The client was running 34 ms late on a 2-CPU machine shared
  with the server; the result was withdrawn and the experiment redesigned (`overload.md`).

## 6. Run it more than once

- Experiments are reported as **ranges over repeated runs,** not single numbers. On the build
  machine, the same batching experiment gave +36% in one run and +105% in another at 64 workers:
  the direction held every time; the size did not.
- **Comparisons change one thing.** In the batching experiment both servers allow the same number
  of model calls at once (threads = batch workers), so the difference is batching alone.

## 7. Limits this tool doesn't remove

- Client and server on one machine compete for its CPUs; a separate client machine is better.
- One synthetic workload per experiment, and small real models: the shapes of the results
  (where batching starts to pay, what happens without overload protection) generalise better
  than the exact numbers.
- Run it yourself: `ibench experiment batching` and `ibench experiment overload` measure your
  machine, and the reports say how far to trust each row.
