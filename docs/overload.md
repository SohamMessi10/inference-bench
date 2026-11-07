# What happens when a server gets more work than it can do

`ibench experiment overload` sends more load than the server can handle, from clients that
give up after a set time, to four servers that differ only in how they protect themselves:

- **no limits:** every request waits its turn, however long the line gets;
- **a bounded queue** (`--max-waiting 4`): when 4 requests are already waiting, the next is
  refused at once with `503 busy`;
- **a deadline** (`--deadline-ms 60`): a request that has already waited 60 ms when the model
  is free is refused with `503 deadline` instead of being run;
- **both.**

For each, it compares what the clients received with what the server's own counters
(`/v1/stats`) say it computed. The difference is **wasted work**: answers the server
computed for clients that had already given up.

## Setup

- A synthetic model that takes 20 ms per call, clearly labelled as not a real model, on 2 model
  threads: it can answer about 100 requests a second.
- 16 clients in a closed loop, each giving up after 100 ms. A refused client waits 20 ms before
  sending again, in every mode, as a real client would back off.
- 1 s warmup, 5 s measured per server. Claude's Linux workspace: Intel Xeon @ 2.10 GHz, 2
  logical CPUs, Python 3.13. Three runs.
- The load generator's own event loop ran at most 3–4 ms late (p99) in every mode, so the
  client was not the bottleneck; these numbers describe the server.

## Results (three runs)

| Server | Answered in time, per second | Median latency of those | Failed requests per run | Answers that reached nobody |
|---|---:|---:|---|---:|
| no limits | **0** | - | 723–735 timeouts | 878–884 of 887–893 (99%) |
| at most 4 waiting | 91.0–92.3 | 64.6–65.5 ms | 2,099–2,130 refused as busy | 0 |
| 60 ms deadline | 79.1–81.9 | 74.1–76.0 ms | 511–523 refused as late | 0 |
| both | 91.1–91.7 | 65.0–65.1 ms | 2,086–2,115 refused as busy | 0 |

## What it shows

**Without limits, the server collapses completely.** It stays fully busy, computing about 890
answers per run, but the line grows until every answer arrives after its client has given up.
Not one request was answered in time, and 99% of the server's work reached nobody (the other 1%
went to warmup requests, before the line had grown). This is congestion collapse: a busy server
doing no useful work.

**A bounded queue keeps the server useful.** It refuses what it can't serve in time, so the
requests it accepts are answered within the limit: 91–92 per second, close to the model's
capacity of about 100, and no wasted work. The refusals are instant, so a client knows at once to
back off or try elsewhere instead of waiting 100 ms for nothing.

**A deadline alone also prevents collapse, but less efficiently.** Requests still queue, and the
server only notices they are too late when their turn comes, so fewer useful answers get through
(79–82 per second) and those that do waited longer (median 74–76 ms against 65 ms).

**Both together** behave like the bounded queue: the queue does almost all the work, and the
deadline catches the few requests that still wait too long.

## A mistake this experiment caught

The first version of this experiment used 64 clients giving up after 50 ms, against the fast
digits-large model. It reported that the protections gave 4× to 17× more useful throughput. That
was wrong, for a reason that matters to any benchmark:

- Successful requests had a median latency of 95 ms, under a 50 ms limit. On investigation the
  load generator's own event loop was running up to 34 ms late (p99): 64 clients and the server
  were sharing 2 CPUs, and the client couldn't keep up. An answer could sit ready in the socket
  and be read after the deadline without the timeout ever firing.
- The load generator now measures its own event-loop lag in every run, counts an answer that
  arrives after the client's limit as `late` rather than as a success, and the report warns when
  the client was overloaded.
- With answers counted honestly, the original setup mostly measured the client, so it was
  replaced by the one above: a server slow enough to overload with few clients.

## Caveats

- One synthetic model on one machine. The size of every number depends on the model's speed,
  the thread count, the queue size and the client's patience; the shape of the result (collapse
  without limits, near-capacity goodput with a bounded queue) is the general lesson.
- Closed-loop clients. With an open loop (requests arriving at a fixed rate whatever happens),
  the collapse without limits would be at least as severe.
- The server refuses work but doesn't prioritise it; real systems may also drop the oldest
  requests first or serve some clients ahead of others.
