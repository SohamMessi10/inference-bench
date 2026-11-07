"""Command line: `ibench train`, `serve`, `load`, `report`, and `coldstart`.

Exit codes: 0 success; 1 a model can't be trained or loaded, or a load test couldn't run or
every request failed; 2 wrong command usage.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from inferbench import __version__
from inferbench.models import DIGITS_MODELS, ModelError, load_model, train_digits

EXIT_OK, EXIT_MODEL, EXIT_USAGE = 0, 1, 2
DEFAULT_MODELS_DIR = Path("models")


def port(text: str) -> int:
    value = int(text)
    if not 1024 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"must be between 1024 and 65535, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ibench", description="Benchmark a small model server on this computer.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the digits models and report their test accuracy")
    train.add_argument("names", nargs="*", metavar="MODEL",
                       help=f"which to train: {', '.join(DIGITS_MODELS)} (default: all)")
    train.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="where weights go (default: models)")

    serve = commands.add_parser("serve", help="serve one model on 127.0.0.1")
    serve.add_argument("--model", default="digits-small", choices=[*DIGITS_MODELS, "synthetic"])
    serve.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    serve.add_argument("--port", type=port, default=8000)
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"],
                       help="this computer only: the server has no accounts")
    serve.add_argument("--threads", type=int, default=4,
                       help="how many requests can run the model at once (1 to 64; default 4)")
    serve.add_argument("--batch", action="store_true",
                       help="group requests that arrive close together into one model call")
    serve.add_argument("--batch-max-images", type=int, default=64, help="largest batch, in images (default 64)")
    serve.add_argument("--batch-wait-ms", type=float, default=2.0,
                       help="longest a request waits for others to join its batch (default 2)")
    serve.add_argument("--max-waiting", type=int, default=None,
                       help="refuse requests with 503 busy when this many are already waiting (default: no limit)")
    serve.add_argument("--deadline-ms", type=float, default=None,
                       help="refuse with 503 deadline a request that waited this long before the model (default: none)")
    serve.add_argument("--fail-rate", type=float, default=0.0,
                       help="make this fraction of model calls fail on purpose, 0 to 1 (default 0)")
    serve.add_argument("--fail-seed", type=int, default=0, help="which calls fail, with --fail-rate (default 0)")
    serve.add_argument("--fixed-ms", type=float, default=5.0, help="synthetic model: cost per call")
    serve.add_argument("--per-item-ms", type=float, default=0.5, help="synthetic model: cost per input")

    load = commands.add_parser("load", help="send load to a running server and record every request",
                               description="Closed-loop load: each worker waits for an answer before sending "
                                           "the next request. Every request is recorded, failures included.")
    load.add_argument("--url", default="http://127.0.0.1:8000", help="the server (this computer only)")
    load.add_argument("--concurrency", type=int, default=4, help="workers sending at once (default 4)")
    load.add_argument("--duration", type=float, default=10.0, help="seconds of measured load (default 10)")
    load.add_argument("--warmup", type=float, default=2.0, help="seconds of load before measuring (default 2)")
    load.add_argument("--images", type=int, default=1, help="images per request, 1 to 64 (default 1)")
    load.add_argument("--timeout", type=float, default=10.0, help="seconds before a request times out (default 10)")
    load.add_argument("--seed", type=int, default=0, help="which images are sent, in which order (default 0)")
    load.add_argument("-o", "--output", type=Path,
                      help="results file (default: results/<model>-c<concurrency>-<time>.jsonl)")

    report = commands.add_parser("report", help="compare results files in one table")
    report.add_argument("files", nargs="+", type=Path, help="results files written by ibench load")
    report.add_argument("--format", choices=["text", "markdown"], default="text")

    cold = commands.add_parser("coldstart", help="time a fresh server's startup and first request",
                               description="Start a fresh server for each trial; time its startup, its first "
                                           "request, and the typical request after that.")
    cold.add_argument("--model", default="digits-small", choices=[*DIGITS_MODELS, "synthetic"])
    cold.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    cold.add_argument("--trials", type=int, default=5, help="fresh servers to start (default 5)")
    cold.add_argument("-o", "--output", type=Path, help="also save the results as JSON")

    exp = commands.add_parser("experiment", help="run a whole experiment: servers, load, and report")
    exp.add_argument("name", choices=["batching", "overload"],
                     help="batching: the same load with and without batching; overload: heavy load with impatient "
                          "clients, against a server with and without a bounded queue and a deadline")
    # Unset options take the chosen experiment's own defaults (see ExperimentConfig and OverloadConfig).
    exp.add_argument("--model", default=None, choices=[*DIGITS_MODELS, "synthetic"],
                     help="batching: digits-large; overload: synthetic, 20 ms a call")
    exp.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    exp.add_argument("--concurrency", default=None,
                     help="batching: worker counts, comma separated (1,4,16,64); overload: one count (16)")
    exp.add_argument("--duration", type=float, default=None, help="measured seconds per run (default 5)")
    exp.add_argument("--warmup", type=float, default=None, help="warmup seconds per run (default 1)")
    exp.add_argument("--images", type=int, default=None, help="batching: images per request (default 1)")
    exp.add_argument("--threads", type=int, default=None, help="model threads (batching 4, overload 2)")
    exp.add_argument("--batch-wait-ms", type=float, default=None, help="batching: default 2")
    exp.add_argument("--batch-max-images", type=int, default=None, help="batching: default 64")
    exp.add_argument("--fixed-ms", type=float, default=None, help="synthetic model: cost per call")
    exp.add_argument("--per-item-ms", type=float, default=None, help="synthetic model: cost per input")
    exp.add_argument("--client-timeout", type=float, default=None, help="overload: seconds before a client gives up (0.1)")
    exp.add_argument("--max-waiting", type=int, default=None, help="overload: the bounded queue's size (4)")
    exp.add_argument("--deadline-ms", type=float, default=None, help="overload: the server's deadline (60)")
    exp.add_argument("--backoff-ms", type=float, default=None,
                     help="overload: how long a refused client waits before sending again (20)")
    exp.add_argument("-o", "--output", type=Path, help="folder for the results (default: results/<name>-...)")
    return parser


def _with_defaults(config_class, args, **fields):
    """Build a config from the options that were given; the rest keep the class's own defaults."""
    given = {field: getattr(args, option) for field, option in fields.items() if getattr(args, option) is not None}
    return config_class(**given)


def overload_command(args, parser) -> int:
    from inferbench.experiment import OverloadConfig, overload_conclusion, run_overload
    from inferbench.procs import ServerStartError

    if args.concurrency is not None:
        try:
            args.concurrency = int(args.concurrency)
        except ValueError:
            parser.error(f"--concurrency must be one whole number for the overload experiment, got {args.concurrency!r}")
    config = _with_defaults(OverloadConfig, args, model="model", concurrency="concurrency", duration="duration",
                            warmup="warmup", threads="threads", client_timeout="client_timeout",
                            max_waiting="max_waiting", deadline_ms="deadline_ms", backoff_ms="backoff_ms",
                            fixed_ms="fixed_ms", per_item_ms="per_item_ms")
    print(f"4 servers, {config.concurrency} workers each, {config.warmup:g} + {config.duration:g} s per server",
          flush=True)
    try:
        out_dir, rows = run_overload(config, args.models_dir, args.output,
                                     progress=lambda line: print(f"  {line}", flush=True))
    except (ValueError, ServerStartError) as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    print()
    for line in overload_conclusion(rows):
        print(f"- {line}")
    print(f"report: {out_dir / 'report.md'}")
    return EXIT_OK


def experiment_command(args, parser) -> int:
    from inferbench.experiment import ExperimentConfig, comparison_rows, conclusion, run_experiment
    from inferbench.procs import ServerStartError

    if args.name == "overload":
        return overload_command(args, parser)

    if args.concurrency is not None:
        try:
            args.concurrency = tuple(int(part) for part in args.concurrency.split(","))
        except ValueError:
            parser.error(f"--concurrency must be whole numbers separated by commas, got {args.concurrency!r}")
    config = _with_defaults(ExperimentConfig, args, model="model", concurrency="concurrency", duration="duration",
                            warmup="warmup", images="images", threads="threads", batch_wait_ms="batch_wait_ms",
                            batch_max_images="batch_max_images", fixed_ms="fixed_ms", per_item_ms="per_item_ms")
    runs = 2 * len(config.concurrency)
    print(f"{runs} runs of {config.warmup:g} + {config.duration:g} s, "
          f"about {runs * (config.warmup + config.duration) + 5:.0f} s "
          f"in all", flush=True)
    try:
        out_dir, results = run_experiment(config, args.models_dir, args.output,
                                          progress=lambda line: print(f"  {line}", flush=True))
    except (ValueError, ServerStartError) as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    print()
    for line in conclusion(comparison_rows(results)):
        print(f"- {line}")
    print(f"report: {out_dir / 'report.md'}; every request: {out_dir}/*.jsonl")
    return EXIT_OK


def report_command(args) -> int:
    from inferbench.loadgen import LoadError, read_results
    from inferbench.metrics import run_metrics
    from inferbench.report import as_markdown, as_text

    runs = []
    for path in args.files:
        try:
            runs.append(run_metrics(read_results(path), path.stem))
        except FileNotFoundError:
            print(f"ibench: {path}: no such file", file=sys.stderr)
            return EXIT_MODEL
        except (LoadError, ValueError, KeyError, TypeError) as exc:
            print(f"ibench: {path} can't be read as results ({exc})", file=sys.stderr)
            return EXIT_MODEL
    sys.stdout.write(as_markdown(runs) if args.format == "markdown" else as_text(runs))
    return EXIT_OK


def coldstart_command(args) -> int:
    import json

    from inferbench.coldstart import ColdStartError, measure

    print(f"starting {args.trials} fresh {args.model} servers, one at a time ...", flush=True)
    try:
        result = measure(args.model, args.models_dir, trials=args.trials)
    except ColdStartError as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    print(f"{'trial':>5}  {'startup s':>9}  {'first ms':>8}  {'steady ms':>9}")
    for i, t in enumerate(result["results"], 1):
        print(f"{i:>5}  {t['startup_s']:>9.3f}  {t['first_ms']:>8.2f}  {t['steady_ms']:>9.2f}")
    m = result["median"]
    print(f"{'median':>6} {m['startup_s']:>9.3f}  {m['first_ms']:>8.2f}  {m['steady_ms']:>9.2f}")
    print(f"the first request took {m['first_ms'] / m['steady_ms']:.1f}x as long as a typical one "
          f"(steady = median of the next {result['steady_requests']})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"saved to {args.output}")
    return EXIT_OK


def run_load_command(args) -> int:
    from inferbench.loadgen import LoadConfig, LoadError, run_load, summary, write_results

    config = LoadConfig(url=args.url, concurrency=args.concurrency, duration=args.duration, warmup=args.warmup,
                        images=args.images, timeout=args.timeout, seed=args.seed)
    try:
        config.check()
        print(f"loading {config.url} with {config.concurrency} workers, {config.images} image(s) per request: "
              f"{config.warmup:g} s warmup, then {config.duration:g} s measured ...", flush=True)
        run = asyncio.run(run_load(config))
    except LoadError as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.output or Path("results") / f"{run.server.get('name', 'server')}-c{config.concurrency}-{stamp}.jsonl"
    write_results(run, path)
    s = summary(run)
    kind = run.server.get("kind", "?")
    print(f"{run.server.get('name')} ({kind}): {s['requests']} requests in {s['measured_seconds']:.2f} s, "
          f"{s['ok']} ok, {s['requests'] - s['ok']} failed {s['errors'] or ''}".rstrip())
    if s["ok"]:
        accuracy = f", {s['accuracy']:.2%} of answers correct" if kind == "real" else ""
        print(f"  {s['ok_per_second']} requests/s ({s['images_per_second']} images/s), "
              f"mean latency {s['mean_latency_ms']} ms{accuracy}")
    print(f"  every request is in {path}")
    return EXIT_OK if s["ok"] else EXIT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "load":
        return run_load_command(args)
    if args.command == "report":
        return report_command(args)
    if args.command == "coldstart":
        return coldstart_command(args)
    if args.command == "experiment":
        return experiment_command(args, parser)
    if args.command == "train":
        if unknown := [n for n in args.names if n not in DIGITS_MODELS]:
            parser.error(f"unknown model {', '.join(unknown)}; choose from {', '.join(DIGITS_MODELS)}")
        for name in dict.fromkeys(args.names or DIGITS_MODELS):
            print(f"training {name} ...", flush=True)
            try:
                model = train_digits(name, args.models_dir)
            except ModelError as exc:
                print(f"ibench: {exc}", file=sys.stderr)
                return EXIT_MODEL
            d = model.info.details
            print(f"  {d['test_accuracy']:.2%} accuracy on {d['test_images']} held-out images, "
                  f"{d['parameters']:,} parameters, trained in {d['train_seconds']} s -> "
                  f"{args.models_dir / (name + '.npz')}")
        return EXIT_OK

    try:
        model = load_model(args.model, args.models_dir, **(
            {"fixed_ms": args.fixed_ms, "per_item_ms": args.per_item_ms} if args.model == "synthetic" else {}))
        if args.fail_rate:
            from inferbench.models import FlakyModel
            model = FlakyModel(model, args.fail_rate, args.fail_seed)
    except ModelError as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    import uvicorn

    from inferbench.server import create_app

    if not 1 <= args.threads <= 64:
        parser.error(f"--threads must be between 1 and 64, got {args.threads}")
    from inferbench.batcher import BatchSettings

    batching = None
    if args.batch:
        batching = BatchSettings(args.batch_max_images, args.batch_wait_ms, workers=args.threads)
        try:
            batching.check()
        except ValueError as exc:
            parser.error(str(exc))
    try:
        app = create_app(model, threads=args.threads, batching=batching, max_waiting=args.max_waiting,
                         deadline_ms=args.deadline_ms)
    except ValueError as exc:
        parser.error(str(exc))
    mode = f"batching up to {batching.max_images} images or {batching.max_wait_ms:g} ms" if batching else "no batching"
    limits = [f"at most {args.max_waiting} waiting" if args.max_waiting is not None else "",
              f"a {args.deadline_ms:g} ms deadline" if args.deadline_ms is not None else "",
              f"{args.fail_rate:.0%} of calls failing on purpose" if args.fail_rate else ""]
    extra = "".join(f", {x}" for x in limits if x)
    print(f"serving {model.info.name} ({model.info.kind}) on http://{args.host}:{args.port} with "
          f"{args.threads} model threads, {mode}{extra}; Ctrl+C to stop", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
