"""Command line: `ibench train` and `ibench serve`.

Exit codes: 0 success, 1 a model can't be trained or loaded, 2 wrong command usage.
"""

from __future__ import annotations

import argparse
import sys
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
    serve.add_argument("--fixed-ms", type=float, default=5.0, help="synthetic model: cost per call")
    serve.add_argument("--per-item-ms", type=float, default=0.5, help="synthetic model: cost per input")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    except ModelError as exc:
        print(f"ibench: {exc}", file=sys.stderr)
        return EXIT_MODEL
    import uvicorn

    from inferbench.server import create_app

    if not 1 <= args.threads <= 64:
        parser.error(f"--threads must be between 1 and 64, got {args.threads}")
    print(f"serving {model.info.name} ({model.info.kind}) on http://{args.host}:{args.port} with "
          f"{args.threads} model threads; Ctrl+C to stop", flush=True)
    uvicorn.run(create_app(model, threads=args.threads), host=args.host, port=args.port,
                log_level="warning", access_log=False)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
