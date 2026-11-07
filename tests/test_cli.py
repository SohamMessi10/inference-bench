import subprocess
import sys

import pytest

from inferbench import __version__
from inferbench.cli import EXIT_MODEL, EXIT_OK, EXIT_USAGE, main


def test_train_reports_accuracy_and_writes_weights(capsys, tmp_path):
    assert main(["train", "digits-small", "--models-dir", str(tmp_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "training digits-small" in out and "accuracy on 450 held-out images" in out
    assert "85,002 parameters" in out  # (64×256 + 256) + (256×256 + 256) + (256×10 + 10)
    assert (tmp_path / "digits-small.npz").exists()


def test_training_the_same_model_twice_on_one_command_trains_it_once(capsys, tmp_path):
    assert main(["train", "digits-small", "digits-small", "--models-dir", str(tmp_path)]) == EXIT_OK
    assert capsys.readouterr().out.count("training digits-small") == 1


@pytest.mark.parametrize("argv", [
    ["train", "gpt"],
    ["serve", "--model", "gpt"],
    ["serve", "--port", "80"],
    ["serve", "--port", "http"],
    ["serve", "--host", "0.0.0.0"],
    ["serve", "--model", "synthetic", "--threads", "0"],
    [],
])
def test_usage_errors_exit_2(capsys, argv):
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == EXIT_USAGE
    assert capsys.readouterr().err


def test_bad_synthetic_cost_exits_1(capsys):
    assert main(["serve", "--model", "synthetic", "--fixed-ms", "-5"]) == EXIT_MODEL
    assert "fixed_ms must be between" in capsys.readouterr().err


def test_command_runs_as_a_module():
    run = subprocess.run([sys.executable, "-m", "inferbench.cli", "--version"], capture_output=True, text=True)
    assert run.stdout.strip() == f"ibench {__version__}"


def test_every_command_in_the_docs_is_valid():
    # Each `ibench ...` line in the README and docs must parse, so the docs can't drift away from
    # the command line. (The design doc once showed `--concurrency 1,4,16` for `load`, which only
    # takes one number.)
    import re
    import shlex
    from pathlib import Path

    from inferbench.cli import build_parser

    root = Path(__file__).resolve().parent.parent
    lines = []
    for doc in [root / "README.md", *sorted((root / "docs").glob("*.md"))]:
        for block in re.findall(r"```(?:bash)?\n(.*?)```", doc.read_text(encoding="utf-8"), re.S):
            lines += [(doc.name, line.strip()) for line in block.splitlines() if line.strip().startswith("ibench ")]
    assert len(lines) >= 10
    parser = build_parser()
    for doc, line in lines:
        words = shlex.split(line.split("  ")[0])[1:]  # drop "ibench" and any aligned comment
        try:
            parser.parse_args(words)
        except SystemExit:
            pytest.fail(f"{doc}: `{line}` isn't a valid command")
