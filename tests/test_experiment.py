import json
from dataclasses import replace

import pytest

from inferbench.cli import EXIT_MODEL, EXIT_OK, EXIT_USAGE, main
from inferbench.experiment import ExperimentConfig, comparison_rows, conclusion, run_experiment
from inferbench.procs import ServerStartError


def row(workers, off, on, p50_off=10.0, p50_on=8.0):
    return {"workers": workers, "throughput_off": off, "throughput_on": on, "p50_off": p50_off, "p50_on": p50_on}


def test_conclusion_states_what_the_numbers_show():
    lines = conclusion([row(1, 300, 180, 2.0, 4.0), row(4, 500, 510), row(16, 500, 600),
                        row(64, 500, 900, 100.0, 60.0), row(8, 0, 10)])
    assert lines == [
        "1 worker: batching lowered throughput to 0.60× of unbatched, median latency 2.00 → 4.00 ms.",
        "4 workers: batching made no clear difference to throughput (within 5%), median latency 10.00 → 8.00 ms.",
        "16 workers: batching raised throughput 1.20×, median latency 10.00 → 8.00 ms.",
        "64 workers: batching raised throughput 1.80×, median latency 100.00 → 60.00 ms.",
        "8 workers: a mode had no successful requests, so there is nothing to compare.",
    ]


def test_server_arguments_differ_only_by_batching():
    config = ExperimentConfig(model="synthetic", threads=2, fixed_ms=7, per_item_ms=1, batch_wait_ms=3)
    off, on = config.server_args("off"), config.server_args("on")
    assert off == ["--model", "synthetic", "--threads", "2", "--fixed-ms", "7", "--per-item-ms", "1"]
    assert on[:len(off)] == off and on[len(off):] == ["--batch", "--batch-wait-ms", "3", "--batch-max-images", "64"]
    assert "--fixed-ms" not in ExperimentConfig(model="digits-large").server_args("off")


def test_a_small_real_experiment(tmp_path, models_dir):
    config = ExperimentConfig(model="synthetic", concurrency=(1, 8), duration=0.5, warmup=0.1, threads=2,
                              fixed_ms=5, per_item_ms=0, batch_wait_ms=2)
    out_dir, results = run_experiment(config, models_dir, tmp_path / "exp", progress=lambda line: None)
    assert sorted(p.name for p in out_dir.iterdir()) == ["off-c1.jsonl", "off-c8.jsonl", "on-c1.jsonl", "on-c8.jsonl",
                                                         "report.md"]
    headers = {p.stem: json.loads(p.read_text().splitlines()[0]) for p in out_dir.glob("*.jsonl")}
    assert headers["off-c8"]["server"]["server"]["batching"] is None
    assert headers["on-c8"]["server"]["server"]["batching"] == {"max_images": 64, "max_wait_ms": 2.0, "workers": 2}
    on_c8 = [json.loads(line) for line in (out_dir / "on-c8.jsonl").read_text().splitlines()[1:-1]]
    assert max(r["server_batch"] for r in on_c8 if r["server_batch"]) > 1  # requests really were combined
    rows = comparison_rows(results)
    assert [r["workers"] for r in rows] == [1, 8] and all(r["failed"] == 0 for r in rows)
    report = (out_dir / "report.md").read_text()
    assert report.startswith("# Batching experiment: synthetic")
    assert "## What it shows" in report and "## Every run" in report and "| on-c8 |" in report


def test_bad_concurrency_levels_are_refused(tmp_path, models_dir):
    with pytest.raises(ValueError, match="concurrency"):
        run_experiment(ExperimentConfig(concurrency=(0,)), models_dir, tmp_path)


def test_an_untrained_model_is_explained(tmp_path):
    with pytest.raises(ServerStartError, match="train it first"):
        run_experiment(ExperimentConfig(model="digits-large", concurrency=(1,), duration=0.2, warmup=0),
                       tmp_path / "no-models", tmp_path / "out", progress=lambda line: None)


def test_experiment_command(capsys, tmp_path, models_dir):
    code = main(["experiment", "batching", "--model", "synthetic", "--models-dir", str(models_dir),
                 "--concurrency", "2", "--duration", "0.3", "--warmup", "0", "--threads", "2",
                 "-o", str(tmp_path / "exp")])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "2 runs of 0 + 0.3 s" in out and "2 workers: batching" in out and "report.md" in out


def test_experiment_command_errors(capsys, tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["experiment", "batching", "--concurrency", "1,two"])
    assert caught.value.code == EXIT_USAGE
    assert main(["experiment", "batching", "--model", "digits-large", "--models-dir", str(tmp_path),
                 "--concurrency", "1", "--duration", "0.2", "-o", str(tmp_path / "x")]) == EXIT_MODEL
    assert "train it first" in capsys.readouterr().err


def test_comparison_rows_put_each_number_in_its_column():
    from inferbench.metrics import run_metrics
    from tests.test_metrics import make_run

    off = replace(run_metrics(make_run([10.0] * 100 + [30.0] * 900), "off-c4"), ok_per_second=100.0)  # p50, p99 30
    on = replace(run_metrics(make_run([5.0] * 990 + [50.0] * 10, failures=["timeout"]), "on-c4"),
                 ok_per_second=150.0)  # p50, p99 5
    [r] = comparison_rows({"off": [off], "on": [on]})
    assert (r["throughput_off"], r["throughput_on"], r["throughput_change"]) == (100.0, 150.0, "+50%")
    assert (r["p50_off"], r["p50_on"], r["p50_change"]) == (30.0, 5.0, "-83%")
    assert (r["p99_off"], r["p99_on"]) == (30.0, 5.0) and r["failed"] == 1


def test_overload_modes_differ_only_in_their_protections():
    from inferbench.experiment import OverloadConfig

    config = OverloadConfig(model="synthetic", max_waiting=3, deadline_ms=20, threads=2)
    modes = dict(config.modes())
    assert list(modes) == ["no limits", "at most 3 waiting", "20 ms deadline", "both"]
    assert modes["no limits"] == [] and modes["both"] == modes["at most 3 waiting"] + modes["20 ms deadline"]
    assert config.server_args(modes["both"])[:4] == ["--model", "synthetic", "--threads", "2"]


def test_overload_conclusion_counts_wasted_work():
    from inferbench.experiment import overload_conclusion

    rows = [{"mode": "no limits", "ok_per_second": 30.0, "ok": 150, "requests": 3000, "server_answered": 3000,
             "wasted": 2700},
            {"mode": "both", "ok_per_second": 500.0, "ok": 2500, "requests": 5000, "server_answered": 2600,
             "wasted": 0}]
    assert overload_conclusion(rows) == [
        "Most useful answers per second: both (500.0/s).",
        "no limits: 150 of 3000 measured requests answered in time; the server computed 3000 answers and 2700 "
        "(90%) reached nobody, because the client had already given up.",
        "both: 2500 of 5000 measured requests answered in time; the server computed 2600 answers and 0 (0%) "
        "reached nobody, because the client had already given up.",
    ]


def test_a_small_real_overload_experiment(tmp_path, models_dir):
    # A synthetic model (40 ms a call, 1 thread) and 8 impatient clients (60 ms): without limits
    # the queue outgrows the timeout and most answers arrive too late; with them, they don't.
    from inferbench.experiment import OverloadConfig, run_overload

    config = OverloadConfig(model="synthetic", concurrency=8, duration=1.0, warmup=0.2, threads=1,
                            client_timeout=0.06, max_waiting=0, deadline_ms=10, fixed_ms=40, per_item_ms=0)
    out_dir, rows = run_overload(config, models_dir, tmp_path / "ovl", progress=lambda line: None)
    assert sorted(p.name for p in out_dir.iterdir()) == [f"mode{i}.jsonl" for i in range(4)] + ["report.md"]
    by_mode = {r["mode"]: r for r in rows}
    unprotected, protected = by_mode["no limits"], by_mode["both"]
    assert unprotected["wasted"] > protected["wasted"]
    assert protected["refused_busy"] > 0 and unprotected["refused_busy"] == 0
    assert "timeout" in unprotected["errors"]
    for index, r in enumerate(rows):
        # Answers received count every phase (the server's counter covers them all), and what
        # reached nobody is exactly the difference.
        records = [json.loads(line) for line in (out_dir / f"mode{index}.jsonl").read_text().splitlines()[1:-1]]
        assert r["client_received"] == sum(rec["status"] == 200 and not rec["error"] for rec in records)
        assert r["client_received"] <= r["server_answered"]
        assert r["wasted"] == r["server_answered"] - r["client_received"]
    assert (out_dir / "report.md").read_text().startswith("# Overload experiment: synthetic")


def test_overload_command(capsys, tmp_path, models_dir):
    code = main(["experiment", "overload", "--model", "synthetic", "--models-dir", str(models_dir),
                 "--concurrency", "4", "--duration", "0.3", "--warmup", "0", "--threads", "1",
                 "--client-timeout", "0.5", "-o", str(tmp_path / "o")])
    assert code == EXIT_OK
    assert "Most useful answers per second" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["experiment", "overload", "--concurrency", "4,8"])
