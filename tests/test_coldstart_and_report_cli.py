import json

import pytest

from inferbench.cli import EXIT_MODEL, EXIT_OK, main
from inferbench.coldstart import ColdStartError, measure, one_trial
from inferbench.loadgen import write_results
from tests.test_metrics import make_run


def test_cold_start_times_a_fresh_server(models_dir):
    trial = one_trial("digits-small", models_dir, steady_requests=5)
    assert 0 < trial.startup_s < 30
    assert trial.first_ms > 0 and trial.steady_ms > 0


def test_cold_start_summary(models_dir):
    result = measure("digits-small", models_dir, trials=2, steady_requests=3)
    assert result["trials"] == 2 and len(result["results"]) == 2
    assert result["median"]["startup_s"] > 0 and result["machine"]["cpus"]


def test_cold_start_reports_a_server_that_cannot_start(tmp_path):
    with pytest.raises(ColdStartError, match="the server exited.*train it first"):
        one_trial("digits-large", tmp_path)


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"trials": 51}, {"steady_requests": 0}])
def test_cold_start_settings_are_checked(models_dir, kwargs):
    with pytest.raises(ColdStartError):
        measure("digits-small", models_dir, **kwargs)
    with pytest.raises(ColdStartError, match="unknown model"):
        measure("gpt", models_dir)


def test_coldstart_command(capsys, models_dir, tmp_path):
    out_file = tmp_path / "cold.json"
    assert main(["coldstart", "--model", "digits-small", "--models-dir", str(models_dir), "--trials", "2",
                 "-o", str(out_file)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "median" in out and "the first request took" in out
    assert json.loads(out_file.read_text())["type"] == "coldstart"


def test_coldstart_command_explains_failure(capsys, tmp_path):
    assert main(["coldstart", "--model", "digits-large", "--models-dir", str(tmp_path), "--trials", "1"]) == EXIT_MODEL
    assert "train it first" in capsys.readouterr().err


def test_report_command(capsys, tmp_path):
    for name in ("one", "two"):
        write_results(make_run([5.0] * 40), tmp_path / f"{name}.jsonl")
    assert main(["report", str(tmp_path / "one.jsonl"), str(tmp_path / "two.jsonl")]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("one ") and out.splitlines()[2].startswith("two ")
    assert main(["report", str(tmp_path / "one.jsonl"), "--format", "markdown"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("| run |")


def test_report_command_refuses_missing_and_foreign_files(capsys, tmp_path):
    assert main(["report", str(tmp_path / "nope.jsonl")]) == EXIT_MODEL
    assert "no such file" in capsys.readouterr().err
    other = tmp_path / "other.jsonl"
    other.write_text('{"a": 1}\n')
    assert main(["report", str(other)]) == EXIT_MODEL
    assert "can't be read as results" in capsys.readouterr().err


def test_cold_start_reports_a_server_whose_requests_fail(models_dir):
    # The model fails every call, so the first request gets a 500. (This error path once left
    # the HTTP response open, which Python 3.14 reports as a leak.)
    with pytest.raises(ColdStartError, match="a request to the new server failed: HTTP Error 500"):
        one_trial("digits-small", models_dir, steady_requests=1, server_args=("--fail-rate", "1"))
