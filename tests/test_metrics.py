import math
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inferbench.loadgen import LoadConfig, Record, Run
from inferbench.metrics import (
    PERCENTILES, TAIL_SAMPLES, percentile, reliable_percentile, run_metrics, samples_needed,
)
from inferbench.report import as_markdown, as_text


def brute_force_percentile(values, p):
    """The definition, checked one candidate at a time: the smallest value with at least p% at or below it."""
    return min(v for v in values if sum(x <= v for x in values) >= p / 100 * len(values) - 1e-9)


@given(st.lists(st.floats(0, 1e6, allow_nan=False), min_size=1, max_size=300),
       st.sampled_from([1, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]))
def test_percentile_matches_its_definition(values, p):
    assert percentile(values, p) == brute_force_percentile(values, p)
    assert percentile(values, p) in values  # always a measured value, never interpolated


def test_percentile_examples():
    values = list(range(1, 201))  # 1..200
    assert percentile(values, 50) == 100
    assert percentile(values, 95) == 190   # not 191: 0.95 × 200 must stay exactly 190
    assert percentile(values, 99) == 198
    assert percentile(values, 100) == 200
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 0)


def test_samples_needed_leaves_ten_values_beyond_the_percentile():
    assert [samples_needed(p) for p in PERCENTILES] == [20, 200, 1000]
    assert samples_needed(99.9) == 10_000  # once 10,001, from floating-point error
    for p in (50, 80, 90, 95, 99, 99.9, 99.99):
        n, exact = samples_needed(p), Fraction(str(p)) / 100
        assert n - math.ceil(exact * n) >= TAIL_SAMPLES              # enough beyond at n
        assert (n - 1) - math.ceil(exact * (n - 1)) < TAIL_SAMPLES   # not at n - 1
    with pytest.raises(ValueError):
        samples_needed(100)


def test_reliable_percentile_refuses_small_samples():
    assert reliable_percentile(list(range(999)), 99) is None
    assert reliable_percentile(list(range(1000)), 99) == 989
    assert reliable_percentile(list(range(19)), 50) is None


def make_run(latencies, *, kind="real", failures=(), queue=1.0, model=2.0, first=50.0, machine=None):
    run = Run(LoadConfig(concurrency=4, images=2), {"name": "digits-small", "kind": kind, "server": {"threads": 4}},
              machine or {"cpu": "Test CPU", "cpus": 8, "python": "3.13.0"}, "2026-11-07T00:00:00+00:00")
    run.records.append(Record("first", 0, 0.0, first, 200, images=2, correct=2, queue_ms=0.5, inference_ms=1.0))
    run.records.append(Record("warmup", 0, 0.1, 999.0, 200, images=2, correct=2, queue_ms=0.0, inference_ms=0.0))
    t = 1.0
    for ms in latencies:
        run.records.append(Record("measure", 0, t, ms, 200, images=2, correct=1, queue_ms=queue, inference_ms=model))
        t += 0.01
    for error in failures:
        run.records.append(Record("measure", 0, t, 5.0, None, error, images=2))
    run.measured_seconds = 2.0
    return run


def test_run_metrics_from_hand_made_records():
    m = run_metrics(make_run([10.0] * 150 + [20.0] * 50, failures=["timeout", "timeout", "http 503"]), "r")
    assert (m.requests, m.ok, m.errors) == (203, 200, {"timeout": 2, "http 503": 1})
    assert m.ok_per_second == 100.0 and m.images_per_second == 200.0     # 200 ok in 2 s, 2 images each
    assert m.latency == {50: 10.0, 95: 20.0, 99: None}                   # 200 samples: p99 needs 1,000
    assert m.mean_ms == 12.5
    assert (m.wait_ms, m.model_ms, m.other_ms) == (1.0, 2.0, 9.5)        # 12.5 - 1 - 2
    assert m.first_ms == 50.0 and m.accuracy == 0.5                       # warmup's 999 ms is excluded
    assert m.error_rate == pytest.approx(3 / 203)
    assert (m.workers, m.images, m.threads) == (4, 2, 4)


def test_a_run_where_everything_failed():
    m = run_metrics(make_run([], failures=["connect"] * 5), "dead")
    assert m.ok == 0 and m.mean_ms is None and m.latency == {50: None, 95: None, 99: None}
    assert "dead" in as_text([m]) and "5 (100.0%)" in as_text([m])


def test_report_marks_unsupported_percentiles_and_explains_itself():
    small = run_metrics(make_run([10.0] * 50), "small")
    text = as_text([small])
    assert "n<200" in text and "n<1000" in text and "10.00" in text
    assert "Closed loop" in text and "p99 needs 1,000" in text
    assert "Machine: Test CPU, 8 logical CPUs, Python 3.13.0" in text


def test_report_notes_failures_synthetic_models_and_mixed_machines():
    runs = [run_metrics(make_run([5.0] * 30, failures=["timeout"]), "a"),
            run_metrics(make_run([5.0] * 30, kind="synthetic", machine={"cpu": "Other", "cpus": 2, "python": "3.12.1"}), "b")]
    text = as_text(runs)
    assert "a: 1 of 31 requests failed (timeout: 1)." in text
    assert "b: a synthetic model" in text and "digits-small (synthetic)" in text
    assert "different machines or Python versions" in text


def test_markdown_report_is_a_table():
    md = as_markdown([run_metrics(make_run([5.0] * 30), "run-1")])
    lines = md.splitlines()
    assert lines[0].startswith("| run | model |") and lines[1].startswith("|---|---|---:|")
    assert lines[2].startswith("| run-1 | digits-small |")
    assert all(line.count("|") == lines[0].count("|") for line in lines[:3])
