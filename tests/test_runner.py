"""Tests for the golden-dataset evaluation runner."""

from pathlib import Path

import pytest

from jev_eval import OfflineSimulator
from jev_eval.eval.runner import (
    build_question,
    load_cases,
    run_eval,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def test_load_cases_reads_all_25():
    cases = load_cases(str(DATA))
    assert len(cases) == 25
    primitives = {c["primitive"] for c in cases}
    assert primitives == {"choice", "score", "noul"}


def test_build_question_rejects_unknown_primitive():
    with pytest.raises(ValueError, match="unknown primitive"):
        build_question(
            {
                "primitive": "essay",
                "question": {"name": "x"},
            }
        )


def test_run_eval_rejects_empty_cases():
    with pytest.raises(ValueError, match="no cases"):
        run_eval(OfflineSimulator(seed=1), [])


def test_run_eval_report_fields():
    cases = load_cases(str(DATA))
    report = run_eval(OfflineSimulator(seed=42), cases)
    assert report.n_cases == 25
    assert report.simulated is True
    assert report.schema_validity_rate == pytest.approx(1.0)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.ece <= 1.0
    assert 0.0 <= report.brier <= 1.0
    assert report.latency_p50_ms <= report.latency_p95_ms
    assert report.est_input_tokens > 0
    assert report.est_cost_usd > 0
    # cost math: tokens / 1M * $0.042
    assert report.est_cost_usd == pytest.approx(
        report.est_input_tokens / 1_000_000 * 0.042
    )


def test_run_eval_per_primitive_metrics():
    cases = load_cases(str(DATA))
    report = run_eval(OfflineSimulator(seed=42), cases)
    assert set(report.by_primitive) == {"choice", "score", "noul"}
    for primitive, m in report.by_primitive.items():
        assert m["n"] > 0
        assert 0.0 <= m["accuracy"] <= 1.0


def test_run_eval_confusion_matrix_sums_to_choice_cases():
    cases = load_cases(str(DATA))
    report = run_eval(OfflineSimulator(seed=42), cases)
    n_choice = sum(1 for c in cases if c["primitive"] == "choice")
    total = sum(sum(row.values()) for row in report.confusion.values())
    assert total == n_choice


def test_simulator_achieves_high_accuracy_on_aligned_golden_set():
    # The golden cases are handcrafted to the simulator's documented cue
    # rules, so accuracy should be well above chance.
    cases = load_cases(str(DATA))
    report = run_eval(OfflineSimulator(seed=42), cases)
    assert report.accuracy >= 0.8
