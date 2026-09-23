"""Tests for calibration and decision-model metrics."""

import pytest

from jev_eval.eval.metrics import (
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    mean,
    percentile,
    reliability_diagram_data,
)


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_score_known_value():
    # mean((0.5-1)^2, (0.5-0)^2) = 0.25
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_brier_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        brier_score([0.5], [1, 0])


def test_brier_score_rejects_empty():
    with pytest.raises(ValueError):
        brier_score([], [])


def test_ece_perfectly_calibrated_is_zero():
    probs = [0.2] * 5 + [0.8] * 5
    labels = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_fully_miscalibrated_is_one():
    assert expected_calibration_error([1.0] * 4, [0, 0, 0, 0]) == pytest.approx(1.0)


def test_reliability_diagram_data_structure():
    probs = [0.1, 0.9, 0.9]
    labels = [0, 1, 1]
    data = reliability_diagram_data(probs, labels, n_bins=10)
    assert len(data) == 2  # bins (0,0.1] and (0.8,0.9]
    low_bin = data[0]
    assert low_bin["bin_low"] == pytest.approx(0.0)
    assert low_bin["count"] == 1.0
    assert low_bin["accuracy"] == pytest.approx(0.0)
    assert low_bin["confidence"] == pytest.approx(0.1)


def test_reliability_diagram_omits_empty_bins():
    data = reliability_diagram_data([0.5], [1], n_bins=10)
    assert len(data) == 1


def test_confusion_matrix_counts():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    m = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert m == {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 2}}


def test_confusion_matrix_rejects_unknown_label():
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["zzz"], ["a", "b"])


def test_percentile_known_values():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)
    assert percentile([7], 50) == pytest.approx(7.0)


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_mean_rejects_empty():
    with pytest.raises(ValueError):
        mean([])
