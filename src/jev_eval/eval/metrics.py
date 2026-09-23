"""Evaluation metrics for typed decisions.

Because Jev returns calibrated probabilities rather than text, classic
LLM-eval string metrics (BLEU, ROUGE, LLM-as-judge agreement) do not apply.
What matters for a decision model:

- Accuracy: did it pick the right option / verdict / level?
- Calibration: does a reported 80% confidence mean ~80% correct? (ECE)
- Brier score: proper scoring rule combining accuracy and calibration.
- Schema validity: is every output well-typed? (Must be 100% by
  construction for a real Jev model; the simulator is checked too.)
"""

import math
from typing import Dict, List, Sequence


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between predicted P(class) and 0/1 labels."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    if not probs:
        raise ValueError("probs must be non-empty")
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """ECE = sum_b |acc(b) - conf(b)| * |b| / n."""
    data = reliability_diagram_data(probs, labels, n_bins=n_bins)
    n = len(probs)
    return sum(
        abs(b["accuracy"] - b["confidence"]) * (b["count"] / n) for b in data
    )


def reliability_diagram_data(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Per-bin accuracy vs. confidence for reliability diagrams.

    Returns a list of dicts with keys: bin_low, bin_high, count,
    accuracy, confidence. Empty bins are omitted.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        low, high = i / n_bins, (i + 1) / n_bins
        idx = [
            j
            for j, p in enumerate(probs)
            if (low < p <= high) or (i == 0 and p == 0.0)
        ]
        if not idx:
            continue
        acc = sum(labels[j] for j in idx) / len(idx)
        conf = sum(probs[j] for j in idx) / len(idx)
        bins.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": float(len(idx)),
                "accuracy": acc,
                "confidence": conf,
            }
        )
    return bins


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Nested dict matrix[label_true][label_pred] = count."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in matrix or p not in matrix[t]:
            raise ValueError(f"unknown label in predictions: true={t!r} pred={p!r}")
        matrix[t][p] += 1
    return matrix


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100) * (len(ordered) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac
