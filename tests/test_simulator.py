"""Tests for OfflineSimulator determinism and schema guarantees."""

import pytest

from jev_eval import (
    ChoiceQuestion,
    NoulQuestion,
    OfflineSimulator,
    RealJevClient,
    ScoreQuestion,
)

STATE = "Production checkout is down. Sev1 outage declared."
QUESTIONS = [
    ChoiceQuestion(
        name="priority",
        prompt="What priority?",
        options=["Critical", "High", "Medium", "Low"],
    ),
    NoulQuestion(name="is_outage", statement="This is a production outage."),
    ScoreQuestion(name="impact", prompt="Rate impact.", levels=5),
]


def test_simulator_is_deterministic_for_same_seed():
    a = OfflineSimulator(seed=7).ask(STATE, QUESTIONS)
    b = OfflineSimulator(seed=7).ask(STATE, QUESTIONS)
    assert a == b


def test_simulator_differs_across_seeds():
    a = OfflineSimulator(seed=1).ask(STATE, QUESTIONS)
    b = OfflineSimulator(seed=2).ask(STATE, QUESTIONS)
    assert a != b


def test_simulator_answers_all_questions_in_order():
    decisions = OfflineSimulator(seed=42).ask(STATE, QUESTIONS)
    assert [d.name for d in decisions] == ["priority", "is_outage", "impact"]


def test_choice_decision_is_schema_valid():
    (d,) = OfflineSimulator(seed=42).ask(STATE, QUESTIONS[:1])
    assert d.selected in QUESTIONS[0].options
    assert set(d.probabilities) == set(QUESTIONS[0].options)
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in d.probabilities.values())


def test_score_decision_is_schema_valid():
    (d,) = OfflineSimulator(seed=42).ask(STATE, QUESTIONS[2:])
    assert 1 <= d.score <= 5
    assert set(d.probabilities) == {1, 2, 3, 4, 5}
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-9


def test_noul_decision_is_schema_valid():
    (d,) = OfflineSimulator(seed=42).ask(STATE, QUESTIONS[1:2])
    assert 0.0 <= d.probability <= 1.0
    assert d.verdict == (d.probability >= 0.5)


def test_simulated_latency_within_documented_envelope():
    decisions = OfflineSimulator(seed=42).ask(STATE, QUESTIONS)
    for d in decisions:
        assert 70.0 <= d.latency_ms <= 500.0


def test_real_client_stub_raises():
    client = RealJevClient(api_key="not-a-real-key")
    with pytest.raises(NotImplementedError, match="waitlisted early access"):
        client.ask(STATE, QUESTIONS)


def test_simulator_handles_generic_choice_without_cues():
    q = ChoiceQuestion(
        name="color", prompt="Pick a color.", options=["red", "blue"]
    )
    (d,) = OfflineSimulator(seed=42).ask("The sky is blue today.", [q])
    assert d.selected in ("red", "blue")
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-9
