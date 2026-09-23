"""Tests for the typed question/decision primitives."""

import dataclasses

import pytest

from jev_eval.models import (
    ChoiceDecision,
    ChoiceQuestion,
    NoulDecision,
    NoulQuestion,
    ScoreDecision,
    ScoreQuestion,
)


def test_choice_question_valid():
    q = ChoiceQuestion(name="p", prompt="Pick", options=["a", "b", "c"])
    assert q.options == ["a", "b", "c"]


def test_choice_question_rejects_single_option():
    with pytest.raises(ValueError):
        ChoiceQuestion(name="p", prompt="Pick", options=["only"])


def test_choice_question_rejects_too_many_options():
    with pytest.raises(ValueError):
        ChoiceQuestion(
            name="p", prompt="Pick", options=[f"opt-{i}" for i in range(256)]
        )


def test_choice_question_rejects_duplicates():
    with pytest.raises(ValueError):
        ChoiceQuestion(name="p", prompt="Pick", options=["a", "a"])


def test_score_question_valid():
    q = ScoreQuestion(name="s", prompt="Rate", levels=5)
    assert q.levels == 5


@pytest.mark.parametrize("levels", [1, 11, 0, -3])
def test_score_question_rejects_out_of_range_levels(levels):
    with pytest.raises(ValueError):
        ScoreQuestion(name="s", prompt="Rate", levels=levels)


def test_noul_question_valid():
    q = NoulQuestion(name="n", statement="It is true.")
    assert q.statement == "It is true."


def test_questions_are_frozen():
    q = NoulQuestion(name="n", statement="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.name = "changed"  # type: ignore[misc]


def test_choice_decision_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        ChoiceDecision(
            name="p", selected="a", probabilities={"a": 0.5, "b": 0.4}
        )


def test_choice_decision_selected_must_be_in_probabilities():
    with pytest.raises(ValueError):
        ChoiceDecision(
            name="p", selected="c", probabilities={"a": 0.5, "b": 0.5}
        )


def test_score_decision_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        ScoreDecision(
            name="s",
            score=6,
            levels=5,
            probabilities={i: 0.2 for i in range(1, 6)},
        )


def test_noul_decision_rejects_probability_out_of_range():
    with pytest.raises(ValueError):
        NoulDecision(name="n", probability=1.5, verdict=True)
