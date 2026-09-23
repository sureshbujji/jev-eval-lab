"""Typed question and decision models mirroring Jev's three primitives.

Jev (TypeSafe AI, Sept 2026) does not generate text. Its entire output
vocabulary is three typed primitives:

- Choice: pick exactly one option from up to 255 alternatives.
- Score:  rate an input on a rubric of 2-10 discrete levels.
- Noul:   yes/no verdict on a statement, with a calibrated probability
          ("Noul" is short for Bernoulli).

All models are frozen dataclasses so questions and decisions are hashable
and safe to reuse across golden-dataset runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class ChoiceQuestion:
    """Ask Jev to pick exactly one option from ``options``."""

    name: str
    prompt: str
    options: List[str]

    def __post_init__(self) -> None:
        if not 2 <= len(self.options) <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"Choice supports 2-{MAX_CHOICE_OPTIONS} options, "
                f"got {len(self.options)}"
            )
        if len(set(self.options)) != len(self.options):
            raise ValueError("Choice options must be unique")


@dataclass(frozen=True)
class ScoreQuestion:
    """Ask Jev to rate an input on a rubric of ``levels`` discrete levels."""

    name: str
    prompt: str
    levels: int

    def __post_init__(self) -> None:
        if not MIN_SCORE_LEVELS <= self.levels <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"Score supports {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, "
                f"got {self.levels}"
            )


@dataclass(frozen=True)
class NoulQuestion:
    """Ask Jev for a yes/no verdict on ``statement`` with a probability."""

    name: str
    statement: str


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


@dataclass(frozen=True)
class ChoiceDecision:
    """Typed result for a ChoiceQuestion."""

    name: str
    selected: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.selected not in self.probabilities:
            raise ValueError("selected option must appear in probabilities")
        total = sum(self.probabilities.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ScoreDecision:
    """Typed result for a ScoreQuestion (levels are 1-based)."""

    name: str
    score: int
    levels: int
    probabilities: Dict[int, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.score <= self.levels:
            raise ValueError(
                f"score must be within 1..{self.levels}, got {self.score}"
            )
        total = sum(self.probabilities.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")


@dataclass(frozen=True)
class NoulDecision:
    """Typed result for a NoulQuestion: P(statement is true)."""

    name: str
    probability: float
    verdict: bool
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability must be within [0, 1], got {self.probability}"
            )


Decision = Union[ChoiceDecision, ScoreDecision, NoulDecision]
