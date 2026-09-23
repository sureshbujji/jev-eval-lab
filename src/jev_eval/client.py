"""Client interface for Jev decision models.

Two implementations are provided:

- :class:`OfflineSimulator`: a deterministic, seeded stand-in that generates
  synthetic decisions using transparent keyword heuristics. It exists so the
  evaluation pipeline can be developed and tested while Jev is in waitlisted
  early access. IT IS NOT JEV and its outputs must never be presented as
  model benchmark numbers.
- :class:`RealJevClient`: a stub showing the intended API shape for when a
  waitlist API key becomes available. ``ask()`` raises NotImplementedError.
"""

import hashlib
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from jev_eval.models import (
    ChoiceDecision,
    ChoiceQuestion,
    Decision,
    NoulDecision,
    NoulQuestion,
    Question,
    ScoreDecision,
    ScoreQuestion,
)

# ---------------------------------------------------------------------------
# Offline simulator cue lexicons (documented, transparent heuristics)
# ---------------------------------------------------------------------------
# The simulator scores each option by how many of its cue words appear in the
# input state. Cue sets are deliberately small and generic so the behavior is
# auditable; extend them for your own domain, or replace the simulator with
# RealJevClient once you have API access.

_OPTION_CUES: Dict[str, set] = {
    "critical": {"critical", "outage", "down", "sev1", "breach", "data loss"},
    "high": {"high", "urgent", "error", "fail", "broken", "production"},
    "medium": {"medium", "moderate", "intermittent", "degraded", "slow"},
    "low": {"low", "minor", "typo", "cosmetic", "question", "how to"},
    "toxic": {"toxic", "hate", "idiot", "threat", "harass", "abuse", "kill"},
    "spam": {"spam", "scam", "crypto", "giveaway", "click here"},
    "safe": {"safe", "clean", "love", "great", "thanks", "hello"},
    "churn": {"churn", "cancel", "leaving", "competitor", "expensive", "refund"},
    "retain": {"retain", "renew", "love", "expand", "happy", "satisfied"},
}

# Generic positive/negative cue words used for the Noul/Score signal.
_POSITIVE_CUES = {
    "critical", "outage", "down", "breach", "error", "fail", "broken",
    "toxic", "hate", "idiot", "threat", "harass", "abuse", "kill",
    "spam", "scam", "fraud", "leak", "cancel", "leaving", "competitor",
    "expensive", "refund", "urgent", "sev1",
}
_NEGATIVE_CUES = {
    "love", "great", "thanks", "excellent", "awesome", "happy",
    "satisfied", "renew", "expand", "hello", "hi", "typo", "cosmetic",
    "minor", "question",
}


def _words(text: str) -> List[str]:
    return text.lower().split()


def _count_cues(text: str, cues: set) -> int:
    """Count cue hits: whole-word match for single words, substring for phrases."""
    lowered = text.lower()
    tokens = set(_words(text))
    hits = 0
    for cue in cues:
        if " " in cue:
            if cue in lowered:
                hits += 1
        elif cue in tokens:
            # count occurrences, not just presence
            hits += sum(1 for t in _words(text) if t == cue)
    return hits


def _seeded_rng(*parts: str) -> random.Random:
    """Deterministic RNG seeded from the given string parts."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class JevClient(ABC):
    """Abstract interface mirroring Jev's ask-state-answer-questions shape."""

    @abstractmethod
    def ask(
        self, state: str, questions: Sequence[Question]
    ) -> List[Decision]:
        """Answer multiple questions about ``state`` in one parallel call."""
        raise NotImplementedError


class OfflineSimulator(JevClient):
    """Deterministic offline stand-in for Jev. NOT the real model.

    Behavior (fully documented so results are reproducible):
    - Same (state, questions, seed) always yields the same decisions.
    - Cue words are counted in the *state* text only; per-option cue sets
      live in ``_OPTION_CUES`` above.
    - Choice: each option scores ``2 * |cue hits| + |word overlap|`` against
      the state; scores become logits with small seeded jitter, then a
      softmax gives the probability distribution.
    - Noul/Score: a signal ``s = (#positive cues) - (#negative cues)`` maps
      through ``sigmoid(0.9 * s)`` plus seeded jitter in [-0.04, 0.04].
    - Simulated latency is drawn uniformly from 70-500 ms, matching Jev's
      publicly reported latency envelope. It is simulated, not measured.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def ask(
        self, state: str, questions: Sequence[Question]
    ) -> List[Decision]:
        return [self._answer_one(state, q) for q in questions]

    # -- internals ------------------------------------------------------
    def _answer_one(self, state: str, question: Question) -> Decision:
        rng = _seeded_rng(str(self.seed), state, question.name)
        latency_ms = rng.uniform(70.0, 500.0)

        if isinstance(question, ChoiceQuestion):
            logits = []
            for option in question.options:
                cues = _OPTION_CUES.get(option.lower(), set())
                cue_hits = _count_cues(state, cues)
                overlap = len(set(_words(option)) & set(_words(state)))
                logits.append(2.0 * cue_hits + 1.0 * overlap + rng.uniform(-0.25, 0.25))
            probs = _softmax(logits)
            selected = question.options[int(max(range(len(probs)), key=lambda i: probs[i]))]
            return ChoiceDecision(
                name=question.name,
                selected=selected,
                probabilities=dict(zip(question.options, probs)),
                latency_ms=latency_ms,
            )

        signal = _count_cues(state, _POSITIVE_CUES) - _count_cues(
            state, _NEGATIVE_CUES
        )
        p = _sigmoid(0.9 * signal) + rng.uniform(-0.04, 0.04)
        p = min(0.99, max(0.01, p))

        if isinstance(question, NoulQuestion):
            return NoulDecision(
                name=question.name,
                probability=p,
                verdict=p >= 0.5,
                latency_ms=latency_ms,
            )

        if isinstance(question, ScoreQuestion):
            levels = question.levels
            center = 1.0 + p * (levels - 1)
            weights = [
                math.exp(-0.5 * ((lvl - center) ** 2))
                for lvl in range(1, levels + 1)
            ]
            total = sum(weights)
            prob_map = {
                lvl: w / total for lvl, w in zip(range(1, levels + 1), weights)
            }
            score = max(prob_map, key=lambda k: prob_map[k])
            return ScoreDecision(
                name=question.name,
                score=score,
                levels=levels,
                probabilities=prob_map,
                latency_ms=latency_ms,
            )

        raise TypeError(f"Unknown question type: {type(question)!r}")


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class RealJevClient(JevClient):
    """Stub for the real Jev API — usable once the waitlist clears.

    Intended HTTP shape (illustrative; endpoint details are not public):

    .. code-block:: text

        POST https://api.typesafe.ai/v1/decide        # illustrative URL
        Authorization: Bearer <JEV_API_KEY>
        Content-Type: application/json

        {
          "model": "jev-latest",                      # alias for jev-1.13.0
          "state": "<text | string[] | name-value pairs>",
          "questions": [
            {"type": "noul",   "name": "is_toxic", "statement": "..."},
            {"type": "choice", "name": "priority", "options": ["Critical", ...]},
            {"type": "score",  "name": "churn_risk", "levels": 5}
          ]
        }

    Response returns one typed decision per question with probabilities;
    outputs are unmetered (input billed at $0.042 / 1M tokens).
    """

    def __init__(self, api_key: str, model: str = "jev-latest") -> None:
        self.api_key = api_key
        self.model = model

    def ask(
        self, state: str, questions: Sequence[Question]
    ) -> List[Decision]:
        raise NotImplementedError(
            "Jev is in waitlisted early access (as of Sept 2026): no public "
            "API is available yet. Use OfflineSimulator for pipeline "
            "development, then implement this method once you have a key."
        )
