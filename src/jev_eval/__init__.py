"""jev-eval-lab: evaluation harness for Jev typed decision models (TypeSafe AI).

Jev is a "System One" decision model: instead of generating text, it returns
typed decisions — Choice, Score, or Noul (yes/no with calibrated probability).

Because Jev is in waitlisted early access (as of Sept 2026), this package ships
with a clearly-labeled OFFLINE SIMULATOR for developing the evaluation
pipeline. No real Jev API is called anywhere in this repository.
"""

from jev_eval.client import JevClient, OfflineSimulator, RealJevClient
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

__all__ = [
    "ChoiceDecision",
    "ChoiceQuestion",
    "Decision",
    "JevClient",
    "NoulDecision",
    "NoulQuestion",
    "OfflineSimulator",
    "Question",
    "RealJevClient",
    "ScoreDecision",
    "ScoreQuestion",
]

__version__ = "0.1.0"
