"""Quickstart: ask all three Jev primitives in one parallel call.

Run from the repo root::

    python examples/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_eval import (
    ChoiceQuestion,
    NoulQuestion,
    OfflineSimulator,
    ScoreQuestion,
)

# NOTE: OfflineSimulator is a deterministic stand-in. Jev is in waitlisted
# early access, so no real API is called here.
client = OfflineSimulator(seed=42)

state = (
    "Support ticket #4821: production checkout is down for EU customers. "
    "Sev1 outage declared, on-call paged."
)

questions = [
    ChoiceQuestion(
        name="priority",
        prompt="What priority should this ticket get?",
        options=["Critical", "High", "Medium", "Low"],
    ),
    NoulQuestion(
        name="is_outage",
        statement="This ticket describes a production outage.",
    ),
    ScoreQuestion(
        name="customer_impact",
        prompt="Rate the customer impact.",
        levels=5,
    ),
]

decisions = client.ask(state, questions)
for d in decisions:
    print(d)
