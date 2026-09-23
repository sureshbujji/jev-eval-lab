"""Run the golden dataset through the offline simulator and print a report.

Run from the repo root::

    python examples/ticket_triage.py

All figures are synthetic (OfflineSimulator) and demonstrate the evaluation
pipeline, not Jev's real performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_eval import OfflineSimulator
from jev_eval.eval.runner import load_cases, print_report, run_eval

DATA = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def main() -> None:
    cases = load_cases(str(DATA))
    client = OfflineSimulator(seed=42)
    report = run_eval(client, cases)
    print_report(report)


if __name__ == "__main__":
    main()
