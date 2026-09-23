"""Golden-dataset evaluation runner for Jev decision models.

A golden case is one JSONL line::

    {"id": ..., "domain": ..., "primitive": "choice|score|noul",
     "state": "<input text>",
     "question": {"name": ..., ...primitive-specific fields...},
     "ground_truth": <option str | level int | bool>}

The runner asks a :class:`JevClient` for decisions, then scores per
primitive: accuracy, Expected Calibration Error, Brier score,
schema-validity rate, latency distribution, and a cost estimate from Jev's
public pricing ($0.042 / 1M input tokens, output unmetered).

All figures produced against :class:`OfflineSimulator` are synthetic and
are labeled as such in the report.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from jev_eval.client import JevClient
from jev_eval.eval.metrics import (
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    mean,
    percentile,
)
from jev_eval.models import (
    ChoiceDecision,
    ChoiceQuestion,
    NoulDecision,
    NoulQuestion,
    Question,
    ScoreDecision,
    ScoreQuestion,
)

JEV_INPUT_PRICE_PER_MTOK = 0.042  # USD per 1M input tokens (public pricing)
CHARS_PER_TOKEN = 4.0  # rough heuristic for English text; estimate only


@dataclass
class CaseResult:
    case_id: str
    domain: str
    primitive: str
    correct: bool
    prob_of_truth: float  # P(decision == ground truth)
    predicted: str  # selected option / score / verdict as string
    truth: str  # ground truth as string
    schema_valid: bool
    latency_ms: float
    input_chars: int


@dataclass
class EvalReport:
    n_cases: int
    simulated: bool
    accuracy: float
    ece: float
    brier: float
    schema_validity_rate: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    est_input_tokens: int
    est_cost_usd: float
    by_primitive: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    case_results: List[CaseResult] = field(default_factory=list)


def load_cases(path: str) -> List[Dict[str, Any]]:
    """Load golden cases from a JSONL file."""
    cases = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return cases


def build_question(case: Dict[str, Any]) -> Question:
    q = case["question"]
    primitive = case["primitive"]
    if primitive == "choice":
        return ChoiceQuestion(
            name=q["name"], prompt=q["prompt"], options=list(q["options"])
        )
    if primitive == "score":
        return ScoreQuestion(
            name=q["name"], prompt=q["prompt"], levels=int(q["levels"])
        )
    if primitive == "noul":
        return NoulQuestion(name=q["name"], statement=q["statement"])
    raise ValueError(f"unknown primitive: {primitive!r}")


def _score_case(case: Dict[str, Any], decision: Any, latency_ms: float) -> CaseResult:
    primitive = case["primitive"]
    truth = case["ground_truth"]
    schema_valid = True

    if primitive == "choice":
        assert isinstance(decision, ChoiceDecision)
        schema_valid = (
            decision.selected in decision.probabilities
            and abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
            and set(decision.probabilities) == set(case["question"]["options"])
        )
        correct = decision.selected == truth
        prob_of_truth = decision.probabilities.get(truth, 0.0)
        predicted, truth_s = decision.selected, str(truth)
    elif primitive == "score":
        assert isinstance(decision, ScoreDecision)
        schema_valid = (
            1 <= decision.score <= decision.levels
            and abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
        )
        correct = decision.score == truth
        prob_of_truth = decision.probabilities.get(truth, 0.0)
        predicted, truth_s = str(decision.score), str(truth)
    elif primitive == "noul":
        assert isinstance(decision, NoulDecision)
        schema_valid = 0.0 <= decision.probability <= 1.0
        correct = decision.verdict == truth
        prob_of_truth = decision.probability if truth else 1.0 - decision.probability
        predicted, truth_s = str(decision.verdict), str(truth)
    else:
        raise ValueError(f"unknown primitive: {primitive!r}")

    state = case["state"]
    qtext = json.dumps(case["question"], sort_keys=True)
    return CaseResult(
        case_id=case["id"],
        domain=case["domain"],
        primitive=primitive,
        correct=bool(correct),
        prob_of_truth=float(prob_of_truth),
        predicted=predicted,
        truth=truth_s,
        schema_valid=bool(schema_valid),
        latency_ms=float(latency_ms),
        input_chars=len(state) + len(qtext),
    )


def run_eval(
    client: JevClient, cases: Sequence[Dict[str, Any]], simulated: bool = True
) -> EvalReport:
    """Run all golden cases against ``client`` and compute metrics."""
    from jev_eval.client import OfflineSimulator

    if not cases:
        raise ValueError("no cases to evaluate")
    simulated = simulated and isinstance(client, OfflineSimulator)

    results: List[CaseResult] = []
    for case in cases:
        question = build_question(case)
        (decision,) = client.ask(case["state"], [question])
        results.append(_score_case(case, decision, decision.latency_ms))

    correct = [1 if r.correct else 0 for r in results]
    probs = [r.prob_of_truth for r in results]
    latencies = [r.latency_ms for r in results]
    total_chars = sum(r.input_chars for r in results)
    est_tokens = int(total_chars / CHARS_PER_TOKEN)

    by_primitive: Dict[str, Dict[str, float]] = {}
    for primitive in ("choice", "score", "noul"):
        subset = [r for r in results if r.primitive == primitive]
        if not subset:
            continue
        sub_correct = [1 if r.correct else 0 for r in subset]
        sub_probs = [r.prob_of_truth for r in subset]
        by_primitive[primitive] = {
            "n": float(len(subset)),
            "accuracy": mean(sub_correct),
            "ece": expected_calibration_error(sub_probs, sub_correct),
            "brier": brier_score(sub_probs, sub_correct),
        }

    choice_results = [r for r in results if r.primitive == "choice"]
    confusion: Dict[str, Dict[str, int]] = {}
    if choice_results:
        labels: List[str] = []
        for case in cases:
            if case["primitive"] == "choice":
                for opt in case["question"]["options"]:
                    if opt not in labels:
                        labels.append(opt)
        confusion = confusion_matrix(
            [r.truth for r in choice_results],
            [r.predicted for r in choice_results],
            labels,
        )

    return EvalReport(
        n_cases=len(results),
        simulated=simulated,
        accuracy=mean(correct),
        ece=expected_calibration_error(probs, correct),
        brier=brier_score(probs, correct),
        schema_validity_rate=mean([1.0 if r.schema_valid else 0.0 for r in results]),
        latency_mean_ms=mean(latencies),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        est_input_tokens=est_tokens,
        est_cost_usd=est_tokens / 1_000_000 * JEV_INPUT_PRICE_PER_MTOK,
        by_primitive=by_primitive,
        confusion=confusion,
        case_results=results,
    )


def print_report(report: EvalReport) -> None:
    """Print a human-readable metrics report to stdout."""
    tag = "SIMULATED (OfflineSimulator)" if report.simulated else "LIVE"
    print(f"=== Jev eval report [{tag}] ===")
    print(f"cases:                 {report.n_cases}")
    print(f"accuracy:              {report.accuracy:.3f}")
    print(f"expected cal. error:   {report.ece:.3f}")
    print(f"brier score:           {report.brier:.3f}")
    print(f"schema validity:       {report.schema_validity_rate:.1%}")
    print(
        f"latency (simulated)    mean={report.latency_mean_ms:.0f}ms "
        f"p50={report.latency_p50_ms:.0f}ms p95={report.latency_p95_ms:.0f}ms"
    )
    print(
        f"cost estimate:         ~{report.est_input_tokens} input tokens "
        f"-> ${report.est_cost_usd:.6f} @ $0.042/1M (output free)"
    )
    print("\n-- per primitive --")
    for primitive, m in report.by_primitive.items():
        print(
            f"  {primitive:<7} n={m['n']:.0f} acc={m['accuracy']:.3f} "
            f"ece={m['ece']:.3f} brier={m['brier']:.3f}"
        )
    if report.confusion:
        labels = list(report.confusion)
        print("\n-- confusion matrix (choice) --")
        print("  true\\pred | " + " ".join(f"{l[:8]:>8}" for l in labels))
        for t in labels:
            row = " ".join(f"{report.confusion[t][p]:>8}" for p in labels)
            print(f"  {t[:9]:<9} | {row}")
    if report.simulated:
        print(
            "\nNOTE: all figures above are synthetic, produced by the offline "
            "simulator for pipeline validation. They are not Jev benchmarks."
        )
