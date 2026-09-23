# jev-eval-lab

An evaluation harness for **Jev**, TypeSafe AI's new "System One" decision model — built for the world where models return *typed decisions* instead of text.

## What is Jev?

On **September 15, 2026**, TypeSafe AI (founded by former OpenAI researcher Diogo Almeida) emerged from stealth with $40M in seed funding and released Jev, the first "System One model" — inspired by Kahneman's *Thinking, Fast and Slow*. Unlike frontier LLMs, Jev does not generate text at all. You send it a state plus a list of typed questions, and it answers all of them **in parallel, in a single call**:

| Primitive | Meaning | Example |
|---|---|---|
| **Choice** | Pick one of up to 255 options | Ticket priority: Critical / High / Medium / Low |
| **Score** | Rate on a rubric of 2–10 levels | Churn risk on a 1–5 scale |
| **Noul** | Yes/no verdict with calibrated probability ("Noul" = Bernoulli) | Does this violate the harassment policy? |

Key reported properties: ~70–500ms latency, $0.042 per million input tokens with **unmetered (free) output**, a 64k-token shared budget for state + questions, and training via **RLCD** (Reinforcement Learning for Calibrated Decisions) — calibration is a first-class property, not a prompting trick.

Sources: [TypeSafe launch announcement (Sept 15, 2026)](https://www.typesafe.ai) · [Simon Willison's deep dive (Sept 21, 2026)](https://simonwillison.net/2026/Sep/21/jev/) · [TechTarget coverage (Sept 22, 2026)](https://www.techtarget.com/it-infrastructure/news/366650696/Jev-decision-model-touted-as-quicker-cheaper-LLM-alternative) · [Medium explainer](https://medium.com/@dremfind/jev-ai-model-typesafes-system-one-model-turns-text-into-typed-decisions-39d15e751875)

## Why evaluate typed decisions differently?

Classic LLM eval (BLEU, ROUGE, LLM-as-judge) assumes string outputs. For a decision model, what matters is:

- **Schema validity** — is every output well-typed? For real Jev this holds by construction; the harness verifies it anyway.
- **Accuracy** — did it pick the right option / verdict / level?
- **Calibration** — does an 80% reported confidence mean ~80% correct? Measured with Expected Calibration Error (ECE) and reliability-diagram data.
- **Brier score** — a proper scoring rule combining accuracy and calibration.
- **Latency & cost** — Jev's whole pitch is economics: the harness estimates cost from its public pricing ($0.042/1M input tokens, output free).

## ⚠️ Offline simulator disclaimer

**Jev is in waitlisted early access — there is no public API to call.** This repo therefore ships with `OfflineSimulator`, a deterministic, seeded stand-in that generates *synthetic* decisions from transparent keyword heuristics (fully documented in `src/jev_eval/client.py`). It exists so the evaluation pipeline can be built and tested today.

**Every figure produced with the simulator is synthetic and labeled as such.** Nothing in this repo claims to benchmark the real Jev model, and no TypeSafe benchmark numbers are reproduced or invented here.

## Quickstart

```bash
git clone https://github.com/sureshbujji/jev-eval-lab.git
cd jev-eval-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ask all three primitives in one parallel call (simulated)
python examples/quickstart.py

# Run the 25-case golden dataset and print a metrics report (simulated)
python examples/ticket_triage.py

# Run the test suite
PYTHONPATH=src pytest -q
```

## Project structure

```
jev-eval-lab/
├── src/jev_eval/
│   ├── __init__.py        # public API
│   ├── models.py          # Choice/Score/Noul question + decision dataclasses
│   ├── client.py          # JevClient ABC, OfflineSimulator, RealJevClient stub
│   └── eval/
│       ├── metrics.py     # ECE, Brier, reliability diagrams, confusion matrix
│       └── runner.py      # golden-dataset runner + report printer
├── data/
│   └── golden_sample.jsonl  # 25 handcrafted cases: triage, moderation, churn
├── examples/
│   ├── quickstart.py        # one parallel ask across all three primitives
│   └── ticket_triage.py     # full golden-set eval with metrics report
├── tests/                   # pytest suite (44 tests)
├── requirements.txt         # pytest only — no heavy dependencies
└── pyproject.toml
```

### Golden dataset format

One JSON object per line:

```json
{"id": "triage-001", "domain": "ticket-triage", "primitive": "choice",
 "state": "Complete outage: checkout is down...",
 "question": {"name": "priority", "prompt": "What priority should this ticket get?",
              "options": ["Critical", "High", "Medium", "Low"]},
 "ground_truth": "Critical"}
```

`ground_truth` is an option string for `choice`, a 1-based level int for `score`, and a boolean for `noul`.

## Roadmap

- [ ] Plug in `RealJevClient` once the waitlist clears and an API key is available (the intended HTTP shape is sketched in `client.py`).
- [ ] Add calibration plots (reliability diagrams) as PNG output.
- [ ] Expand golden sets: agent tool-call routing, RAG answer verification, PII detection.
- [ ] Latency/cost comparison: Jev vs. frontier LLM on identical decision workloads.

## License

MIT — see [LICENSE](LICENSE).
