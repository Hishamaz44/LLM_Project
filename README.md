# Multi-Agent Debate System

Two debater agents argue opposite sides of a contested question; a 3-judge LLM panel
scores each round on logic, evidence quality, and rhetorical fairness. The experiment
then checks whether those judge scores hold up when the presentation order of the two
arguments is swapped, and when both sides are written to different length budgets — each
debater composes its case at several target word counts (e.g. 60/120/180), judged as
same-length pairs, so we can see whether scores climb with length — and whether a
heterogeneous judge panel (3 different models) shows less of this bias than a
homogeneous one (1 model x3).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

Edit `config.yaml` and set real OpenRouter model IDs (e.g. `openai/gpt-4o-mini`,
`anthropic/claude-3.5-haiku`, `google/gemini-flash-1.5`,
`meta-llama/llama-3.1-8b-instruct`). `for_debater_model`/`against_debater_model` default to
the same model so any score asymmetry the experiment finds is attributable to presentation,
not debater capability — change this only for a secondary comparison. **`heterogeneous_judge_models`
must be 3 distinct IDs** — if they're all the same model the heterogeneous panel collapses
into the homogeneous one and the central comparison is meaningless. Verify every ID
against <https://openrouter.ai/models> before a real run.

## Running

```bash
# Tests first — no API key needed, no cost, validates all the pure logic.
python -m pytest

# Smoke test: one topic, to confirm the full pipeline + your API key work.
python run_experiment.py --limit 1 --out results/smoke.jsonl

# Full experiment (all 12 curated topics in debate_system/topics.py).
python run_experiment.py --out results/run.jsonl
```

Every LLM call is cached on disk in `cache.json` (keyed by model + prompt + params),
so re-running after a crash or a bug fix doesn't re-spend budget on calls already made.

Each run also writes a human-readable Markdown transcript (debate text + round scores)
alongside the JSONL results — default `results/run.md`, or pass `--report path/to/file.md`.

## Analyzing results

The bias metrics (order effect, length effect, panel diversity) live in
`debate_system/analysis.py` as pure functions of a completed `results.jsonl` — no model or
cache access, so they're deterministic and need no API key. Two ways to view them:

```bash
# Interactive dashboard: filter by topic/round, see the metrics + charts, and drill into
# individual judges' raw scores. Runs entirely off the logged results.jsonl.
streamlit run app.py
```

Or open `analysis.ipynb` for the same comparisons in notebook form (both build on
`analysis.py`, so the numbers match).

## Layout

- `debate_system/topics.py` — the 12 curated debate questions (edit freely).
- `debate_system/debate.py` — generates each debate (2 rounds: opening + rebuttal) at
  every configured length target.
- `debate_system/judge.py` — judge prompt, response parsing, panel aggregation.
- `debate_system/evaluate.py` — the length × order × panel-type condition matrix run
  against each debate round (each length target judged as a length-matched pair).
- `debate_system/experiment.py`, `run_experiment.py` — orchestration + CLI.
- `debate_system/report.py` — writes the Markdown transcript of debate text + scores.
- `debate_system/cache.py`, `client.py` — OpenRouter call wrapper with disk caching.
- `debate_system/analysis.py` — bias metrics over `results.jsonl` (pure, offline).
- `app.py` — Streamlit dashboard on top of `analysis.py`.
- `tests/` — pure-logic unit tests (no network calls).
- `analysis.ipynb` — the same bias metrics in notebook form.
