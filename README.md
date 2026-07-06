# Multi-Agent Debate System

Two debater agents argue opposite sides of a contested question; a 3-judge LLM panel
scores each round on logic, evidence quality, and rhetorical fairness. The experiment then
probes three biases in those judge scores:

- **Order bias** — do the scores, and the declared winner, change when the two arguments
  swap presentation order? Every pair is judged both ways (For-first and Against-first).
- **Length / verbosity bias** — each debater composes its case at several target word counts
  (e.g. 60/120/180), and each is judged as a *same-length* pair (60 vs 60, 120 vs 120, 180 vs
  180). Length can't help one side beat the other, so a preference for longer writing shows up
  as the overall score *level* climbing with length. Because a longer argument can legitimately
  say more, there's also a **padding mode** that holds content fixed and only inflates word
  count — the cleaner test of pure verbosity bias (see [Length modes](#length-modes-generate-vs-pad)).
- **Panel diversity** — does a heterogeneous judge panel (3 different models) show less of these
  biases than a homogeneous one (1 model ×3)?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

Edit `config.yaml` and set real OpenRouter model IDs (e.g. `openai/gpt-4o-mini`,
`anthropic/claude-3.5-haiku`, `google/gemini-flash-1.5`,
`meta-llama/llama-3.1-8b-instruct`). The two debaters intentionally run on **different models**
(`for_debater_model`/`against_debater_model`) so the arguments are genuinely varied rather than
one model debating itself. This is safe because none of the bias metrics use the raw
For-vs-Against score gap: they're all measured *within* length-matched, order-swapped pairs, so a
capability difference between the two debaters is a constant that cancels out (the dashboard flags
a side that scores higher wherever it's placed as a "side effect," not order bias). You can still
set both to the same model if you want that raw gap to be capability-neutral, but it isn't
required.

**`heterogeneous_judge_models` must be 3 distinct IDs** — if they're all the same model the
heterogeneous panel collapses into the homogeneous one and the central comparison is
meaningless. Verify every ID against <https://openrouter.ai/models> before a real run.

### Length modes: generate vs. pad

`length_mode` in the config controls how the per-length variants are produced:

- **`generate`** (default, `config.yaml`) — each target length is an independent, freshly
  written argument, so content naturally grows with length. This is the natural condition, but
  a rising score could mean either verbosity bias *or* that longer arguments genuinely say more.
- **`pad`** (`config_pad.yaml`) — the shortest target is written once, then expanded to each
  longer target **without adding any new argument, reason, or evidence** — only restating and
  elaborating. Content is held fixed so only word count grows, isolating verbosity bias from the
  "longer arguments say more" confound.

Run both and compare the length score-climb: whatever climb survives in the padding run is pure
verbosity bias.

## Running

```bash
# Tests first — no API key needed, no cost, validates all the pure logic.
python -m pytest

# Smoke test: one topic, to confirm the full pipeline + your API key work.
python run_experiment.py --limit 1 --out results/smoke.jsonl

# Full experiment, generate mode — every enabled topic in debate_system/topics.py.
python run_experiment.py --out results/run.jsonl

# Full experiment, padding mode — same topics/models, content held fixed (verbosity-only test).
python run_experiment.py --config config_pad.yaml --out results/run_pad.jsonl
```

`--config` selects the config file (default `config.yaml`), `--out` the JSONL results path, and
`--limit N` runs only the first N topics. By default `topics.py` has **2 topics enabled** and
about 10 more commented out — uncomment the ones you want before a real run.

Every LLM call is cached on disk in `cache.json` (keyed by model + prompt + params), so
re-running after a crash or a bug fix doesn't re-spend budget on calls already made.

Each run also writes a human-readable Markdown transcript (debate text + round scores)
alongside the JSONL results — default `results/run.md`, or pass `--report path/to/file.md`.

## Analyzing results

The bias metrics (order effect, length effect, panel diversity) live in
`debate_system/analysis.py` as pure functions of a completed `results.jsonl` — no model or
cache access, so they're deterministic and need no API key. Two ways to view them:

```bash
# Interactive dashboard. Runs entirely off the logged results.jsonl.
streamlit run app.py
```

The dashboard has four sections — (1) order effect, (2) length effect, (3) panel diversity,
(4) drill into a single evaluation's per-judge raw scores — each with plain-language
explanations of what the numbers mean. A sidebar selector switches between any `results/*.jsonl`
file, so you can flip between the generate run (`run.jsonl`) and the padding run (`run_pad.jsonl`)
to see how much of the length climb is pure verbosity.

Or open `analysis.ipynb` for the same comparisons in notebook form (both build on
`analysis.py`, so the numbers match).

## Layout

- `config.yaml` / `config_pad.yaml` — the generate-mode and padding-mode run configs (debater
  models, judge panels, length targets, `length_mode`).
- `debate_system/topics.py` — the curated debate questions (2 enabled by default; ~10 more
  commented out — uncomment to include them).
- `debate_system/debate.py` — generates each debate (2 rounds: opening + rebuttal) at every
  configured length target; in pad mode it writes the shortest length once and expands it to the
  longer targets without adding content.
- `debate_system/judge.py` — judge prompt, response parsing, panel aggregation.
- `debate_system/evaluate.py` — the length × order × panel-type condition matrix run against
  each debate round (each length target judged as a length-matched pair).
- `debate_system/experiment.py`, `run_experiment.py` — orchestration + CLI.
- `debate_system/report.py` — writes the Markdown transcript of debate text + scores.
- `debate_system/cache.py`, `client.py` — OpenRouter call wrapper with disk caching.
- `debate_system/analysis.py` — bias metrics over `results.jsonl` (pure, offline).
- `app.py` — Streamlit dashboard on top of `analysis.py`.
- `tests/` — pure-logic unit tests (no network calls).
- `analysis.ipynb` — the same bias metrics in notebook form.
