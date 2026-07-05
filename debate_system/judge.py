"""Judge panel: prompt construction, response parsing, and aggregation.

Order-swapping is just a choice of which text fills the "Debater 1" vs "Debater 2"
slot before building the prompt — the prompt template itself never changes, and the
labels are stance-neutral so the judge isn't anchored by knowing who is "Pro".
"""

import json
import re
from collections import Counter
from dataclasses import dataclass

from .client import call_model
from .config import Config

DIMENSIONS = ("logic", "evidence", "fairness")
VALID_WINNERS = ("Debater 1", "Debater 2", "Tie")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


# Builds the stance-neutral scoring prompt sent to each judge model.
def build_judge_prompt(topic: str, text_first: str, text_second: str) -> str:
    return (
        f"You are an impartial debate judge. Topic: \"{topic}\"\n\n"
        f'Debater 1 argued:\n"{text_first}"\n\n'
        f'Debater 2 argued:\n"{text_second}"\n\n'
        "Score each debater from 1 (weak) to 10 (excellent) on three dimensions:\n"
        "- logic: soundness and coherence of the reasoning\n"
        "- evidence: quality and relevance of evidence/examples used\n"
        "- fairness: whether the argument plays fair rhetorically (no strawmanning, "
        "no ad hominem, accurately represents the other side)\n\n"
        "Respond with ONLY a JSON object in this exact shape, no other text:\n"
        '{"debater1": {"logic": <int>, "evidence": <int>, "fairness": <int>}, '
        '"debater2": {"logic": <int>, "evidence": <int>, "fairness": <int>}, '
        '"winner": "Debater 1" | "Debater 2" | "Tie"}'
    )


# One judge's scores for both debaters, plus their pick of the winner.
@dataclass
class JudgeScore:
    judge_model: str
    debater1: dict
    debater2: dict
    winner: str


# Extracts and validates the JSON block from a raw judge response into a JudgeScore.
def parse_judge_response(raw: str, judge_model: str) -> JudgeScore:
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw!r}")
    try:
        # Some models emit trailing commas before `}`/`]`, which is invalid strict JSON.
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", match.group(0))
        data = json.loads(cleaned)
        winner = data["winner"]
        if winner not in VALID_WINNERS:
            raise ValueError(f"Unexpected winner value: {winner!r}")
        return JudgeScore(
            judge_model=judge_model,
            debater1={dim: float(data["debater1"][dim]) for dim in DIMENSIONS},
            debater2={dim: float(data["debater2"][dim]) for dim in DIMENSIONS},
            winner=winner,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed judge response: {raw!r}") from exc


# Validator for call_model: True iff `raw` parses into a valid JudgeScore.
def _is_parseable(raw: str) -> bool:
    try:
        parse_judge_response(raw, "validation")
        return True
    except ValueError:
        return False


# Sends the topic + text pair to every judge model in the panel and collects their scores.
def run_panel(
    topic: str,
    text_first: str,
    text_second: str,
    judge_models: list[str],
    config: Config,
) -> list[JudgeScore]:
    """Run every judge in `judge_models` once. Repeated model IDs (homogeneous panel)
    get distinct cache slots so they're independently sampled and independently cached,
    instead of returning 3 copies of one response. Malformed responses are retried
    inside `call_model` (and never cached), so what comes back always parses."""
    prompt = build_judge_prompt(topic, text_first, text_second)
    scores = []
    for slot, model in enumerate(judge_models):
        raw = call_model(
            model,
            prompt,
            max_tokens=config.judge_max_tokens,
            temperature=config.judge_temperature,
            slot=slot,
            validate=_is_parseable,
        )
        scores.append(parse_judge_response(raw, model))
    return scores


# Combines a panel's individual JudgeScores into averaged dimension scores + one winner.
def aggregate_panel(scores: list[JudgeScore]) -> dict:
    """Average each dimension across judges and pick a plurality-vote winner. Ties in the
    vote count resolve to "Tie" so a split panel gives the same result on every run."""
    debater1_avg = {dim: sum(s.debater1[dim] for s in scores) / len(scores) for dim in DIMENSIONS}
    debater2_avg = {dim: sum(s.debater2[dim] for s in scores) / len(scores) for dim in DIMENSIONS}

    counts = Counter(s.winner for s in scores)
    top_count = max(counts.values())
    leaders = [w for w, c in counts.items() if c == top_count]
    winner = leaders[0] if len(leaders) == 1 else "Tie"

    return {
        "debater1_avg": debater1_avg,
        "debater2_avg": debater2_avg,
        "panel_winner": winner,
    }
