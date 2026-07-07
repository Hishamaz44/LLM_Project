"""Builds the length x order x panel-type condition matrix for one debate round and runs
the judge panel for each combination, producing flat dicts ready for results.jsonl.

Length is a *generation-time* variable now: each round already holds a For/Against argument
written to each target word count (see debate.RoundContent), so here we simply judge each
length as its own length-matched pair — no truncation. Both sides share the target length in
a given pair, so verbosity bias shows up as scores rising with length, not as a For/Against gap.
"""

from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .debate import RoundContent
from .judge import aggregate_panel, run_panel


# Counts words in `text` by whitespace splitting.
def _word_count(text: str) -> int:
    return len(text.split())


# Maps the judge's stance-blind verdict ("Debater 1"/"Debater 2" = shown first/second) back to
# a stance ("for"/"against"), which depends on which side was placed first for this pairing.
_WINNER_MAP = {
    "for_first": {"Debater 1": "for", "Debater 2": "against", "Tie": "tie"},
    "against_first": {"Debater 1": "against", "Debater 2": "for", "Tie": "tie"},
}


# Runs one debate round through every length x order x panel-type combination and returns a flat
# list of result records ready to be written out. The combinations are independent, so they fan out
# across `jobs` workers; results are still collected in the original (deterministic) order, so the
# output is byte-identical to a sequential run regardless of which condition finishes first.
def run_all_conditions(
    topic_id: str,
    topic_text: str,
    round_content: RoundContent,
    config: Config,
    jobs: int = 1,
) -> list[dict]:
    panels = {
        "homogeneous": [config.homogeneous_judge_model] * 3,
        "heterogeneous": config.heterogeneous_judge_models,
    }

    # Enumerate every (length, order, panel) condition up front, in a fixed order.
    specs = []
    for length_target in sorted(round_content.for_debater_texts):
        for_text = round_content.for_debater_texts[length_target]
        against_text = round_content.against_debater_texts[length_target]
        for order in ("for_first", "against_first"):
            for panel_type, judge_models in panels.items():
                specs.append((length_target, for_text, against_text, order, panel_type, judge_models))

    # Judges one condition and returns its finished record. Pure w.r.t. shared state (only reads
    # config/topic_text and calls the thread-safe call_model/cache), so it's safe to run in parallel.
    def evaluate_condition(spec) -> dict:
        length_target, for_text, against_text, order, panel_type, judge_models = spec
        text_first, text_second = (
            (for_text, against_text) if order == "for_first" else (against_text, for_text)
        )
        scores = run_panel(topic_text, text_first, text_second, judge_models, config)
        agg = aggregate_panel(scores)

        # agg["debater1_avg"]/["debater2_avg"] are the shown-first/shown-second slots.
        if order == "for_first":
            for_debater_avg, against_debater_avg = agg["debater1_avg"], agg["debater2_avg"]
        else:
            for_debater_avg, against_debater_avg = agg["debater2_avg"], agg["debater1_avg"]
        panel_winner = _WINNER_MAP[order][agg["panel_winner"]]

        return {
            "topic_id": topic_id,
            "round_num": round_content.round_num,
            "order": order,
            # The word budget both sides were asked to hit for this pair. The actual lengths are in
            # for_debater_word_count/against_debater_word_count — compare them to length_target to
            # see how well the models obeyed the budget.
            "length_target": length_target,
            "panel_type": panel_type,
            "for_debater_model": config.for_debater_model,
            "against_debater_model": config.against_debater_model,
            "for_debater_word_count": _word_count(for_text),
            "against_debater_word_count": _word_count(against_text),
            # Position slots (shown first / shown second) — feed the order-bias metric.
            "first_shown_avg": agg["debater1_avg"],
            "second_shown_avg": agg["debater2_avg"],
            # Stance-mapped scores — feed the length/panel metrics.
            "for_debater_avg": for_debater_avg,
            "against_debater_avg": against_debater_avg,
            "panel_winner": panel_winner,
            # Raw per-judge scores stay stance-blind (Debater 1/2 = shown first/second).
            "judges_raw": [
                {
                    "judge_model": s.judge_model,
                    "debater1": s.debater1,
                    "debater2": s.debater2,
                    "winner": s.winner,
                }
                for s in scores
            ],
        }

    if jobs > 1 and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(evaluate_condition, specs))
    return [evaluate_condition(spec) for spec in specs]
