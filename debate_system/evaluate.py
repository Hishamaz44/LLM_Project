"""Builds the length x order x panel-type condition matrix for one debate round and runs
the judge panel for each combination, producing flat dicts ready for results.jsonl.

Length is a *generation-time* variable now: each round already holds a PRO/CON argument
written to each target word count (see debate.RoundContent), so here we simply judge each
length as its own length-matched pair — no truncation. Both sides share the target length in
a given pair, so verbosity bias shows up as scores rising with length, not as a PRO/CON gap.
"""

from .config import Config
from .debate import RoundContent
from .judge import aggregate_panel, run_panel


# Counts words in `text` by whitespace splitting.
def _word_count(text: str) -> int:
    return len(text.split())


_WINNER_MAP = {
    "pro_first": {"Debater 1": "pro", "Debater 2": "con", "Tie": "tie"},
    "con_first": {"Debater 1": "con", "Debater 2": "pro", "Tie": "tie"},
}


# Runs one debate round through every length x order x panel-type combination
# and returns a flat list of result records ready to be written out.
def run_all_conditions(
    topic_id: str,
    topic_text: str,
    round_content: RoundContent,
    config: Config,
) -> list[dict]:
    panels = {
        "homogeneous": [config.homogeneous_judge_model] * 3,
        "heterogeneous": config.heterogeneous_judge_models,
    }

    records = []
    for length_target in sorted(round_content.pro_texts):
        pro_text = round_content.pro_texts[length_target]
        con_text = round_content.con_texts[length_target]
        for order in ("pro_first", "con_first"):
            text_first, text_second = (
                (pro_text, con_text) if order == "pro_first" else (con_text, pro_text)
            )
            for panel_type, judge_models in panels.items():
                scores = run_panel(topic_text, text_first, text_second, judge_models, config)
                agg = aggregate_panel(scores)

                if order == "pro_first":
                    pro_avg, con_avg = agg["debater1_avg"], agg["debater2_avg"]
                else:
                    pro_avg, con_avg = agg["debater2_avg"], agg["debater1_avg"]
                panel_winner = _WINNER_MAP[order][agg["panel_winner"]]

                records.append(
                    {
                        "topic_id": topic_id,
                        "round_num": round_content.round_num,
                        "order": order,
                        # The word budget both sides were asked to hit for this pair. The
                        # actual lengths are in pro_word_count/con_word_count — compare them
                        # to length_target to see how well the models obeyed the budget.
                        "length_target": length_target,
                        "panel_type": panel_type,
                        "pro_model": config.pro_model,
                        "con_model": config.con_model,
                        "pro_word_count": _word_count(pro_text),
                        "con_word_count": _word_count(con_text),
                        "debater1_avg": agg["debater1_avg"],
                        "debater2_avg": agg["debater2_avg"],
                        "pro_avg": pro_avg,
                        "con_avg": con_avg,
                        "panel_winner": panel_winner,
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
                )
    return records
