import json

from debate_system import analysis


# Encodes a total into one dimension so `<prefix>_total` equals the number we pass in,
# keeping the expected metrics easy to compute by hand.
def _dims(total):
    return {"logic": float(total), "evidence": 0.0, "fairness": 0.0}


# Builds one results.jsonl record; every field has a sensible default so each test only
# sets what it cares about. `for_avg`/`against_avg` are the stance scores; `first`/`second`
# are the shown-first/shown-second slots that feed the order metric.
def _rec(topic="t01", rnd=1, order="for_first", length_target=120, panel="heterogeneous",
         for_avg=15, against_avg=15, first=15, second=15, winner="for", for_wc=100, against_wc=100):
    return {
        "topic_id": topic, "round_num": rnd, "order": order, "length_target": length_target,
        "panel_type": panel, "for_debater_model": "m", "against_debater_model": "m",
        "for_debater_word_count": for_wc, "against_debater_word_count": against_wc,
        "first_shown_avg": _dims(first), "second_shown_avg": _dims(second),
        "for_debater_avg": _dims(for_avg), "against_debater_avg": _dims(against_avg),
        "panel_winner": winner,
        "judges_raw": [{"judge_model": "m", "debater1": _dims(first), "debater2": _dims(second), "winner": "Debater 1"}],
    }


def _df(records, tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return analysis.load_results(path)


def test_load_results_adds_total_and_diff_columns(tmp_path):
    df = _df([_rec(for_avg=9, against_avg=6)], tmp_path)
    assert df.loc[0, "for_debater_avg_total"] == 9.0
    assert df.loc[0, "against_debater_avg_total"] == 6.0
    assert df.loc[0, "score_diff"] == 3.0
    assert df.loc[0, "mean_score"] == 7.5  # (9 + 6) / 2


def test_load_results_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    try:
        analysis.load_results(path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_order_effect_measures_first_minus_second(tmp_path):
    df = _df([_rec(first=10, second=8), _rec(first=12, second=10)], tmp_path)  # diffs +2, +2
    assert analysis.order_effect(df)["first_minus_second"] == 2.0


def test_flip_rate_counts_winner_changes_on_order_swap(tmp_path):
    records = [
        _rec(topic="t01", order="for_first", winner="for"),
        _rec(topic="t01", order="against_first", winner="against"),  # flipped
        _rec(topic="t02", order="for_first", winner="for"),
        _rec(topic="t02", order="against_first", winner="for"),  # not flipped
    ]
    assert analysis.flip_rate(_df(records, tmp_path)) == 0.5


def test_length_effect_reports_score_climb_with_length(tmp_path):
    # Longer target -> higher mean score (mean_score = (for + against) / 2).
    records = [
        _rec(length_target=60, for_avg=10, against_avg=10),   # mean 10
        _rec(length_target=120, for_avg=14, against_avg=14),  # mean 14
        _rec(length_target=180, for_avg=18, against_avg=18),  # mean 18
    ]
    le = analysis.length_effect(_df(records, tmp_path))
    assert le["score_climb"] == 8.0            # 18 (longest) - 10 (shortest)
    assert le["corr_length_score"] > 0         # scores track length
    assert le["by_length"].loc[180] == 18.0


def test_length_effect_reports_mean_target_miss(tmp_path):
    # Target 100, actual 90 / 110 -> each side misses by 10.
    le = analysis.length_effect(_df([_rec(length_target=100, for_wc=90, against_wc=110)], tmp_path))
    assert le["mean_target_miss"] == 10.0


def test_panel_comparison_splits_flip_rate_by_panel(tmp_path):
    records = [
        _rec(panel="heterogeneous", order="for_first", winner="for"),
        _rec(panel="heterogeneous", order="against_first", winner="against"),  # hetero flips
        _rec(panel="homogeneous", order="for_first", winner="for"),
        _rec(panel="homogeneous", order="against_first", winner="for"),  # homo doesn't
    ]
    pc = analysis.panel_comparison(_df(records, tmp_path))
    assert pc.loc["heterogeneous", "flip_rate"] == 1.0
    assert pc.loc["homogeneous", "flip_rate"] == 0.0
