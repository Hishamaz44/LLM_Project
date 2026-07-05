import json

from debate_system import analysis


# Encodes a total into one dimension so `<prefix>_total` equals the number we pass in,
# keeping the expected metrics easy to compute by hand.
def _dims(total):
    return {"logic": float(total), "evidence": 0.0, "fairness": 0.0}


# Builds one results.jsonl record; every field has a sensible default so each test only
# sets what it cares about.
def _rec(topic="t01", rnd=1, order="pro_first", length_target=120, panel="heterogeneous",
         pro=15, con=15, d1=15, d2=15, winner="pro", pw=100, cw=100):
    return {
        "topic_id": topic, "round_num": rnd, "order": order, "length_target": length_target,
        "panel_type": panel, "pro_model": "m", "con_model": "m",
        "pro_word_count": pw, "con_word_count": cw,
        "debater1_avg": _dims(d1), "debater2_avg": _dims(d2),
        "pro_avg": _dims(pro), "con_avg": _dims(con),
        "panel_winner": winner,
        "judges_raw": [{"judge_model": "m", "debater1": _dims(d1), "debater2": _dims(d2), "winner": winner}],
    }


def _df(records, tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return analysis.load_results(path)


def test_load_results_adds_total_and_diff_columns(tmp_path):
    df = _df([_rec(pro=9, con=6)], tmp_path)
    assert df.loc[0, "pro_avg_total"] == 9.0
    assert df.loc[0, "con_avg_total"] == 6.0
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
    df = _df([_rec(d1=10, d2=8), _rec(d1=12, d2=10)], tmp_path)  # diffs +2, +2
    assert analysis.order_effect(df)["first_minus_second"] == 2.0


def test_flip_rate_counts_winner_changes_on_order_swap(tmp_path):
    records = [
        _rec(topic="t01", order="pro_first", winner="pro"),
        _rec(topic="t01", order="con_first", winner="con"),  # flipped
        _rec(topic="t02", order="pro_first", winner="pro"),
        _rec(topic="t02", order="con_first", winner="pro"),  # not flipped
    ]
    assert analysis.flip_rate(_df(records, tmp_path)) == 0.5


def test_length_effect_reports_score_climb_with_length(tmp_path):
    # Longer target -> higher mean score (mean_score = (pro + con) / 2).
    records = [
        _rec(length_target=60, pro=10, con=10),   # mean 10
        _rec(length_target=120, pro=14, con=14),  # mean 14
        _rec(length_target=180, pro=18, con=18),  # mean 18
    ]
    le = analysis.length_effect(_df(records, tmp_path))
    assert le["score_climb"] == 8.0            # 18 (longest) - 10 (shortest)
    assert le["corr_length_score"] > 0         # scores track length
    assert le["by_length"].loc[180] == 18.0


def test_length_effect_reports_mean_target_miss(tmp_path):
    # Target 100, actual 90 / 110 -> each side misses by 10.
    le = analysis.length_effect(_df([_rec(length_target=100, pw=90, cw=110)], tmp_path))
    assert le["mean_target_miss"] == 10.0


def test_panel_comparison_splits_flip_rate_by_panel(tmp_path):
    records = [
        _rec(panel="heterogeneous", order="pro_first", winner="pro"),
        _rec(panel="heterogeneous", order="con_first", winner="con"),  # hetero flips
        _rec(panel="homogeneous", order="pro_first", winner="pro"),
        _rec(panel="homogeneous", order="con_first", winner="pro"),  # homo doesn't
    ]
    pc = analysis.panel_comparison(_df(records, tmp_path))
    assert pc.loc["heterogeneous", "flip_rate"] == 1.0
    assert pc.loc["homogeneous", "flip_rate"] == 0.0
