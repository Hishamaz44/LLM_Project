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


# Builds a record with a custom multi-model judge panel, for the self-preference metric.
# `judges` is a list of (judge_model, first_slot_total, second_slot_total).
def _nep_rec(order, for_model, against_model, judges):
    return {
        "topic_id": "t01", "round_num": 1, "order": order, "length_target": 120,
        "panel_type": "heterogeneous",
        "for_debater_model": for_model, "against_debater_model": against_model,
        "for_debater_word_count": 100, "against_debater_word_count": 100,
        "first_shown_avg": _dims(0), "second_shown_avg": _dims(0),
        "for_debater_avg": _dims(0), "against_debater_avg": _dims(0),
        "panel_winner": "tie",
        "judges_raw": [
            {"judge_model": m, "debater1": _dims(ft), "debater2": _dims(st), "winner": "Tie"}
            for (m, ft, st) in judges
        ],
    }


def test_self_preference_detects_own_model_scoring_higher(tmp_path):
    # For argument written by A; for_first so Debater 1 == For. Judge A scores For 10, peers C/D
    # score the same argument 6 and 8 -> self 10 vs peer mean 7 -> delta +3.
    rec = _nep_rec("for_first", "A", "B", [("A", 10, 5), ("C", 6, 5), ("D", 8, 5)])
    sp = analysis.self_preference_effect(_df([rec], tmp_path))
    assert sp["n_comparisons"] == 1
    assert sp["delta"] == 3.0
    assert sp["scored_self_higher_rate"] == 1.0
    assert sp["by_model"].loc["A", "self_preference_delta"] == 3.0


def test_self_preference_maps_scores_through_presentation_order(tmp_path):
    # Against argument written by B; against_first so Debater 1 == Against. Judge B scores Against 9,
    # peer C scores the same argument 5 -> delta +4. A never judges, so the For side yields nothing.
    rec = _nep_rec("against_first", "A", "B", [("B", 9, 3), ("C", 5, 3)])
    sp = analysis.self_preference_effect(_df([rec], tmp_path))
    assert sp["n_comparisons"] == 1
    assert sp["delta"] == 4.0
    assert sp["by_model"].loc["B", "self_preference_delta"] == 4.0


def test_self_preference_none_when_no_judge_authored(tmp_path):
    rec = _nep_rec("for_first", "A", "B", [("C", 8, 8), ("D", 7, 7), ("E", 9, 9)])
    sp = analysis.self_preference_effect(_df([rec], tmp_path))
    assert sp["n_comparisons"] == 0
    assert sp["by_model"].empty


def test_self_preference_skips_when_no_peer_judge(tmp_path):
    # Author A is also all three judges -> everyone is "self", no peer baseline -> nothing to compare.
    rec = _nep_rec("for_first", "A", "B", [("A", 10, 5), ("A", 9, 5), ("A", 8, 5)])
    sp = analysis.self_preference_effect(_df([rec], tmp_path))
    assert sp["n_comparisons"] == 0


def test_compare_runs_one_row_per_tier_file_in_size_order(tmp_path):
    recs = [_rec(length_target=60, for_avg=10, against_avg=10),
            _rec(length_target=180, for_avg=12, against_avg=12)]
    for name in ("small", "small_pad", "big"):  # medium.jsonl absent on purpose -> skipped
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    cmp = analysis.compare_runs(tmp_path)

    # One row per existing file, sorted small<medium<big (not alphabetical), generate before pad.
    assert list(zip(cmp["size"].astype(str), cmp["mode"])) == [
        ("small", "generate"), ("small", "pad"), ("big", "generate")
    ]
    assert set(analysis.COMPARISON_METRICS).issubset(cmp.columns)


def test_compare_runs_empty_when_no_tier_files(tmp_path):
    assert analysis.compare_runs(tmp_path).empty
