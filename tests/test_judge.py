import pytest

from debate_system.judge import JudgeScore, aggregate_panel, build_judge_prompt, parse_judge_response


def test_build_judge_prompt_places_each_text_in_the_correct_slot():
    prompt = build_judge_prompt("topic?", "FIRST_TEXT", "SECOND_TEXT")
    assert 'Debater 1 argued:\n"FIRST_TEXT"' in prompt
    assert 'Debater 2 argued:\n"SECOND_TEXT"' in prompt


def test_build_judge_prompt_swap_just_changes_which_text_is_first():
    pro, con = "PRO_TEXT", "CON_TEXT"
    pro_first = build_judge_prompt("topic?", pro, con)
    con_first = build_judge_prompt("topic?", con, pro)
    assert 'Debater 1 argued:\n"PRO_TEXT"' in pro_first
    assert 'Debater 1 argued:\n"CON_TEXT"' in con_first


def test_parse_judge_response_extracts_well_formed_json():
    raw = (
        '{"debater1": {"logic": 7, "evidence": 6, "fairness": 8}, '
        '"debater2": {"logic": 5, "evidence": 4, "fairness": 6}, '
        '"winner": "Debater 1"}'
    )
    score = parse_judge_response(raw, "test-model")
    assert score.debater1 == {"logic": 7.0, "evidence": 6.0, "fairness": 8.0}
    assert score.debater2 == {"logic": 5.0, "evidence": 4.0, "fairness": 6.0}
    assert score.winner == "Debater 1"


def test_parse_judge_response_extracts_json_surrounded_by_extra_text():
    raw = (
        "Sure, here is my evaluation:\n"
        '{"debater1": {"logic": 7, "evidence": 6, "fairness": 8}, '
        '"debater2": {"logic": 5, "evidence": 4, "fairness": 6}, '
        '"winner": "Tie"}\n'
        "Let me know if you need anything else."
    )
    score = parse_judge_response(raw, "test-model")
    assert score.winner == "Tie"


def test_parse_judge_response_raises_value_error_on_malformed_output():
    with pytest.raises(ValueError):
        parse_judge_response("not even close to json", "test-model")


def test_parse_judge_response_raises_value_error_on_missing_fields():
    raw = '{"debater1": {"logic": 7}, "winner": "Tie"}'
    with pytest.raises(ValueError):
        parse_judge_response(raw, "test-model")


def test_parse_judge_response_rejects_out_of_set_winner():
    raw = (
        '{"debater1": {"logic": 7, "evidence": 6, "fairness": 8}, '
        '"debater2": {"logic": 5, "evidence": 4, "fairness": 6}, '
        '"winner": "PRO"}'
    )
    with pytest.raises(ValueError):
        parse_judge_response(raw, "test-model")


def test_aggregate_panel_averages_scores_and_picks_majority_winner():
    scores = [
        JudgeScore(
            "m1",
            {"logic": 8, "evidence": 8, "fairness": 8},
            {"logic": 2, "evidence": 2, "fairness": 2},
            "Debater 1",
        ),
        JudgeScore(
            "m2",
            {"logic": 6, "evidence": 6, "fairness": 6},
            {"logic": 4, "evidence": 4, "fairness": 4},
            "Debater 1",
        ),
        JudgeScore(
            "m3",
            {"logic": 4, "evidence": 4, "fairness": 4},
            {"logic": 6, "evidence": 6, "fairness": 6},
            "Debater 2",
        ),
    ]
    agg = aggregate_panel(scores)
    assert agg["debater1_avg"]["logic"] == 6.0
    assert agg["debater2_avg"]["logic"] == 4.0
    assert agg["panel_winner"] == "Debater 1"


def test_aggregate_panel_resolves_a_three_way_split_to_tie():
    scores = [
        JudgeScore("m1", {"logic": 5, "evidence": 5, "fairness": 5}, {"logic": 5, "evidence": 5, "fairness": 5}, "Debater 1"),
        JudgeScore("m2", {"logic": 5, "evidence": 5, "fairness": 5}, {"logic": 5, "evidence": 5, "fairness": 5}, "Debater 2"),
        JudgeScore("m3", {"logic": 5, "evidence": 5, "fairness": 5}, {"logic": 5, "evidence": 5, "fairness": 5}, "Tie"),
    ]
    assert aggregate_panel(scores)["panel_winner"] == "Tie"
