import debate_system.evaluate as ev
from debate_system.config import Config
from debate_system.debate import RoundContent, _opening_prompt, _rebuttal_prompt


def test_debater_prompts_ask_for_the_target_length():
    assert "60 words" in _opening_prompt("topic", "FOR", 60)
    assert "120 words" in _rebuttal_prompt("topic", "FOR", "my opening", "their opening", 120)


def _cfg(length_targets):
    return Config(
        for_debater_model="m",
        against_debater_model="m",
        homogeneous_judge_model="h",
        heterogeneous_judge_models=["a", "b", "c"],
        length_targets=length_targets,
    )


# Fixed panel aggregation so the record-building logic can be tested without any network.
_FAKE_AGG = {
    "debater1_avg": {"logic": 8.0, "evidence": 8.0, "fairness": 8.0},
    "debater2_avg": {"logic": 6.0, "evidence": 6.0, "fairness": 6.0},
    "panel_winner": "Debater 1",
}


def _patch_panel(monkeypatch):
    monkeypatch.setattr(ev, "run_panel", lambda *a, **k: [])
    monkeypatch.setattr(ev, "aggregate_panel", lambda scores: _FAKE_AGG)


def test_run_all_conditions_emits_one_record_per_length_order_panel(monkeypatch):
    _patch_panel(monkeypatch)
    rc = RoundContent(
        round_num=1,
        for_debater_texts={60: "for sixty words here", 120: "for one twenty words here now yes"},
        against_debater_texts={60: "against sixty words", 120: "against one twenty words here"},
    )
    records = ev.run_all_conditions("t01", "topic", rc, _cfg([60, 120]))

    # 2 lengths x 2 orders x 2 panels
    assert len(records) == 8
    assert {r["length_target"] for r in records} == {60, 120}


def test_run_all_conditions_logs_the_actual_generated_word_counts(monkeypatch):
    _patch_panel(monkeypatch)
    rc = RoundContent(
        round_num=1,
        for_debater_texts={60: "for sixty words here"},  # 4 words
        against_debater_texts={60: "against sixty words"},  # 3 words
    )
    r = ev.run_all_conditions("t01", "topic", rc, _cfg([60]))[0]
    assert r["for_debater_word_count"] == 4
    assert r["against_debater_word_count"] == 3


def test_run_all_conditions_maps_debater_slots_back_to_for_against_per_order(monkeypatch):
    _patch_panel(monkeypatch)  # debater1 (shown first) scores 8s and wins; debater2 scores 6s
    rc = RoundContent(round_num=1, for_debater_texts={60: "a b c"}, against_debater_texts={60: "d e f"})
    records = ev.run_all_conditions("t01", "topic", rc, _cfg([60]))

    for_first = next(r for r in records if r["order"] == "for_first")
    assert for_first["for_debater_avg"]["logic"] == 8.0  # debater1 == For when For is shown first
    assert for_first["panel_winner"] == "for"

    against_first = next(r for r in records if r["order"] == "against_first")
    assert against_first["for_debater_avg"]["logic"] == 6.0  # debater1 == Against when Against is shown first
    assert against_first["panel_winner"] == "against"


def test_run_all_conditions_output_identical_with_concurrency(monkeypatch):
    # Fanning conditions out across workers must not change the records or their order.
    _patch_panel(monkeypatch)
    rc = RoundContent(
        round_num=1,
        for_debater_texts={60: "a b c", 120: "d e f g", 180: "h i j k l"},
        against_debater_texts={60: "m n", 120: "o p q", 180: "r s t u"},
    )
    cfg = _cfg([60, 120, 180])
    sequential = ev.run_all_conditions("t01", "topic", rc, cfg, jobs=1)
    parallel = ev.run_all_conditions("t01", "topic", rc, cfg, jobs=4)
    assert parallel == sequential  # same records, same order
    assert len(sequential) == 3 * 2 * 2  # lengths x orders x panels
