"""Generates the debate transcript: independent openings, then rebuttals."""

from dataclasses import dataclass

from .client import call_model
from .config import Config


# One round's PRO/CON text at each target length. `pro_texts`/`con_texts` map a target word
# count (e.g. 60/120/180) to the argument the debater wrote to that budget, so a round holds
# a full set of length-controlled variants rather than a single natural-length text.
@dataclass
class RoundContent:
    round_num: int
    pro_texts: dict[int, str]
    con_texts: dict[int, str]


# Full transcript of a debate on one topic: which models argued, and every round's content.
@dataclass
class Debate:
    topic_id: str
    topic_text: str
    pro_model: str
    con_model: str
    rounds: list[RoundContent]


# Builds the prompt for a debater's opening statement at a target length.
def _opening_prompt(topic: str, stance: str, target_words: int) -> str:
    return (
        f'Debate topic: "{topic}"\n\n'
        f"Argue the {stance} position as persuasively as you can, using logic and concrete evidence.\n"
        f"Make a complete, self-contained case in about {target_words} words (aim for that length). "
        "Do not mention these instructions."
    )


# Builds the prompt for a debater's rebuttal at a target length, given both openings.
def _rebuttal_prompt(topic: str, stance: str, own_opening: str, opponent_opening: str, target_words: int) -> str:
    return (
        f'Debate topic: "{topic}"\n\n'
        f'You are arguing the {stance} position. Your opening statement was:\n"{own_opening}"\n\n'
        f'Your opponent\'s opening statement was:\n"{opponent_opening}"\n\n'
        "Write a rebuttal that directly addresses their strongest point and reinforces your case.\n"
        f"Make it a complete rebuttal in about {target_words} words (aim for that length). "
        "Do not mention these instructions."
    )


# Generates the openings (and rebuttals, if 2 rounds) for one target length. Returns the
# round-1 and round-2 PRO/CON texts; round-2 texts are empty when config.rounds < 2.
def _generate_at_length(topic_text: str, target: int, config: Config) -> tuple[str, str, str, str]:
    pro_opening = call_model(
        config.pro_model,
        _opening_prompt(topic_text, "FOR", target),
        max_tokens=config.debater_max_tokens,
        temperature=config.debater_temperature,
    )
    con_opening = call_model(
        config.con_model,
        _opening_prompt(topic_text, "AGAINST", target),
        max_tokens=config.debater_max_tokens,
        temperature=config.debater_temperature,
    )
    pro_rebuttal = con_rebuttal = ""
    if config.rounds >= 2:
        pro_rebuttal = call_model(
            config.pro_model,
            _rebuttal_prompt(topic_text, "FOR", pro_opening, con_opening, target),
            max_tokens=config.debater_max_tokens,
            temperature=config.debater_temperature,
        )
        con_rebuttal = call_model(
            config.con_model,
            _rebuttal_prompt(topic_text, "AGAINST", con_opening, pro_opening, target),
            max_tokens=config.debater_max_tokens,
            temperature=config.debater_temperature,
        )
    return pro_opening, con_opening, pro_rebuttal, con_rebuttal


# Runs the full debate for one topic at every configured target length. Each length is an
# independent, length-matched mini-debate (a rebuttal at length L answers the opening at L),
# so every judged pair is coherent and matched.
def generate_debate(topic_id: str, topic_text: str, config: Config) -> Debate:
    round1_pro, round1_con, round2_pro, round2_con = {}, {}, {}, {}
    for target in config.length_targets:
        pro_open, con_open, pro_reb, con_reb = _generate_at_length(topic_text, target, config)
        round1_pro[target], round1_con[target] = pro_open, con_open
        if config.rounds >= 2:
            round2_pro[target], round2_con[target] = pro_reb, con_reb

    rounds = [RoundContent(round_num=1, pro_texts=round1_pro, con_texts=round1_con)]
    if config.rounds >= 2:
        rounds.append(RoundContent(round_num=2, pro_texts=round2_pro, con_texts=round2_con))

    return Debate(
        topic_id=topic_id,
        topic_text=topic_text,
        pro_model=config.pro_model,
        con_model=config.con_model,
        rounds=rounds,
    )
