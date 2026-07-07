"""Generates the debate transcript: independent openings, then rebuttals."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .client import call_model
from .config import Config


# One round's For/Against text at each target length. `for_debater_texts`/`against_debater_texts`
# map a target word count (e.g. 60/120/180) to the argument the debater wrote to that budget, so
# a round holds a full set of length-controlled variants rather than a single natural-length text.
@dataclass
class RoundContent:
    round_num: int
    for_debater_texts: dict[int, str]
    against_debater_texts: dict[int, str]


# Full transcript of a debate on one topic: which models argued, and every round's content.
@dataclass
class Debate:
    topic_id: str
    topic_text: str
    for_debater_model: str
    against_debater_model: str
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


# Builds the prompt that expands an existing argument to a longer target WITHOUT adding content.
# Used only in length_mode="pad": the shortest target is generated normally, then padded up to
# each longer target so content stays fixed and only word count changes — which isolates verbosity
# bias from the "longer arguments legitimately say more" confound.
def _pad_prompt(topic: str, stance: str, argument: str, target_words: int) -> str:
    return (
        f'Debate topic: "{topic}"\n\n'
        f'You previously argued the {stance} position with this text:\n"{argument}"\n\n'
        f"Rewrite it to about {target_words} words (aim for that length) while making the exact "
        "same case. Do NOT add any new argument, reason, example, or piece of evidence — only "
        "expand what is already there by elaborating, restating, and adding connective phrasing. "
        "Do not mention these instructions."
    )


# Generates the openings (and rebuttals, if 2 rounds) for one target length. Returns the
# round-1 and round-2 For/Against texts; round-2 texts are empty when config.rounds < 2.
def _generate_at_length(topic_text: str, target: int, config: Config) -> tuple[str, str, str, str]:
    for_opening = call_model(
        config.for_debater_model,
        _opening_prompt(topic_text, "FOR", target),
        max_tokens=config.debater_max_tokens,
        temperature=config.debater_temperature,
    )
    against_opening = call_model(
        config.against_debater_model,
        _opening_prompt(topic_text, "AGAINST", target),
        max_tokens=config.debater_max_tokens,
        temperature=config.debater_temperature,
    )
    for_rebuttal = against_rebuttal = ""
    if config.rounds >= 2:
        for_rebuttal = call_model(
            config.for_debater_model,
            _rebuttal_prompt(topic_text, "FOR", for_opening, against_opening, target),
            max_tokens=config.debater_max_tokens,
            temperature=config.debater_temperature,
        )
        against_rebuttal = call_model(
            config.against_debater_model,
            _rebuttal_prompt(topic_text, "AGAINST", against_opening, for_opening, target),
            max_tokens=config.debater_max_tokens,
            temperature=config.debater_temperature,
        )
    return for_opening, against_opening, for_rebuttal, against_rebuttal


# Expands one debater's text to a longer target via the padding prompt, using that side's model.
# Empty text (e.g. an absent rebuttal in a 1-round debate) passes through untouched.
def _pad_text(topic_text: str, stance: str, argument: str, target: int, config: Config) -> str:
    if not argument:
        return ""
    model = config.for_debater_model if stance == "FOR" else config.against_debater_model
    return call_model(
        model,
        _pad_prompt(topic_text, stance, argument, target),
        max_tokens=config.debater_max_tokens,
        temperature=config.debater_temperature,
    )


# Returns (for_opening, against_opening, for_rebuttal, against_rebuttal) for one target length.
# In "generate" mode each target is an independent fresh generation (content grows with length).
# In "pad" mode the shortest target's base texts are expanded to this target, so content is held
# fixed and only length changes; the base target itself returns its texts unchanged.
def _texts_for_target(
    topic_text: str, target: int, base_texts: tuple[str, str, str, str] | None, config: Config
) -> tuple[str, str, str, str]:
    if config.length_mode != "pad":
        return _generate_at_length(topic_text, target, config)
    for_open, against_open, for_reb, against_reb = base_texts
    if target == min(config.length_targets):
        return for_open, against_open, for_reb, against_reb
    return (
        _pad_text(topic_text, "FOR", for_open, target, config),
        _pad_text(topic_text, "AGAINST", against_open, target, config),
        _pad_text(topic_text, "FOR", for_reb, target, config),
        _pad_text(topic_text, "AGAINST", against_reb, target, config),
    )


# Runs the full debate for one topic at every configured target length. Each length is an
# independent, length-matched mini-debate (a rebuttal at length L answers the opening at L),
# so every judged pair is coherent and matched. In pad mode the shortest length is generated
# once and expanded to the longer targets, so content is fixed and only word count varies.
def generate_debate(topic_id: str, topic_text: str, config: Config, jobs: int = 1) -> Debate:
    # Pad mode writes the shortest-length pair once; every longer target is that same content
    # expanded (see _texts_for_target), holding content fixed while length varies. This base must
    # exist before the per-target work below, so it's computed up front (sequentially).
    base_texts = (
        _generate_at_length(topic_text, min(config.length_targets), config)
        if config.length_mode == "pad"
        else None
    )

    # Each target's texts are independent once `base_texts` exists, so fan them out across `jobs`
    # workers. Results are keyed by target and reassembled in config order below, so output is
    # identical to the sequential path regardless of which target finishes first.
    targets = list(config.length_targets)

    def texts_for(target: int) -> tuple[str, str, str, str]:
        return _texts_for_target(topic_text, target, base_texts, config)

    if jobs > 1 and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            by_target = dict(zip(targets, pool.map(texts_for, targets)))
    else:
        by_target = {target: texts_for(target) for target in targets}

    round1_for, round1_against, round2_for, round2_against = {}, {}, {}, {}
    for target in targets:
        for_open, against_open, for_reb, against_reb = by_target[target]
        round1_for[target], round1_against[target] = for_open, against_open
        if config.rounds >= 2:
            round2_for[target], round2_against[target] = for_reb, against_reb

    rounds = [RoundContent(round_num=1, for_debater_texts=round1_for, against_debater_texts=round1_against)]
    if config.rounds >= 2:
        rounds.append(RoundContent(round_num=2, for_debater_texts=round2_for, against_debater_texts=round2_against))

    return Debate(
        topic_id=topic_id,
        topic_text=topic_text,
        for_debater_model=config.for_debater_model,
        against_debater_model=config.against_debater_model,
        rounds=rounds,
    )
