"""Renders a running Markdown transcript of an experiment (debate text + scores),
replacing ad-hoc console prints so experiment.py stays focused on orchestration."""

from pathlib import Path

from .debate import Debate

_ROUND_LABELS = {1: "Round 1 — Opening statements", 2: "Round 2 — Rebuttals"}
_WINNER_LABELS = {
    "for_first": {"Debater 1": "For", "Debater 2": "Against", "Tie": "Tie"},
    "against_first": {"Debater 1": "Against", "Debater 2": "For", "Tie": "Tie"},
}


class MarkdownReport:
    """Append-only Markdown writer. Use as a context manager so the file is
    always closed, even if the experiment raises partway through."""

    # Constructor: opens the output file for writing and ensures its parent dir exists.
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w")

    # Context manager entry: just returns self, no setup needed beyond __init__.
    def __enter__(self) -> "MarkdownReport":
        return self

    # Context manager exit: closes the file regardless of how the block ended.
    def __exit__(self, *exc_info) -> None:
        self._file.close()

    # Writes the topic heading and each round's For/Against text at every target length.
    def log_debate(self, topic_id: str, topic_text: str, debate: Debate) -> None:
        self._write(f"## {topic_id} — {topic_text}\n\n")
        for r in debate.rounds:
            label = _ROUND_LABELS.get(r.round_num, f"Round {r.round_num}")
            self._write(f"### {label}\n\n")
            for target in sorted(r.for_debater_texts):
                self._write(
                    f"**~{target} words**\n\n"
                    f"**For Debater**\n\n{r.for_debater_texts[target]}\n\n"
                    f"**Against Debater**\n\n{r.against_debater_texts[target]}\n\n"
                )

    # Writes a score-by-length table (does the score climb with length?) plus each individual
    # judge's raw scores for the longest-length pair, using the heterogeneous, for_first rows.
    def log_round_result(self, round_num: int, records: list[dict]) -> None:
        view = sorted(
            (r for r in records
             if r["order"] == "for_first" and r["panel_type"] == "heterogeneous"),
            key=lambda r: r["length_target"],
        ) or records
        total = lambda d: d["logic"] + d["evidence"] + d["fairness"]

        self._write(f"**Scores by length (round {round_num}, heterogeneous panel, for_first)**\n\n")
        self._write("| Target words | Actual For/Against | For total | Against total | Winner |\n")
        self._write("|---|---|---|---|---|\n")
        for r in view:
            self._write(
                f"| ~{r['length_target']} | {r['for_debater_word_count']}/{r['against_debater_word_count']} "
                f"| {total(r['for_debater_avg']):.1f} | {total(r['against_debater_avg']):.1f} | {r['panel_winner'].capitalize()} |\n"
            )
        self._write("\n")

        # Detailed per-judge scores for the longest-length pair (the most complete arguments).
        canonical = view[-1]

        self._write(f"Individual judge scores (longest pair, ~{canonical['length_target']} words):\n\n")
        self._write(
            "| Judge | For logic | For evidence | For fairness | Against logic | Against evidence | Against fairness | Winner |\n"
        )
        self._write("|---|---|---|---|---|---|---|---|\n")
        for_first = canonical["order"] == "for_first"
        winner_labels = _WINNER_LABELS[canonical["order"]]
        for j in canonical["judges_raw"]:
            for_scores, against_scores = (j["debater1"], j["debater2"]) if for_first else (j["debater2"], j["debater1"])
            self._write(
                f"| {j['judge_model']} | {for_scores['logic']:.1f} | {for_scores['evidence']:.1f} | "
                f"{for_scores['fairness']:.1f} | {against_scores['logic']:.1f} | {against_scores['evidence']:.1f} | "
                f"{against_scores['fairness']:.1f} | {winner_labels[j['winner']]} |\n"
            )
        self._write("\n")

    # Writes a closing message (e.g. a "done" note) at the end of the report.
    def log_summary(self, message: str) -> None:
        self._write(f"---\n\n{message}\n")

    # Internal helper: writes to the file and flushes so progress is visible mid-run.
    def _write(self, text: str) -> None:
        self._file.write(text)
        self._file.flush()
