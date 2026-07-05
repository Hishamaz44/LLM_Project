"""Renders a running Markdown transcript of an experiment (debate text + scores),
replacing ad-hoc console prints so experiment.py stays focused on orchestration."""

from pathlib import Path

from .debate import Debate

_ROUND_LABELS = {1: "Round 1 — Opening statements", 2: "Round 2 — Rebuttals"}
_WINNER_LABELS = {
    "pro_first": {"Debater 1": "PRO", "Debater 2": "CON", "Tie": "Tie"},
    "con_first": {"Debater 1": "CON", "Debater 2": "PRO", "Tie": "Tie"},
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

    # Writes the topic heading and each round's PRO/CON text at every target length.
    def log_debate(self, topic_id: str, topic_text: str, debate: Debate) -> None:
        self._write(f"## {topic_id} — {topic_text}\n\n")
        for r in debate.rounds:
            label = _ROUND_LABELS.get(r.round_num, f"Round {r.round_num}")
            self._write(f"### {label}\n\n")
            for target in sorted(r.pro_texts):
                self._write(
                    f"**~{target} words**\n\n"
                    f"**PRO**\n\n{r.pro_texts[target]}\n\n"
                    f"**CON**\n\n{r.con_texts[target]}\n\n"
                )

    # Writes a score-by-length table (does the score climb with length?) plus each individual
    # judge's raw scores for the longest-length pair, using the heterogeneous, pro_first rows.
    def log_round_result(self, round_num: int, records: list[dict]) -> None:
        view = sorted(
            (r for r in records
             if r["order"] == "pro_first" and r["panel_type"] == "heterogeneous"),
            key=lambda r: r["length_target"],
        ) or records
        total = lambda d: d["logic"] + d["evidence"] + d["fairness"]

        self._write(f"**Scores by length (round {round_num}, heterogeneous panel, pro_first)**\n\n")
        self._write("| Target words | Actual PRO/CON | PRO total | CON total | Winner |\n")
        self._write("|---|---|---|---|---|\n")
        for r in view:
            self._write(
                f"| ~{r['length_target']} | {r['pro_word_count']}/{r['con_word_count']} "
                f"| {total(r['pro_avg']):.1f} | {total(r['con_avg']):.1f} | {r['panel_winner'].upper()} |\n"
            )
        self._write("\n")

        # Detailed per-judge scores for the longest-length pair (the most complete arguments).
        canonical = view[-1]

        self._write(f"Individual judge scores (longest pair, ~{canonical['length_target']} words):\n\n")
        self._write(
            "| Judge | PRO logic | PRO evidence | PRO fairness | CON logic | CON evidence | CON fairness | Winner |\n"
        )
        self._write("|---|---|---|---|---|---|---|---|\n")
        pro_first = canonical["order"] == "pro_first"
        winner_labels = _WINNER_LABELS[canonical["order"]]
        for j in canonical["judges_raw"]:
            pro_scores, con_scores = (j["debater1"], j["debater2"]) if pro_first else (j["debater2"], j["debater1"])
            self._write(
                f"| {j['judge_model']} | {pro_scores['logic']:.1f} | {pro_scores['evidence']:.1f} | "
                f"{pro_scores['fairness']:.1f} | {con_scores['logic']:.1f} | {con_scores['evidence']:.1f} | "
                f"{con_scores['fairness']:.1f} | {winner_labels[j['winner']]} |\n"
            )
        self._write("\n")

    # Writes a closing message (e.g. a "done" note) at the end of the report.
    def log_summary(self, message: str) -> None:
        self._write(f"---\n\n{message}\n")

    # Internal helper: writes to the file and flushes so progress is visible mid-run.
    def _write(self, text: str) -> None:
        self._file.write(text)
        self._file.flush()
