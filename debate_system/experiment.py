"""Orchestrates the full experiment: generate debates, run all judge conditions, log results."""

import json
from pathlib import Path

from .config import Config
from .debate import generate_debate
from .evaluate import run_all_conditions
from .report import MarkdownReport
from .topics import TOPICS


# Runs every topic through debate generation + all judge conditions, writing
# results as JSONL and a parallel Markdown transcript as it goes.
def run_experiment(
    config: Config,
    out_path: str | Path,
    limit: int | None = None,
    report_path: str | Path | None = None,
    jobs: int = 1,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path) if report_path else out_path.with_suffix(".md")

    topics = TOPICS[:limit] if limit else TOPICS

    with out_path.open("w") as f, MarkdownReport(report_path) as report:
        for i, topic_text in enumerate(topics):
            topic_id = f"t{i + 1:02d}"
            debate = generate_debate(topic_id, topic_text, config, jobs=jobs)
            report.log_debate(topic_id, topic_text, debate)
            for round_content in debate.rounds:
                records = run_all_conditions(topic_id, topic_text, round_content, config, jobs=jobs)
                report.log_round_result(round_content.round_num, records)
                for record in records:
                    f.write(json.dumps(record) + "\n")
                f.flush()
        report.log_summary(f"Done. Results saved to `{out_path}`.")
