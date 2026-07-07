"""CLI entrypoint.

Usage:
    python run_experiment.py --config config.yaml --out results/run1.jsonl
    python run_experiment.py --limit 1   # smoke test: just the first topic
"""

import argparse

from dotenv import load_dotenv

from debate_system.config import load_config
from debate_system.experiment import run_experiment


# Parses CLI args, loads config, and kicks off the experiment run.
def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent debate experiment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/run.jsonl")
    parser.add_argument("--report", default=None, help="Markdown transcript path (default: <out> with .md extension)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N topics")
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Max concurrent API calls (default 8). Set to 1 for fully sequential; lower it if "
        "OpenRouter starts returning 429 rate-limit errors.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    run_experiment(config, args.out, limit=args.limit, report_path=args.report, jobs=args.jobs)
    print(f"Done. Results saved to {args.out}")


if __name__ == "__main__":
    main()
