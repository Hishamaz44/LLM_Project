"""Bias metrics over a completed experiment's results.jsonl.

Pure functions of the already-logged records — no model or cache access — so the numbers
are deterministic, reproducible, and computable offline (no API key). This is the single
source of truth for the order/length/panel metrics; the Streamlit dashboard (`app.py`)
and the notebook both build on it instead of redefining the formulas.
"""

import json
from pathlib import Path

import pandas as pd

DIMENSIONS = ("logic", "evidence", "fairness")


# Loads results.jsonl into a DataFrame and adds the derived columns the metrics use:
# per-debater score totals, a per-row mean score (both debaters), the For/Against word/score
# edges, and how far the generated arguments landed from their length target.
def load_results(path: str | Path) -> pd.DataFrame:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    df = pd.json_normalize(records)
    if "length_target" not in df.columns and "length_mode" in df.columns:
        raise ValueError(
            f"{path} uses the retired length_mode schema (natural/equalized). It predates "
            "the length-target redesign — regenerate it with `python run_experiment.py`."
        )
    if "pro_avg.logic" in df.columns or "pro_word_count" in df.columns:
        raise ValueError(
            f"{path} uses the retired pro/con schema. It predates the For/Against-debater "
            "rename — regenerate it with `python run_experiment.py` (or migrate the keys)."
        )
    for prefix in ("first_shown_avg", "second_shown_avg", "for_debater_avg", "against_debater_avg"):
        df[f"{prefix}_total"] = sum(df[f"{prefix}.{dim}"] for dim in DIMENSIONS)
    # Since each judged pair is length-matched, length bias moves the *overall* score level,
    # so we track the mean of both debaters' totals per row.
    df["mean_score"] = (df["for_debater_avg_total"] + df["against_debater_avg_total"]) / 2
    df["word_count_diff"] = df["for_debater_word_count"] - df["against_debater_word_count"]
    df["score_diff"] = df["for_debater_avg_total"] - df["against_debater_avg_total"]
    # How far the two arguments landed from the length budget they were asked to hit — a
    # residual that quantifies how well the debater models obeyed the target.
    if "length_target" in df.columns:
        df["target_miss"] = (
            (df["for_debater_word_count"] - df["length_target"]).abs()
            + (df["against_debater_word_count"] - df["length_target"]).abs()
        ) / 2
    return df


# Order (position) bias: how much the first-shown argument is favored, and the per-slot
# average totals. 0 = no bias; positive = judges favor whichever argument is shown first.
def order_effect(df: pd.DataFrame) -> dict:
    first_minus_second = (df["first_shown_avg_total"] - df["second_shown_avg_total"]).mean()
    by_order = df.groupby("order")[["first_shown_avg_total", "second_shown_avg_total"]].mean()
    return {"first_minus_second": float(first_minus_second), "by_order": by_order}


# Per (topic, round, length, panel), does the declared winner change when the presentation
# order is swapped? Returns the pivot with a boolean `flipped` column.
def winner_flips(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["topic_id", "round_num", "length_target", "panel_type"],
        columns="order",
        values="panel_winner",
        aggfunc="first",
    )
    if "for_first" in pivot.columns and "against_first" in pivot.columns:
        pivot["flipped"] = pivot["for_first"] != pivot["against_first"]
    else:
        # Only one order present (e.g. filtered) — nothing can flip.
        pivot["flipped"] = False
    return pivot


# Fraction of (topic, round, length, panel) cells whose winner flips on order swap.
def flip_rate(df: pd.DataFrame) -> float:
    flipped = winner_flips(df)["flipped"]
    return float(flipped.mean()) if len(flipped) else float("nan")


# Length (verbosity) bias. Each judged pair is length-matched, so bias shows up as the
# overall score level rising with the shared argument length — not as a For/Against gap.
# Reports the mean score at each length target, the climb from the shortest target to the
# longest, the length-vs-score correlation, and how far arguments missed their target.
def length_effect(df: pd.DataFrame) -> dict:
    if "length_target" not in df.columns or df.empty:
        return {
            "by_length": pd.Series(dtype=float),
            "score_climb": float("nan"),
            "corr_length_score": float("nan"),
            "mean_target_miss": float("nan"),
        }
    by_length = df.groupby("length_target")["mean_score"].mean()
    corr = (
        df["length_target"].corr(df["mean_score"])
        if df["length_target"].nunique() > 1
        else float("nan")
    )
    climb = float(by_length.iloc[-1] - by_length.iloc[0]) if len(by_length) > 1 else float("nan")
    mean_miss = float(df["target_miss"].mean()) if "target_miss" in df.columns else float("nan")
    return {
        "by_length": by_length,
        "score_climb": climb,
        "corr_length_score": float(corr),
        "mean_target_miss": mean_miss,
    }


# Compares the order-flip rate and the length bias (score climb / correlation with length)
# between the homogeneous and heterogeneous panels, indexed by panel_type.
def panel_comparison(df: pd.DataFrame) -> pd.DataFrame:
    flips = winner_flips(df).reset_index()
    rows = []
    for panel in sorted(df["panel_type"].unique()):
        panel_df = df[df["panel_type"] == panel]
        panel_flips = flips.loc[flips["panel_type"] == panel, "flipped"]
        le = length_effect(panel_df)
        rows.append(
            {
                "panel_type": panel,
                "flip_rate": float(panel_flips.mean()) if len(panel_flips) else float("nan"),
                "length_score_climb": le["score_climb"],
                "corr_length_score": le["corr_length_score"],
            }
        )
    return pd.DataFrame(rows).set_index("panel_type")


# The headline numbers used in the writeup/slides.
def summary(df: pd.DataFrame) -> dict:
    le = length_effect(df)
    return {
        "topic_rounds": int(df[["topic_id", "round_num"]].drop_duplicates().shape[0]),
        "total_evaluations": int(len(df)),
        "order_bias_first_minus_second": order_effect(df)["first_minus_second"],
        "winner_flip_rate": flip_rate(df),
        "length_bias_climb": le["score_climb"],
        "length_bias_corr": le["corr_length_score"],
    }
