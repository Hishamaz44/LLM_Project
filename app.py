"""Streamlit dashboard for the debate-bias experiment.

A read-only view over a completed results.jsonl — pure presentation of the metrics in
`debate_system/analysis.py`. It never calls a model or touches the cache, so what you see
is exactly the logged experiment (deterministic, offline, no API key needed).

Run:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from debate_system import analysis

st.set_page_config(page_title="Debate Bias Analysis", layout="wide")
st.title("Multi-Agent Debate — Bias Analysis")

RESULTS_DIR = Path("results")
DEFAULT_RESULTS = "results/run.jsonl"


# Cached on (path, mtime) so edits to the file invalidate the cache automatically.
@st.cache_data
def load(path: str, mtime: float) -> pd.DataFrame:
    return analysis.load_results(path)


# --- Data source -----------------------------------------------------------------
available = sorted(str(p) for p in RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
if not available:
    st.error(
        f"No results found in `{RESULTS_DIR}/`. Run an experiment first, e.g.\n\n"
        "```\npython run_experiment.py --limit 1 --out results/smoke.jsonl\n```"
    )
    st.stop()

default_index = available.index(DEFAULT_RESULTS) if DEFAULT_RESULTS in available else 0
choice = st.sidebar.selectbox("Results file", available, index=default_index)

try:
    df_all = load(choice, Path(choice).stat().st_mtime)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load `{choice}`: {exc}")
    st.stop()

# --- Filters (topic/round only, so order/length/panel pairings stay intact) --------
topics = sorted(df_all["topic_id"].unique())
rounds = sorted(int(r) for r in df_all["round_num"].unique())
sel_topics = st.sidebar.multiselect("Topics", topics, default=topics)
sel_rounds = st.sidebar.multiselect("Rounds", rounds, default=rounds)

df = df_all[df_all["topic_id"].isin(sel_topics) & df_all["round_num"].isin(sel_rounds)]
if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

models = ", ".join(sorted(set(df["pro_model"]) | set(df["con_model"])))
st.caption(f"Debater models: {models}")

# --- Headline summary --------------------------------------------------------------
s = analysis.summary(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Topic × rounds", s["topic_rounds"])
c2.metric("Panel evaluations", s["total_evaluations"])
c3.metric("Order bias (first − second)", f"{s['order_bias_first_minus_second']:+.2f}")
c4.metric("Winner flip rate", f"{s['winner_flip_rate']:.0%}")

# --- 1. Order effect ---------------------------------------------------------------
st.header("1. Order effect")
st.caption(
    "If judges were unbiased, the argument shown first should score the same as the one "
    "shown second, and the winner shouldn't flip just because we swap which side is first."
)
oe = analysis.order_effect(df)
st.write(
    f"Average (first − second) total score: **{oe['first_minus_second']:+.2f}**  "
    "— 0 = no order bias; positive = judges favor whichever argument is shown first."
)
st.dataframe(oe["by_order"], use_container_width=True)

# --- 2. Length effect --------------------------------------------------------------
st.header("2. Length effect")
st.caption(
    "Each pair is length-matched (both sides written to the same word budget), so verbosity "
    "bias shows up as the overall score climbing as the arguments get longer — not as a "
    "PRO/CON gap. Does the mean score rise from the shortest target to the longest?"
)
le = analysis.length_effect(df)
lc1, lc2, lc3 = st.columns(3)
lc1.metric("corr(length, score)", f"{le['corr_length_score']:+.2f}")
lc2.metric("score climb (longest − shortest)", f"{le['score_climb']:+.2f}")
lc3.metric("mean words off target", f"{le['mean_target_miss']:.1f}")

if len(le["by_length"]) > 1:
    st.caption("Mean score (both debaters) at each target length — a rising line = length bias.")
    st.bar_chart(le["by_length"])

# --- 3. Panel diversity ------------------------------------------------------------
st.header("3. Does a diverse judge panel reduce bias?")
st.caption(
    "Order-flip rate and length bias (score climb with length), broken down by a "
    "homogeneous panel (1 model ×3) vs a heterogeneous one (3 different models)."
)
pc = analysis.panel_comparison(df)
st.dataframe(pc, use_container_width=True)
if "flip_rate" in pc.columns:
    st.bar_chart(pc[["flip_rate"]])

# --- 4. Drill into a single evaluation ---------------------------------------------
st.header("4. Drill into a single evaluation")
st.caption("Inspect the individual judges' raw scores for one condition to see where they disagreed.")

d1, d2, d3, d4, d5 = st.columns(5)
d_topic = d1.selectbox("Topic", sorted(df["topic_id"].unique()))
d_round = d2.selectbox("Round", sorted(int(r) for r in df["round_num"].unique()))
d_length = d3.selectbox("Length target", sorted(int(t) for t in df["length_target"].unique()))
d_order = d4.selectbox("Order", sorted(df["order"].unique()))
d_panel = d5.selectbox("Panel", sorted(df["panel_type"].unique()))

match = df[
    (df["topic_id"] == d_topic)
    & (df["round_num"] == d_round)
    & (df["length_target"] == d_length)
    & (df["order"] == d_order)
    & (df["panel_type"] == d_panel)
]
if match.empty:
    st.info("No row for that combination of conditions.")
else:
    row = match.iloc[0]
    st.write(
        f"PRO words: **{row['pro_word_count']}** · CON words: **{row['con_word_count']}** · "
        f"panel winner: **{row['panel_winner']}**"
    )
    pro_is_first = d_order == "pro_first"
    judges = pd.DataFrame(
        [
            {
                "judge": j["judge_model"],
                **{f"PRO {dim}": (j["debater1"] if pro_is_first else j["debater2"])[dim] for dim in analysis.DIMENSIONS},
                **{f"CON {dim}": (j["debater2"] if pro_is_first else j["debater1"])[dim] for dim in analysis.DIMENSIONS},
                "winner": j["winner"],
            }
            for j in row["judges_raw"]
        ]
    )
    st.dataframe(judges, use_container_width=True, hide_index=True)
