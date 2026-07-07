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

models = ", ".join(sorted(set(df["for_debater_model"]) | set(df["against_debater_model"])))
st.caption(f"Debater models: {models}")

# --- Headline summary --------------------------------------------------------------
s = analysis.summary(df)
st.caption("At-a-glance headline numbers — each one is explained in full in its section below.")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Debates analyzed",
    s["topic_rounds"],
    help="Distinct topic-and-round pairs. Each is judged under every length × order × panel condition.",
)
c2.metric(
    "Judge evaluations",
    s["total_evaluations"],
    help="Total panel judgments logged — one per length × order × panel × debate.",
)
c3.metric(
    "First-position advantage",
    f"{s['order_bias_first_minus_second']:+.2f}",
    help="Average points the argument shown FIRST scores above the one shown SECOND (out of 30). "
    "0 = no order bias; positive = being shown first helps. Detailed in section 1.",
)
c4.metric(
    "Winner-flip rate",
    f"{s['winner_flip_rate']:.0%}",
    help="How often the panel's winner changes just because we swap which argument is shown first. "
    "0% = order never decides the winner. Detailed in section 1.",
)

# --- 1. Order effect ---------------------------------------------------------------
st.header("1. Order effect — does being shown first change the score?")
st.markdown(
    "Every debate is judged **twice**: once with the FOR argument shown first, once with the "
    "AGAINST argument shown first. The arguments themselves are identical both times — only their "
    "**position** changes. So a fair panel should give an argument the same score whether it comes "
    "first or second, and should pick the same winner either way. Anything we see here is therefore "
    "caused by **position alone**, not by what the arguments actually say.\n\n"
    "Each argument is scored out of **30** — three criteria (logic, evidence, fairness) at 1–10 "
    "each, averaged across the three judges."
)

oe = analysis.order_effect(df)
o1, o2 = st.columns(2)
o1.metric(
    "First-position advantage",
    f"{oe['first_minus_second']:+.2f}",
    help="Average score of the first-shown argument minus the second-shown one (out of 30).",
)
o2.metric(
    "Winner-flip rate",
    f"{analysis.flip_rate(df):.0%}",
    help="Share of debates whose winner changes when we swap which side is shown first.",
)
st.caption(
    "**First-position advantage** near **0** means order didn't move the scores; a **positive** "
    "value means the first-shown argument was favored. **Winner-flip rate** near **0%** means the "
    "same side wins regardless of order (fair judges); a **high** rate means order alone can flip "
    "who wins — the strongest sign of order bias."
)

# Plain-language version of the per-ordering breakdown: rename the jargon index/columns and add
# an explicit "advantage" column so the first-vs-second gap is readable without decoding the labels.
by_order = oe["by_order"].rename(
    index={"for_first": "FOR shown first", "against_first": "AGAINST shown first"},
    columns={
        "first_shown_avg_total": "Score of 1st-shown argument",
        "second_shown_avg_total": "Score of 2nd-shown argument",
    },
)
by_order["Advantage to 1st slot (1st − 2nd)"] = (
    by_order["Score of 1st-shown argument"] - by_order["Score of 2nd-shown argument"]
)
by_order.index.name = "Ordering"
st.markdown("**First slot vs second slot, in each ordering** (scores out of 30):")
st.dataframe(by_order.round(2), use_container_width=True)
st.caption(
    "Each row is the same debate run in one ordering; **Advantage** is how much the first slot beat "
    "the second. Three patterns to look for:\n"
    "- **Positive in both rows** → the first slot wins whoever sits there = real **order bias**.\n"
    "- **Near 0 in both rows** → position didn't matter = judges are **order-fair**.\n"
    "- **Opposite signs** (one +, one −) → *not* order bias but a **side effect**: one side scores "
    "higher wherever it's placed, so the two rows cancel out in the small headline number above."
)

# --- 2. Length effect --------------------------------------------------------------
st.header("2. Length effect — do longer arguments score higher?")
st.markdown(
    "In each pair both sides are written to the **same word budget** (60 vs 60, 120 vs 120, "
    "180 vs 180), so length can never help one side beat the other. Instead, a preference for "
    "longer writing shows up as the **overall score level rising** as the shared length grows. "
    "If the judges only cared about substance, the average score would stay flat across the "
    "three lengths.\n\n"
    "One honest caveat: a longer argument can genuinely *say more*, so a rising line isn't proof "
    "of bias on its own — that's why we look for a steady climb across all three lengths, and why "
    "the **padding run** (same content, only the length changes) is the cleaner test."
)
le = analysis.length_effect(df)
lc1, lc2, lc3 = st.columns(3)
lc1.metric(
    "Score gain, shortest → longest",
    f"{le['score_climb']:+.2f}",
    help="Average score at the longest length minus the average at the shortest (out of 30). "
    "Positive = longer arguments scored higher. This is the headline length-bias number.",
)
lc2.metric(
    "Length–score correlation",
    f"{le['corr_length_score']:+.2f}",
    help="How consistently score tracks length, from −1 to +1. Near +1 = score rises at every "
    "length step (a steady bias); near 0 = length and score are unrelated; negative = longer "
    "scored lower.",
)
lc3.metric(
    "Avg. words off the target",
    f"{le['mean_target_miss']:.1f}",
    help="On average, how far the arguments landed from the word budget they were told to hit — "
    "a sanity check that the length control worked. Small = the models obeyed the targets.",
)

if len(le["by_length"]) > 1:
    st.markdown("**Average score at each argument length** (out of 30) — a rising line means length bias:")
    length_chart = le["by_length"].rename("Average score (out of 30)")
    length_chart.index.name = "Argument length (words)"
    st.bar_chart(length_chart)

# --- 3. Panel diversity ------------------------------------------------------------
st.header("3. Panel diversity — does using different judge models cancel out bias?")
st.markdown(
    "This is the core question of the project. Three **identical** judges tend to share the same "
    "blind spots, so their biases reinforce each other. Three **different** judges may each be "
    "biased in their own way, so averaging them could partly cancel the bias out. To test that, "
    "we measure the **same two biases from sections 1 and 2** separately for each kind of panel."
)
st.markdown(
    "**How to read the three columns:**\n"
    "- **Order bias (winner-flip rate):** how often the panel's winner changes just because we "
    "swap which argument is shown first (section 1). It is the share of debates where order alone "
    "decided the outcome. **Lower is better** — 0% means order never changed the winner.\n"
    "- **Length bias (score gain by length):** how many points (out of 30) the average score rises "
    "from the shortest argument to the longest (section 2). **Closer to 0 is better** — a large "
    "positive number means longer writing was rewarded for its own sake.\n"
    "- **Length–score correlation:** how *consistently* score tracks length, on a −1 to +1 scale. "
    "**Closer to 0 is better** — near +1 means the score climbs at every length step (a steady, "
    "reliable bias rather than noise)."
)

pc = analysis.panel_comparison(df)
panel_labels = {
    "homogeneous": "Homogeneous panel (same model ×3)",
    "heterogeneous": "Heterogeneous panel (3 different models)",
}
# Build the columns as plain lists (not Series) so the renamed index below isn't reindexed
# against the original panel_type labels — passing a mismatched index= to Series values NaNs them.
pc_display = pd.DataFrame(
    {
        "Order bias (winner-flip rate)": [
            f"{x:.0%}" if pd.notna(x) else "—" for x in pc["flip_rate"]
        ],
        "Length bias (score gain by length)": [
            f"{x:+.2f}" if pd.notna(x) else "—" for x in pc["length_score_climb"]
        ],
        "Length–score correlation": [
            f"{x:+.2f}" if pd.notna(x) else "—" for x in pc["corr_length_score"]
        ],
    },
    index=[panel_labels.get(p, p) for p in pc.index],
)
pc_display.index.name = "Judge panel"
st.dataframe(pc_display, use_container_width=True)
st.caption(
    "**A diverse panel helps** if the heterogeneous row shows a lower winner-flip rate, a score "
    "gain closer to 0, and a correlation closer to 0 than the homogeneous row. If the two rows are "
    "about the same, mixing judge models didn't reduce bias in this run."
)

if "flip_rate" in pc.columns:
    st.markdown("**Order bias by panel** (winner-flip rate — a shorter bar means less order bias):")
    flip_chart = pd.DataFrame(
        {"Winner-flip rate": pc["flip_rate"].values},
        index=[panel_labels.get(p, p) for p in pc.index],
    )
    flip_chart.index.name = "Judge panel"
    st.bar_chart(flip_chart)

# --- 4. Nepotism (self-preference) bias --------------------------------------------
st.header("4. Nepotism — do judges favour arguments written by their own model?")
st.markdown(
    "Some models take a turn as **both** a debater and a judge. Whenever a judge is grading an "
    "argument written by **its own model**, we can ask a pointed question: does it score that "
    "argument higher than the **other** judges — different models — score the *exact same text*? "
    "Because the peer judges read the identical argument, any gap can't be blamed on the argument "
    "being better written — it's the model **preferring its own work**.\n\n"
    "This can only be measured when a model appears on **both** sides: as an author *and* as a judge "
    "on a panel that also contains other models (the heterogeneous panel). Models that only judge, "
    "or only debate, don't show up here."
)

sp = analysis.self_preference_effect(df)
if sp["n_comparisons"] == 0:
    st.info(
        "No self-preference comparisons in the current selection. This metric needs a judge whose "
        "model also authored an argument, together with at least one different-model judge grading "
        "the same text — i.e. a heterogeneous panel where a judge model matches a debater model."
    )
else:
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric(
        "Self-preference Δ",
        f"{sp['delta']:+.2f}",
        help="Average points a judge gives its own model's argument above what peer judges give the "
        "same text (out of 30). 0 = even-handed; positive = the model favours its own writing.",
    )
    sp2.metric(
        "Scored own model higher",
        f"{sp['scored_self_higher_rate']:.0%}",
        help="Share of self-vs-peer comparisons where the model rated its own argument above the "
        "peer average. 50% = no consistent lean; higher = it usually favours itself.",
    )
    sp3.metric(
        "Comparisons",
        sp["n_comparisons"],
        help="How many self-vs-peer comparisons back these numbers. Small n = read them as a "
        "proof-of-concept, not a firm finding.",
    )
    st.markdown("**Per model** — only models that both authored *and* judged appear (score out of 30):")
    sp_display = pd.DataFrame(
        {
            "Self-preference Δ (out of 30)": [f"{x:+.2f}" for x in sp["by_model"]["self_preference_delta"]],
            "Scored own model higher": [f"{x:.0%}" for x in sp["by_model"]["scored_self_higher_rate"]],
            "Comparisons": [int(x) for x in sp["by_model"]["n"]],
        },
        index=sp["by_model"].index,
    )
    sp_display.index.name = "Model (author & judge)"
    st.dataframe(sp_display, use_container_width=True)
    st.caption(
        "A **positive Δ** together with a **higher-than-50%** rate means the model consistently "
        "inflates its own arguments relative to how rival judges score the identical text — the "
        "signature of nepotism. Values near **0 / ~50%** mean the model judges its own work "
        "even-handedly. Remember peers grade the *same* argument, so quality is already controlled for."
    )

# --- 5. Drill into a single evaluation ---------------------------------------------
st.header("5. Inspect one evaluation — where did the judges (dis)agree?")
st.markdown(
    "Pick one exact condition below to see the **raw score each of the three judges gave**, before "
    "any averaging. Every judge scores both arguments from 1–10 on logic, evidence, and fairness, "
    "then names a winner — so this is where you can see whether the panel agreed or split."
)

d1, d2, d3, d4, d5 = st.columns(5)
d_topic = d1.selectbox("Topic", sorted(df["topic_id"].unique()))
d_round = d2.selectbox("Round", sorted(int(r) for r in df["round_num"].unique()))
d_length = d3.selectbox(
    "Argument length",
    sorted(int(t) for t in df["length_target"].unique()),
    format_func=lambda t: f"{t} words",
)
d_order = d4.selectbox(
    "Presentation order",
    sorted(df["order"].unique()),
    format_func=lambda o: {"for_first": "FOR shown first", "against_first": "AGAINST shown first"}.get(o, o),
)
d_panel = d5.selectbox(
    "Judge panel",
    sorted(df["panel_type"].unique()),
    format_func=lambda p: {"homogeneous": "Homogeneous (same model ×3)", "heterogeneous": "Heterogeneous (3 models)"}.get(p, p),
)

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
        f"FOR argument: **{row['for_debater_word_count']} words** · "
        f"AGAINST argument: **{row['against_debater_word_count']} words** · "
        f"panel winner: **{row['panel_winner'].capitalize()}**"
    )
    for_is_first = d_order == "for_first"
    # The judge's raw verdict is stance-blind (Debater 1/2 = shown first/second). Translate it
    # to a stance for display so the winner column matches the For/Against score columns.
    winner_label = (
        {"Debater 1": "For", "Debater 2": "Against", "Tie": "Tie"} if for_is_first
        else {"Debater 1": "Against", "Debater 2": "For", "Tie": "Tie"}
    )
    judges = pd.DataFrame(
        [
            {
                "Judge model": j["judge_model"],
                **{f"FOR: {dim}": (j["debater1"] if for_is_first else j["debater2"])[dim] for dim in analysis.DIMENSIONS},
                **{f"AGAINST: {dim}": (j["debater2"] if for_is_first else j["debater1"])[dim] for dim in analysis.DIMENSIONS},
                "Winner": winner_label[j["winner"]],
            }
            for j in row["judges_raw"]
        ]
    )
    st.dataframe(judges, use_container_width=True, hide_index=True)

# --- Cross-run comparison (loads every tier run, ignores the sidebar file selector) ------
SIZE_ORDER = list(analysis.TIER_SIZES)


# Cached on the tier files' mtimes so re-running an experiment refreshes the comparison.
@st.cache_data
def load_comparison(key: tuple) -> pd.DataFrame:
    return analysis.compare_runs(RESULTS_DIR)


cmp_key = tuple(f"{p}:{p.stat().st_mtime}" for p in sorted(RESULTS_DIR.glob("*.jsonl")))
cmp_df = load_comparison(cmp_key)

# --- 6. Compare across model sizes -------------------------------------------------
st.header("6. Compare across model sizes — all runs at once")
st.markdown(
    "Everything above analyzes the **one** file picked in the sidebar. This section and the next "
    "instead load **every** tier run in `results/` — small, medium, big, in both generate and pad "
    "mode — and line their headline bias metrics up side by side. The core question of the size "
    "sweep: **does a bigger, more capable judge model resist these biases better?** Read each chart "
    "left→right across the sizes; the two bars per size are generate vs pad mode."
)

if cmp_df.empty:
    st.info(
        "No tier runs found in `results/`. Expected files like `small.jsonl`, `medium_pad.jsonl`, "
        "`big.jsonl` — run the `config_*.yaml` tier configs first."
    )
else:
    st.caption("Runs loaded: " + ", ".join(f"{r.size}/{r.mode}" for r in cmp_df.itertuples()))
    table = cmp_df.rename(columns={"size": "Size", "mode": "Mode", **analysis.COMPARISON_METRICS})
    st.dataframe(table.set_index(["Size", "Mode"]).round(3), use_container_width=True)

    for col, label in analysis.COMPARISON_METRICS.items():
        pivot = (
            cmp_df.pivot(index="size", columns="mode", values=col)
            .reindex(SIZE_ORDER)
            .dropna(how="all")
        )
        st.markdown(f"**{label}** — by model size (bars: generate vs pad)")
        st.bar_chart(pivot, stack=False)

# --- 7. Generate vs Pad ------------------------------------------------------------
st.header("7. Generate vs Pad — how much length bias is *pure* verbosity?")
st.markdown(
    "Pad mode keeps an argument's content fixed and only inflates its word count, so any length "
    "effect that **survives** padding is verbosity bias in its purest form — the judge rewarding "
    "length for its own sake. Below, the length metrics are grouped by **mode** so you can read the "
    "generate→pad drop directly, and the table quantifies how much of each size's length climb was "
    "real content vs. pure verbosity."
)
if cmp_df.empty:
    st.info("No tier runs found — see section 6.")
else:
    for col in ("length_climb", "length_corr"):
        pivot = (
            cmp_df.pivot(index="mode", columns="size", values=col)
            .reindex(["generate", "pad"])
            .dropna(how="all")
        )
        st.markdown(f"**{analysis.COMPARISON_METRICS[col]}** — generate vs pad (bars: model sizes)")
        st.bar_chart(pivot, stack=False)

    climb = cmp_df.pivot(index="size", columns="mode", values="length_climb").reindex(SIZE_ORDER)
    if {"generate", "pad"}.issubset(climb.columns):
        survived = pd.DataFrame(
            {
                "Generate climb": climb["generate"].round(2),
                "Pad climb (pure verbosity)": climb["pad"].round(2),
                "Verbosity share (pad ÷ generate)": (climb["pad"] / climb["generate"]).round(2),
            }
        ).dropna(how="all")
        st.markdown("**How much of the length climb is pure verbosity?**")
        st.dataframe(survived, use_container_width=True)
        st.caption(
            "**Pad climb** is the length effect with content held fixed — pure verbosity bias. "
            "**Verbosity share** = pad ÷ generate: near **0** means the length effect was mostly real "
            "content (*longer genuinely says more*); near **1** means the judges were rewarding length "
            "itself. Negative values mean scores actually dropped with length in that run."
        )
