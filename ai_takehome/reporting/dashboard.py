"""Streamlit dashboard over results/metrics.jsonl.

Shows every tracked run: headline numbers for the latest, a run-over-run
comparison, a per-question breakdown, and the raw table.

    streamlit run reporting/dashboard.py

Add runs with:  python -m evaluation.track_results --backend st --note "what changed"
"""

from __future__ import annotations

import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "results" / "metrics.jsonl"


def display_path(path: pathlib.Path) -> str:
    """Relative to cwd when possible; absolute otherwise (streamlit may run
    from anywhere, and relative_to() raises rather than falling back)."""
    try:
        return str(path.relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(path)

# Categorical slots 1-3 from the validated palette. Three is the cap for
# all-pairs charts; a fourth series would put yellow beside orange and fail
# the CVD floor. Aqua sits under 3:1 on the light surface, so the table view
# below is not optional -- it is the required relief.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

METRICS = {
    "retrieval_precision": "Retrieval precision",
    "retrieval_recall": "Retrieval recall",
    "answer_recall": "Answer recall",
}

st.set_page_config(page_title="Denials RAG — eval results", layout="wide")


@st.cache_data
def load(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["run_label"] = (
        df["timestamp"].dt.strftime("%m-%d %H:%M")
        + "  " + df["model"].astype(str)
        + " @" + df["threshold"].astype(str)
    )
    df["short_question"] = df["question"].str.slice(0, 38)
    return df


def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: mean metrics and pass count."""
    agg = df.groupby(["run_id", "run_label", "version", "model", "threshold", "note"],
                     as_index=False).agg(
        retrieval_precision=("retrieval_precision", "mean"),
        retrieval_recall=("retrieval_recall", "mean"),
        answer_recall=("answer_recall", "mean"),
        passed=("answer_passed", "sum"),
        questions=("answer_passed", "count"),
        timestamp=("timestamp", "min"),
    )
    return agg.sort_values("timestamp")


def grouped_bars(data: pd.DataFrame, x_field: str, x_title: str, height: int):
    """Grouped bars, one color per metric, direct-labelled."""
    base = alt.Chart(data).encode(
        y=alt.Y(f"{x_field}:N", title=x_title, sort=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=12)),
        yOffset=alt.YOffset("metric:N", sort=list(METRICS.values())),
    )
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                         height=9).encode(
        x=alt.X("value:Q", title="score", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", grid=True, tickCount=5)),
        color=alt.Color("metric:N", sort=list(METRICS.values()),
                        scale=alt.Scale(range=SERIES),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip(f"{x_field}:N", title=x_title),
                 alt.Tooltip("metric:N", title="metric"),
                 alt.Tooltip("value:Q", title="score", format=".2f")],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#555").encode(
        x=alt.X("value:Q"),
        text=alt.Text("value:Q", format=".2f"),
    )
    return (bars + labels).properties(height=height).configure_view(strokeWidth=0)


def to_long(data: pd.DataFrame, key: str) -> pd.DataFrame:
    long = data.melt(id_vars=[key], value_vars=list(METRICS),
                     var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(METRICS)
    return long


df = load(RESULTS_PATH)

st.title("Denials RAG — evaluation results")

if df.empty:
    st.warning(
        f"No results yet at `{RESULTS_PATH}`.\n\n"
        "Generate some with `python -m evaluation.track_results --backend st --note \"first run\"`."
    )
    st.stop()

st.caption(
    f"{df['run_id'].nunique()} runs · {len(df)} question-records · "
    f"source `{display_path(RESULTS_PATH)}`"
)

# --- filters, one row above the charts -------------------------------------
c1, c2, c3 = st.columns([2, 2, 3])
models = sorted(df["model"].dropna().unique())
chosen_models = c1.multiselect("Model", models, default=models)
versions = sorted(df["version"].dropna().unique())
chosen_versions = c2.multiselect("Pipeline version", versions, default=versions)

view = df[df["model"].isin(chosen_models) & df["version"].isin(chosen_versions)]
if view.empty:
    st.info("No runs match those filters.")
    st.stop()

summary = run_summary(view)
run_labels = list(summary["run_label"])
selected_label = c3.selectbox("Inspect run", run_labels, index=len(run_labels) - 1)
selected = view[view["run_label"] == selected_label]

# --- headline numbers for the selected run ---------------------------------
latest = summary[summary["run_label"] == selected_label].iloc[0]
previous = summary[summary["timestamp"] < latest["timestamp"]]
prior = previous.iloc[-1] if len(previous) else None


def delta(field: str) -> str | None:
    if prior is None:
        return None
    return f"{latest[field] - prior[field]:+.2f}"


m1, m2, m3, m4 = st.columns(4)
m1.metric("Answer eval", f"{int(latest['passed'])}/{int(latest['questions'])}")
m2.metric("Retrieval precision", f"{latest['retrieval_precision']:.2f}",
          delta("retrieval_precision"))
m3.metric("Retrieval recall", f"{latest['retrieval_recall']:.2f}",
          delta("retrieval_recall"))
m4.metric("Model", latest["model"], help=f"pipeline v{latest['version']}, "
                                         f"threshold {latest['threshold']}")
if latest["note"]:
    st.caption(f"Note: {latest['note']}")

st.divider()

# --- per-question breakdown for the selected run ---------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("By question")
    st.caption(f"{selected_label}")
    st.altair_chart(
        grouped_bars(to_long(selected, "short_question"), "short_question", "question",
                     height=max(260, 62 * len(selected))),
        width='stretch',
    )

with right:
    st.subheader("Across runs")
    st.caption("Mean over the five evaluation questions")
    st.altair_chart(
        grouped_bars(to_long(summary, "run_label"), "run_label", "run",
                     height=max(260, 62 * len(summary))),
        width='stretch',
    )

st.divider()

# --- answers, so a score can be checked against what was actually said ------
st.subheader("Generated answers")
for _, row in selected.iterrows():
    icon = "✅" if row["answer_passed"] else "❌"
    with st.expander(f"{icon}  {row['question']}"):
        st.write(row["answer_text"])
        st.caption(
            f"retrieved {row['retrieval_n_retrieved']} of "
            f"{row['retrieval_n_relevant']} relevant · "
            f"precision {row['retrieval_precision']:.2f} · "
            f"recall {row['retrieval_recall']:.2f} · "
            f"answer P_true {row['answer_precision_true']:.2f}"
        )

# --- table view (required relief for the low-contrast series) --------------
st.subheader("Table")
tab_run, tab_all = st.tabs(["Selected run", "All runs"])
with tab_run:
    st.dataframe(
        selected[["question", "retrieval_n_retrieved", "retrieval_n_relevant",
                  "retrieval_precision", "retrieval_recall", "retrieval_f1",
                  "answer_recall", "answer_precision_true", "answer_passed"]],
        width='stretch', hide_index=True,
    )
with tab_all:
    st.dataframe(
        summary[["run_label", "version", "model", "threshold", "passed",
                 "retrieval_precision", "retrieval_recall", "answer_recall", "note"]],
        width='stretch', hide_index=True,
    )
