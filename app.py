"""VoterPulse Streamlit app — EDA, metrics, and a live predictor."""

from __future__ import annotations

from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st

from src.features import IMPROVED_FEATURES, add_base_features, is_reserved, is_urban

ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_frame():
    df = add_base_features(pd.read_csv(ROOT / "data" / "voter_turnout.csv"))
    return df


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "tuned_gradient_boosting.joblib")
    cfg = joblib.load(ROOT / "models" / "feature_config.joblib")
    return model, cfg


st.set_page_config(page_title="VoterPulse", layout="wide")
st.title("VoterPulse")
st.caption("Predicting Lok Sabha constituency turnout · ML assignment")

df = load_frame()
model, cfg = load_model()

tab_home, tab_eda, tab_metrics, tab_predict = st.tabs(
    ["Problem", "EDA", "Evaluation", "Predictor"]
)

with tab_home:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Constituencies", f"{len(df):,}")
    c2.metric("States / UTs", df["state_name"].nunique())
    c3.metric("Mean target turnout", f"{df['target_turnout_pct'].mean():.1f}%")
    c4.metric("Prev ↔ target corr", f"{df['previous_turnout_pct'].corr(df['target_turnout_pct']):.2f}")
    st.markdown(
        """
**Problem.** Given a parliamentary constituency's electoral roll and its previous
election turnout, predict the next turnout percentage.

This is **supervised regression**. Success means beating the naive baseline
(reuse last turnout) on a held-out 20% of seats.

**Best model.** Tuned Gradient Boosting · test R² **0.926** · MAE **2.23 pp**.
"""
    )
    st.dataframe(df.head(12), use_container_width=True)

with tab_eda:
    st.subheader("Turnout distribution")
    st.bar_chart(df[["previous_turnout_pct", "target_turnout_pct"]], height=280)
    st.subheader("Average target turnout by state")
    state = (
        df.groupby("state_name")["target_turnout_pct"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(state, height=420)
    st.subheader("Urban vs other seats")
    st.write(df.groupby("is_urban")["target_turnout_pct"].mean())
    fig_dir = ROOT / "reports" / "figures"
    cols = st.columns(2)
    for i, name in enumerate(
        ["02_prev_vs_target.png", "06_actual_vs_pred.png", "05_model_r2.png", "03_state_turnout.png"]
    ):
        path = fig_dir / name
        if path.exists():
            cols[i % 2].image(str(path), use_container_width=True)

with tab_metrics:
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        st.subheader("Test-set scores")
        st.dataframe(pd.read_json(metrics_path), use_container_width=True)
    st.markdown(
        """
| Metric | Why we use it |
| --- | --- |
| MAE | Average miss in percentage points — the viva-friendly number |
| RMSE | Same units, but large mistakes hurt more |
| R² | Share of variance explained vs predicting the mean |
| MAPE | Relative error, so a miss on a 50% seat is not treated like a miss on an 85% seat |
"""
    )

with tab_predict:
    st.subheader("Live predictor")
    states = sorted(df["state_name"].unique())
    state = st.selectbox("State / UT", states)
    seats = df.loc[df["state_name"] == state, "pc_name"].tolist()
    name = st.selectbox("Constituency", seats)
    row = df[(df["state_name"] == state) & (df["pc_name"] == name)].iloc[0]

    prev = st.slider("Previous turnout %", 40.0, 95.0, float(row["previous_turnout_pct"]))
    electors = st.number_input("Total electors", value=int(row["total_electors"]), step=1000)
    female_share = st.slider("Female share of roll", 0.35, 0.65, float(row["female_share"]))
    reserved = st.checkbox("Reserved (SC/ST)", value=bool(row["is_reserved"]))
    urban = st.checkbox("Urban / metro", value=bool(row["is_urban"]))

    feat = {
        "previous_turnout_pct": prev,
        "log_electors": math.log1p(electors),
        "female_share": female_share,
        "third_share": float(row["third_share"]),
        "gender_gap": float(row["gender_gap"]),
        "state_mean_prev": cfg["state_means"].get(state, cfg["global_prev_mean"]),
        "is_reserved": int(reserved),
        "is_urban": int(urban),
    }
    X = pd.DataFrame([feat], columns=IMPROVED_FEATURES)
    pred = float(model.predict(X)[0])
    st.metric("Predicted turnout", f"{pred:.2f}%", delta=f"{pred - row['target_turnout_pct']:+.2f} vs actual")
    st.write(
        f"Recorded target **{row['target_turnout_pct']:.2f}%** · previous **{row['previous_turnout_pct']:.2f}%**."
    )
    st.caption(
        f"Name flags: reserved={is_reserved(name)}, urban={is_urban(name)}."
    )
