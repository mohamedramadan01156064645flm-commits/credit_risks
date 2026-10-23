"""Interactive demo of the credit scoring model.

Run locally:
    uv run streamlit run app/demo.py

Locally, the demo offers real test set clients (if data/processed/features.parquet exists).
Online, Kaggle data is not published (competition rules): the demo provides
representative profiles that can be customized with sliders.
"""
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# On Streamlit Community Cloud, the package is not installed: import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk_scoring import decision
from credit_risk_scoring.features import FEATURES_PATH
from credit_risk_scoring.scoring import assess, load_model, to_frame

st.set_page_config(page_title="Credit Scoring", page_icon=":material/account_balance:", layout="wide")

GRADE_COLORS = {"A": "green", "B": "green", "C": "yellow", "D": "orange", "E": "red"}
EDUCATION = ["Higher education", "Secondary / secondary special", "Incomplete higher",
             "Lower secondary", "Academic degree"]
YEAR = 365.25

# Reference client: median (or most frequent category) of each variable on training data.
# Aggregated statistics only, no real client data. Representative profiles are built from this.
MEDIAN_CLIENT_PATH = Path(__file__).parent / "client_median.json"

# Representative profiles (fictional): only features differing from median client are listed
PROFILES = {
    "Strong profile": {
        "EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.72, "EXT_SOURCE_3": 0.70, "AMT_INCOME_TOTAL": 250_000,
        "AMT_CREDIT": 400_000, "AMT_ANNUITY": 20_000, "AGE_YEARS": 48, "DAYS_EMPLOYED": -12 * YEAR,
        "NAME_EDUCATION_TYPE": "Higher education", "OCCUPATION_TYPE": "Managers",
    },
    "Average profile": {
        "EXT_SOURCE_1": 0.50, "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.50, "AMT_INCOME_TOTAL": 150_000,
        "AMT_CREDIT": 600_000, "AMT_ANNUITY": 30_000, "AGE_YEARS": 38, "DAYS_EMPLOYED": -4 * YEAR,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    },
    "Fragile profile": {
        "EXT_SOURCE_1": 0.20, "EXT_SOURCE_2": 0.20, "EXT_SOURCE_3": 0.15, "AMT_INCOME_TOTAL": 100_000,
        "AMT_CREDIT": 900_000, "AMT_ANNUITY": 45_000, "AGE_YEARS": 26, "DAYS_EMPLOYED": -0.5 * YEAR,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "INST_LATE_RATIO": 0.3, "INST_LATE_RATIO_LAST_YEAR": 0.4, "PREV_REFUSED_RATIO": 0.5,
    },
    "Pensioner": {
        "EXT_SOURCE_1": None, "EXT_SOURCE_2": 0.60, "EXT_SOURCE_3": 0.65, "AMT_INCOME_TOTAL": 90_000,
        "AMT_CREDIT": 200_000, "AMT_ANNUITY": 12_000, "AGE_YEARS": 63, "DAYS_EMPLOYED_ANOM": 1,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special", "NAME_INCOME_TYPE": "Pensioner",
        "OCCUPATION_TYPE": None, "ORGANIZATION_TYPE": "XNA",
    },
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(max_entries=1)
def get_test_clients(n: int = 300) -> pd.DataFrame | None:
    """Sample of test set applications (never seen by the model), with their actual outcome."""
    if os.environ.get("DEMO_PROFILES_ONLY") or os.environ.get("DEMO_PROFILS_SEULEMENT") or not FEATURES_PATH.exists():
        return None
    from credit_risk_scoring.modeling import load_features, split

    df = load_features()
    _, X_test, _, y_test = split(df)
    sample = X_test.sample(n, random_state=7).copy()
    sample["TARGET"] = y_test.loc[sample.index]
    sample["SK_ID_CURR"] = df.loc[sample.index, "SK_ID_CURR"]
    return sample.set_index("SK_ID_CURR")


@st.cache_data
def get_median_client() -> dict:
    if not MEDIAN_CLIENT_PATH.exists():
        return {}
    return json.loads(MEDIAN_CLIENT_PATH.read_text(encoding="utf-8"))


def clean(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else value


def show(value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ") if abs(value) >= 1000 else f"{value:.3g}"
    return str(value)


model = get_model()
clients = get_test_clients()

st.title("Credit Risk Scoring")
st.caption("LightGBM model trained on 246,000 credit applications (Home Credit) · decision based on "
           f"profitability: margin {decision.MARGIN:.0%}, loss given default {decision.LGD:.0%}")

# ---------- Application Selection ----------
with st.sidebar:
    st.header("Credit Application")
    sources = ["Real clients", "Representative profiles"] if clients is not None else ["Representative profiles"]
    source = st.segmented_control("Source", sources, default=sources[0], required=True)

    if source == "Real clients":
        client_id = st.selectbox("Test set client", clients.index,
                                 help="Applications never seen by the model; actual outcome is known.")
        row = clients.loc[client_id]
        base = {c: clean(row[c]) for c in model.features}
        st.caption("Actual outcome: " + ("payment default :material/close:" if row["TARGET"] == 1
                                        else "loan repaid :material/check:"))
        key = str(client_id)
    else:
        profile = st.selectbox("Profile", list(PROFILES),
                               help="Fictional profiles built from the median client, customizable below.")
        base = {**get_median_client(), **PROFILES[profile]}
        base["AMT_GOODS_PRICE"] = None   # recomputed from credit amount
        key = profile

    st.subheader("Simulate a modification")

    def num(label, name, default, lo, hi, step, fmt=None):
        value = base.get(name)
        value = default if value is None else float(np.clip(value, lo, hi))
        return st.slider(label, float(lo), float(hi), float(value), float(step), format=fmt, key=f"{name}_{key}")

    values = {}
    for name in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        known = st.toggle(f"External score {name[-1]} available", value=base.get(name) is not None,
                          key=f"has_{name}_{key}")
        values[name] = num(f"External score {name[-1]}", name, 0.5, 0.0, 1.0, 0.01) if known else None
    values["AMT_INCOME_TOTAL"] = num("Annual income", "AMT_INCOME_TOTAL", 150_000, 20_000, 1_000_000, 5_000, "%d")
    values["AMT_CREDIT"] = num("Credit amount", "AMT_CREDIT", 500_000, 40_000, 3_000_000, 10_000, "%d")
    values["AMT_ANNUITY"] = num("Annuity", "AMT_ANNUITY", 25_000, 2_000, 200_000, 500, "%d")
    values["DAYS_BIRTH"] = -num("Age (years)", "AGE_YEARS", 40, 20, 70, 1, "%d") * YEAR
    retired = base.get("DAYS_EMPLOYED_ANOM") == 1
    if st.toggle("Unemployed or pensioner", value=retired, key=f"retired_{key}"):
        values["DAYS_EMPLOYED"] = 365243
    else:
        seniority = -(base.get("DAYS_EMPLOYED") or -5 * YEAR) / YEAR
        base["_SENIORITY"] = seniority
        values["DAYS_EMPLOYED"] = -num("Employment length (years)", "_SENIORITY", 5, 0, 40, 0.5) * YEAR
    education = base.get("NAME_EDUCATION_TYPE")
    values["NAME_EDUCATION_TYPE"] = st.selectbox(
        "Education level", EDUCATION, index=EDUCATION.index(education) if education in EDUCATION else 1,
        key=f"education_{key}",
    )

features = {k: v for k, v in {**base, **values}.items() if k in model.features}
# Derived features are recomputed from input values (as in the API)
for derived in ["AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
                "CREDIT_GOODS_RATIO", "INCOME_PER_PERSON", "EMPLOYED_AGE_RATIO", "DAYS_EMPLOYED_ANOM"]:
    features.pop(derived, None)
if features.get("AMT_GOODS_PRICE") is None:
    features["AMT_GOODS_PRICE"] = 0.9 * features["AMT_CREDIT"]

result = assess(model, to_frame(model, features), n_reasons=6)

# ---------- Result ----------
with st.container(horizontal=True):
    st.metric("Default probability", f"{result.pd:.1%}", border=True)
    st.metric("Score", f"{result.score:.0f} points", border=True)
    st.metric("Risk grade", result.grade, border=True)
    st.metric("Decision", result.decision, border=True)

accepted = result.decision == "ACCEPTED"
st.badge(f"Grade {result.grade} · {decision.risk_grades().set_index('grade').loc[result.grade, 'label']}",
         color=GRADE_COLORS[result.grade])
st.progress(min(result.pd / 0.4, 1.0),
            text=f"PD {result.pd:.1%} · refusal threshold {result.pd_threshold:.1%} "
                 f"({decision.pd_to_score(result.pd_threshold):.0f} points)")
if accepted:
    st.success("The application is profitable on average for the bank: credit granted.", icon=":material/check_circle:")
else:
    st.error("Default risk exceeds profitability threshold: credit rejected.", icon=":material/block:")

st.subheader("Why this decision?")
st.caption("Contribution of each feature to this client's risk (SHAP values). "
           "Red: increases risk · green: decreases risk.")
reasons = result.risk_reasons + result.favorable_reasons
if reasons:
    df = pd.DataFrame([{"variable": r.label, "effect": r.effect, "value": show(r.value)} for r in reasons])
    df = df.sort_values("effect", ascending=False)
    df["direction"] = np.where(df["effect"] > 0, "Increases risk", "Decreases risk")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("effect:Q", title="Effect on risk (log-odds)", axis=alt.Axis(tickCount=6, format=".1f")),
            y=alt.Y("variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                                          range=["#C62828", "#2E7D32"]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["variable", "value", alt.Tooltip("effect:Q", format=".3f")],
        )
        .properties(height=34 * len(df))
    )
    st.altair_chart(chart, alt="Top feature contributions to client risk")
    with st.expander("Detailed values"):
        st.dataframe(df.drop(columns="direction").set_index("variable"), alt="Detailed contributions by feature")

with st.expander("Risk grades and policy"):
    grades = decision.risk_grades()
    st.dataframe(grades.style.format({"pd_min": "{:.1%}", "pd_max": "{:.1%}",
                                      "score_min": "{:.0f}", "score_max": "{:.0f}"}),
                 hide_index=True, alt="Risk grades A to E")
    st.markdown(f"A loan is profitable if **(1 − PD) × {decision.MARGIN:.0%} > PD × {decision.LGD:.0%}**, "
                f"meaning **PD < {decision.breakeven_threshold():.1%}**.")

st.caption("Project by Kevin Moukam Nkameni · [source code on GitHub](https://github.com/MKkevin2712/credit-risk-scoring)")
