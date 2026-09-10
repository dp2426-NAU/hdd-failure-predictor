"""
Operator Lookup -- the technician dashboard.

Pick a drive from the current live snapshot (or the static training sample),
or type in SMART readings by hand, and see its individual failure-risk score
plus which readings are driving that score.
"""

import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

import auth
import live_feed
from styles import inject_base_css, stat_card, COLOR_ACCENT, COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL

MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/feature_names.json"
METRICS_PATH = "model/metrics.json"
SAMPLE_DATA_PATH = "data/processed_balanced.csv"

FEATURE_LABELS = {
    "smart_5_raw": "Reallocated Sectors (SMART 5)",
    "smart_9_raw": "Power-On Hours (SMART 9)",
    "smart_187_raw": "Reported Uncorrectable Errors (SMART 187)",
    "smart_188_raw": "Command Timeout (SMART 188)",
    "smart_194_raw": "Temperature °C (SMART 194)",
    "smart_197_raw": "Current Pending Sectors (SMART 197)",
    "smart_198_raw": "Offline Uncorrectable Sectors (SMART 198)",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    # SHAP's TreeExplainer is exact and fast for tree ensembles like ours --
    # no approximation needed. Leading underscore on the param tells
    # Streamlit's cache not to try (and fail) to hash the model object.
    return shap.TreeExplainer(_model)


@st.cache_data
def load_support_files():
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    sample_df = pd.read_csv(SAMPLE_DATA_PATH)
    return feature_columns, metrics, sample_df


st.set_page_config(page_title="Operator Lookup · Drive Failure", page_icon="🔧", layout="wide")
auth.require_password()
inject_base_css()
live_feed.init_session_state()

model = load_model()
explainer = load_explainer(model)
feature_columns, metrics, sample_df = load_support_files()

st.title("🔧 Operator Lookup")
st.caption(f"Check one drive's risk of failing within the next {metrics.get('horizon_days', 7)} days, and see "
           "exactly which SMART readings are driving THIS prediction (SHAP values) -- not just which "
           "features matter on average across the whole model.")

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Check a drive")
    source = st.radio(
        "Input source",
        ["Current live fleet snapshot", "Training sample", "Enter SMART values manually"],
        label_visibility="collapsed",
    )

    actual_label = None
    if source == "Current live fleet snapshot":
        snap = live_feed.get_fleet_snapshot()
        chosen = st.selectbox("Drive (serial number) — as of the Fleet Overview's current simulated day", snap["serial_number"].tolist())
        row = snap[snap["serial_number"] == chosen].iloc[0]
        input_values = {col: float(row[col]) for col in feature_columns}
    elif source == "Training sample":
        options = sample_df["serial_number"].unique().tolist()
        chosen = st.selectbox("Example drive (serial number)", options)
        row = sample_df[sample_df["serial_number"] == chosen].iloc[-1]
        input_values = {col: float(row[col]) for col in feature_columns}
        actual_label = int(row["failure"]) if "failure" in row else None
    else:
        input_values = {}
        for col in feature_columns:
            default = float(sample_df[col].median())
            input_values[col] = st.number_input(FEATURE_LABELS.get(col, col), value=default, min_value=0.0)

    X_input = pd.DataFrame([input_values])[feature_columns]
    risk = float(model.predict_proba(X_input)[0][1])

    if risk >= 0.66:
        risk_color, risk_word = COLOR_CRITICAL, "High risk"
    elif risk >= 0.33:
        risk_color, risk_word = COLOR_WARNING, "Elevated risk"
    else:
        risk_color, risk_word = COLOR_GOOD, "Low risk"

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#e9efed"}},
        title={"text": risk_word, "font": {"size": 18, "color": risk_color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#93a19d"},
            "bar": {"color": risk_color},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "#2b3538",
            "steps": [
                {"range": [0, 33], "color": "rgba(76,175,125,0.15)"},
                {"range": [33, 66], "color": "rgba(224,171,84,0.15)"},
                {"range": [66, 100], "color": "rgba(224,112,90,0.15)"},
            ],
        },
    ))
    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge, use_container_width=True)

    if actual_label is not None:
        st.caption(f"Ground truth for this example: {'**failed**' if actual_label else 'stayed healthy'} in the training data.")

with right:
    st.subheader("Why THIS drive scored this way")
    st.caption("SHAP contributions for this specific prediction — how much each reading pushed the risk "
               "score up (red) or down (green) from the model's baseline, not just which features matter globally.")

    shap_raw = explainer.shap_values(X_input)
    # TreeExplainer on a binary RandomForest returns shape (n_samples, n_features, n_classes);
    # we want the positive ("will fail soon") class, index 1.
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1][0]
        base_value = explainer.expected_value[1]
    elif shap_raw.ndim == 3:
        shap_values = shap_raw[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_values = shap_raw[0]
        base_value = explainer.expected_value

    shap_df = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(c, c) for c in feature_columns],
        "contribution": shap_values,
    }).sort_values("contribution", key=abs, ascending=True)
    bar_colors = [COLOR_CRITICAL if v > 0 else COLOR_GOOD for v in shap_df["contribution"]]

    shap_fig = go.Figure(go.Bar(
        x=shap_df["contribution"], y=shap_df["feature"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}" for v in shap_df["contribution"]], textposition="outside",
    ))
    shap_fig.update_layout(
        height=280, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e9efed",
        xaxis=dict(title=f"← lowers risk    SHAP contribution    raises risk →   (baseline: {base_value:.0%})",
                   gridcolor="#2b3538", zeroline=True, zerolinecolor="#93a19d"),
        yaxis_title=None,
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    with st.expander("Global feature importance (model-wide, not this drive)"):
        importances = metrics["feature_importances"]
        imp_df = pd.DataFrame(
            [{"feature": FEATURE_LABELS.get(k, k), "importance": v} for k, v in importances.items()]
        ).sort_values("importance", ascending=True)
        bar = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color=COLOR_ACCENT))
        bar.update_layout(
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e9efed",
            xaxis=dict(title="Average importance across all drives", gridcolor="#2b3538"), yaxis_title=None,
        )
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("This drive's readings vs. a typical healthy drive")
    healthy_medians = sample_df[sample_df["failure"] == 0][feature_columns].median()
    compare_df = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(c, c) for c in feature_columns],
        "this drive": [input_values[c] for c in feature_columns],
        "typical healthy drive": [healthy_medians[c] for c in feature_columns],
    })
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

st.divider()
report = metrics["classification_report"]["1"]
c1, c2, c3 = st.columns(3)
with c1: stat_card("Precision", f"{report['precision']:.0%}")
with c2: stat_card("Recall", f"{report['recall']:.0%}")
with c3: stat_card("F1", f"{report['f1-score']:.0%}")
