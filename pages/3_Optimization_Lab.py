"""
Optimization Lab -- the differentiator page.

Two things most tutorial-style "predict drive failure" projects don't do:

1. Remaining Useful Life (survival analysis) -- instead of a yes/no failure
   flag, estimate how much longer a specific drive is likely to keep running,
   via a Cox Proportional Hazards model (lifelines). This is the standard
   framing used in real industrial predictive maintenance.

2. A cost-optimization simulator -- the classifier's risk score alone doesn't
   tell you what threshold to actually alert at. This page answers that,
   using REAL held-out test predictions (not a hypothetical), and shows the
   genuinely useful, non-obvious result: the right threshold depends on your
   organization's cost structure, there's no universal answer. The 3D surface
   here is a real data-driven relationship (threshold x cost-ratio x
   projected cost), not a decorative chart -- one of the few cases where 3D
   earns its place in a dashboard.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth
from styles import inject_base_css, stat_card, COLOR_ACCENT, COLOR_GOOD, COLOR_CRITICAL
from cost_simulator import build_cost_surface

st.set_page_config(page_title="Optimization Lab · Drive Failure", page_icon="🧪", layout="wide")
auth.require_password()
inject_base_css()

st.title("🧪 Optimization Lab")
st.caption("Remaining-useful-life estimation and cost-optimized alert thresholds — going beyond a plain "
           "risk score to answer \"how long has it got\" and \"what threshold should we actually use.\"")


# ---------------------------------------------------------------- RUL ----
@st.cache_resource
def load_survival_model():
    return joblib.load("model/survival_model.pkl")


@st.cache_data
def load_survival_support():
    with open("model/survival_metrics.json") as f:
        metrics = json.load(f)
    km_curve = pd.DataFrame(json.load(open("model/km_curve.json")))
    survival_df = pd.read_csv("data/survival_data.csv")
    return metrics, km_curve, survival_df


surv = load_survival_model()
cph, surv_features = surv["model"], surv["feature_columns"]
surv_metrics, km_curve, survival_df = load_survival_support()

st.markdown("### Remaining useful life")
c1, c2, c3 = st.columns([1, 1, 1])
with c1: stat_card("Concordance index", f"{surv_metrics['concordance_index']:.3f}",
                    sub="How well the model ranks who fails first (0.5=random, 1.0=perfect)")
with c2: stat_card("Drives in survival model", f"{surv_metrics['n_drives']:,}")
with c3: stat_card("Failure events observed", f"{surv_metrics['n_events']:,}")

rul_left, rul_right = st.columns([1, 1.4])
with rul_left:
    st.markdown("##### Pick a drive")
    options = survival_df["serial_number"].tolist()
    chosen = st.selectbox("Drive", options, label_visibility="collapsed")
    drive_row = survival_df[survival_df["serial_number"] == chosen].iloc[0]
    covariates = pd.DataFrame([drive_row[surv_features].to_dict()])

    surv_func = cph.predict_survival_function(covariates)
    days = surv_func.index.values
    probs = surv_func.values.flatten()
    median_idx = np.argmin(np.abs(probs - 0.5)) if (probs <= 0.5).any() else None
    median_days = float(days[median_idx]) if median_idx is not None else None

    if median_days is not None:
        stat_card("Estimated remaining useful life", f"~{median_days:.0f} days",
                   sub="Days until this drive's estimated 50% survival probability")
    else:
        stat_card("Estimated remaining useful life", "> observation window",
                   sub="Model doesn't project a 50% failure probability within the data it has seen")

    st.caption("⚠️ Simplification disclosed: covariates use each drive's LAST observed reading as a static "
               "input to the Cox model, not a full time-varying history — a reasonable first pass for a "
               "course project, not a claim of clinical-grade survival modeling. See README.md.")

with rul_right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_curve["day"], y=km_curve["survival_probability"],
                              mode="lines", name="Fleet-wide (Kaplan-Meier)",
                              line=dict(color=COLOR_ACCENT, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=days, y=probs, mode="lines", name=f"This drive ({chosen})",
                              line=dict(color=COLOR_CRITICAL, width=2.5)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#93a19d", annotation_text="50% survival")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e9efed",
        xaxis=dict(title="Days", gridcolor="#2b3538"),
        yaxis=dict(title="Survival probability", gridcolor="#2b3538", tickformat=".0%"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------- cost surface ----
st.markdown("### Cost-optimized alert threshold")
st.caption("Built from the classifier's real held-out test predictions — not a hypothetical. Adjust your "
           "organization's cost assumptions and watch the optimal threshold move.")

cs1, cs2, cs3 = st.columns(3)
with cs1:
    replacement_cost = st.slider("Cost per proactive drive replacement ($)", 100, 3000, 500, step=50)
with cs2:
    horizon_days = st.select_slider("Planning horizon", options=[30, 90, 180, 365], value=365)
with cs3:
    st.caption("Cost ratio (downtime ÷ replacement) is the Y-axis of the surface below — it's the variable "
               "that actually moves the optimal threshold, which is why it's an axis rather than a slider.")

thresholds, cost_ratios, Z, optimal = build_cost_surface(replacement_cost=replacement_cost, horizon_days=horizon_days)

surface_col, callout_col = st.columns([1.6, 1])
with surface_col:
    surf_fig = go.Figure(data=[go.Surface(
        x=thresholds, y=cost_ratios, z=Z.T,
        colorscale=[[0, COLOR_GOOD], [0.5, "#e0ab54"], [1, COLOR_CRITICAL]],
        showscale=True, colorbar=dict(title="Projected cost ($)"),
    )])
    surf_fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e9efed",
        scene=dict(
            xaxis=dict(title="Alert threshold", backgroundcolor="rgba(0,0,0,0)", gridcolor="#2b3538"),
            yaxis=dict(title="Downtime ÷ replacement cost ratio", backgroundcolor="rgba(0,0,0,0)", gridcolor="#2b3538"),
            zaxis=dict(title="Projected cost ($)", backgroundcolor="rgba(0,0,0,0)", gridcolor="#2b3538"),
        ),
    )
    st.plotly_chart(surf_fig, use_container_width=True)

with callout_col:
    st.markdown("##### Optimal threshold by cost ratio")
    for o in optimal:
        st.markdown(
            f'<div class="stat-card" style="margin-bottom:10px;">'
            f'<p class="stat-label">Ratio {o["cost_ratio"]:.0f}× — threshold {o["optimal_threshold"]:.2f}</p>'
            f'<p class="stat-value" style="font-size:18px;">${o["min_cost"]:,.0f}</p>'
            f'<p class="stat-sub">projected over {horizon_days} days</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption("A low cost ratio (cheap downtime relative to replacement) favors a HIGHER threshold — don't "
               "cry wolf. A high ratio (expensive downtime) favors a LOWER threshold — alert earlier, even "
               "at the cost of some unnecessary replacements. Read this off the surface, don't guess it.")

st.divider()
st.caption("Cost model and its assumptions are disclosed in cost_simulator.py — this is a decision-support "
           "illustration built on real test-set outcomes, not a sourced financial claim.")
