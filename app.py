"""
Predicting Drive Failure -- Home

Landing page for the three dashboards:
  - Fleet Overview    (pages/1_Fleet_Overview.py) -- leadership/budget view, live-feel
  - Operator Lookup   (pages/2_Operator_Lookup.py) -- technician view, single-drive detail, SHAP-explained
  - Optimization Lab  (pages/3_Optimization_Lab.py) -- remaining-useful-life + cost-optimized threshold

Run locally:
    streamlit run app.py
"""

import json

import streamlit as st

from styles import inject_base_css, stat_card
import auth
import live_feed
import provenance

with open("model/metrics.json") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Predicting Drive Failure", page_icon="💽", layout="wide")
auth.require_password()
inject_base_css()
live_feed.init_session_state()

st.title("💽 Predicting Drive Failure")
st.caption("A predictive-maintenance prototype for IT infrastructure, built on Backblaze's real Drive Stats schema.")

st.markdown(provenance.banner_html(), unsafe_allow_html=True)
st.markdown(
    """
    <div class="disclosure" style="margin-top:10px;">
    <b>What "live" means here:</b> there is no live sensor behind this demo — no public real-time API for
    hard-drive SMART telemetry exists. What you're seeing on the Fleet Overview page is a
    <b>chronological replay</b> of drive-day records, advancing one simulated day at a time, exactly the
    way this dashboard would behave if it were wired up to a real fleet.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("🖥️ Fleet Overview")
    st.write("The leadership / budget view — fleet-wide risk trend, a live event feed, a 3D rack of every "
             "monitored drive colored by risk, and a cost-avoidance estimate tied to real downtime-cost research.")
    st.page_link("pages/1_Fleet_Overview.py", label="Open Fleet Overview →", icon="🖥️")
with col2:
    st.subheader("🔧 Operator Lookup")
    st.write("The technician view — pick one drive (or enter SMART readings by hand) and see its individual "
             "failure-risk score, explained per-drive with SHAP rather than only a global importance chart.")
    st.page_link("pages/2_Operator_Lookup.py", label="Open Operator Lookup →", icon="🔧")
with col3:
    st.subheader("🧪 Optimization Lab")
    st.write("The differentiator: remaining-useful-life estimation (survival analysis) plus a 3D "
             "cost-optimized alert-threshold surface built from real test-set outcomes.")
    st.page_link("pages/3_Optimization_Lab.py", label="Open Optimization Lab →", icon="🧪")

st.write("")
with st.expander("What makes this different from a standard \"predict drive failure\" project", expanded=False):
    st.markdown(
        "- **Early-warning labeling, not same-day diagnosis.** The classifier is trained to predict failure "
        f"within the next {metrics.get('horizon_days', 7)} days from *prior* "
        "readings — the day a drive actually fails is excluded from training, because using it would just be "
        "recognizing an already-failed drive, not predicting one.\n"
        "- **Remaining useful life, not just yes/no.** A Cox Proportional Hazards survival model (Optimization "
        "Lab) estimates *how long* a drive likely has left, the standard framing in real industrial predictive "
        "maintenance — most course projects stop at binary classification.\n"
        "- **Per-prediction explainability.** SHAP values (Operator Lookup) explain *this specific drive's* "
        "score, not just which features matter on average across the model.\n"
        "- **A genuine decision-support tool, not just a risk score.** The 3D cost surface (Optimization Lab) "
        "is built from real held-out test predictions and shows that the *right* alert threshold depends on "
        "an organization's own cost structure — there's no universal answer, which is the actually useful, "
        "non-obvious result.\n"
    )

st.write("")
st.subheader("Model snapshot")
report = metrics["classification_report"]["1"]
c1, c2, c3, c4 = st.columns(4)
with c1: stat_card("Precision (failure class)", f"{report['precision']:.0%}")
with c2: stat_card("Recall (failure class)", f"{report['recall']:.0%}")
with c3: stat_card("F1 score", f"{report['f1-score']:.0%}")
with c4: stat_card("Test rows", f"{metrics['n_test']:,}")

st.caption("Prototype for an IT capstone project · trained with scikit-learn · schema and failure-rate "
           "baseline sourced from Backblaze's public Drive Stats reports.")
