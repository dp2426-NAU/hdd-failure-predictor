"""
live_feed.py -- the "live-feel" simulation engine.

HONEST DESCRIPTION OF WHAT THIS DOES: there is no live sensor feeding this
dashboard. There is no public real-time API for hard-drive SMART telemetry.
What this module does is replay real-schema drive-day records in
chronological order, one simulated "day" per tick, so the dashboard behaves
the way a live fleet-monitoring tool would if it were wired up to actual
servers. Every dashboard screen that uses this says so plainly. This is a
standard, legitimate technique for demoing monitoring tools (it's how most
IoT/ops-dashboard demos work) -- the dishonest version would be *not*
labeling it as a replay.

When you plug in a real Backblaze quarterly download (see README.md), this
same engine replays genuine historical drive-days instead of the simulated
sample -- the mechanism doesn't change, only the authenticity of the data
underneath it does.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import shap
import streamlit as st

import assistant
import email_alerts
import provenance

SAMPLE_DATA_PATH = "data/sample_drive_stats.csv"
REAL_DATA_PATH = "data/real_drive_stats.csv"  # written by scripts/prepare_live_feed_data.py
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/feature_names.json"

RISK_HIGH = 0.66
RISK_ELEVATED = 0.33


def _resolve_data_path() -> str:
    # Same provenance-driven pattern as the Home page banner (provenance.py):
    # prefer the real replay timeline once it's been built from real data,
    # but don't hard-fail back to the bundled sample if prepare_live_feed_data.py
    # hasn't been run yet (e.g. right after only prepare_data.py has).
    p = provenance.load_provenance()
    if p.get("source") == "real_backblaze_download" and os.path.exists(REAL_DATA_PATH):
        return REAL_DATA_PATH
    return SAMPLE_DATA_PATH


DATA_PATH = _resolve_data_path()


@st.cache_resource
def _load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def _load_feature_columns():
    with open(FEATURES_PATH) as f:
        return json.load(f)


@st.cache_resource
def _load_explainer():
    # Exact SHAP TreeExplainer, same as Operator Lookup's own explainer --
    # this is what draft_incident_summary() explains a critical-alert drive
    # with, so the alert email's reasoning matches what a technician would
    # see if they looked the drive up themselves.
    return shap.TreeExplainer(_load_model())


@st.cache_data
def _load_timeline() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_dates() -> list:
    df = _load_timeline()
    return sorted(df["date"].unique())


def risk_tier(risk: float) -> str:
    if risk >= RISK_HIGH:
        return "critical"
    if risk >= RISK_ELEVATED:
        return "elevated"
    return "healthy"


def _score(df_rows: pd.DataFrame) -> pd.Series:
    model = _load_model()
    feature_columns = _load_feature_columns()
    X = df_rows[feature_columns].fillna(0)
    return pd.Series(model.predict_proba(X)[:, 1], index=df_rows.index)


def init_session_state():
    if "live_day_index" not in st.session_state:
        st.session_state.live_day_index = 0
    if "live_history" not in st.session_state:
        st.session_state.live_history = []  # list of {date, avg_risk, high_risk_count, cumulative_failures}
    if "live_events" not in st.session_state:
        st.session_state.live_events = []  # list of {date, message, level}
    if "cumulative_failures" not in st.session_state:
        st.session_state.cumulative_failures = 0
    if "failures_with_prior_warning" not in st.session_state:
        st.session_state.failures_with_prior_warning = 0
    if "previously_high_risk" not in st.session_state:
        st.session_state.previously_high_risk = set()
    if "alerted_serials" not in st.session_state:
        st.session_state.alerted_serials = set()  # drives already emailed this session -- one alert per drive


def advance_one_day():
    """Move the simulated clock forward one day, looping back to the start
    at the end of the dataset so a live demo can run indefinitely."""
    dates = get_dates()
    st.session_state.live_day_index = (st.session_state.live_day_index + 1) % len(dates)
    if st.session_state.live_day_index == 0:
        # looped back to the beginning -- reset the running counters/history
        st.session_state.live_history = []
        st.session_state.live_events = []
        st.session_state.cumulative_failures = 0
        st.session_state.failures_with_prior_warning = 0
        st.session_state.previously_high_risk = set()
        st.session_state.alerted_serials = set()


def reset():
    st.session_state.live_day_index = 0
    st.session_state.live_history = []
    st.session_state.live_events = []
    st.session_state.cumulative_failures = 0
    st.session_state.failures_with_prior_warning = 0
    st.session_state.previously_high_risk = set()
    st.session_state.alerted_serials = set()


def current_date():
    dates = get_dates()
    return dates[st.session_state.live_day_index]


def get_fleet_snapshot() -> pd.DataFrame:
    """Every drive's most recent known reading as of the current simulated day."""
    df = _load_timeline()
    as_of = current_date()
    window = df[df["date"] <= as_of]
    latest = window.sort_values("date").groupby("serial_number", as_index=False).last()
    latest["risk"] = _score(latest)
    latest["tier"] = latest["risk"].apply(risk_tier)
    return latest


def _shap_contributions(row: pd.Series, feature_columns: list[str]) -> list[tuple[str, float]]:
    """Same SHAP extraction logic as Operator Lookup's page, factored out
    so the alert email explains a drive the same way the dashboard would."""
    explainer = _load_explainer()
    X_input = pd.DataFrame([row[feature_columns].to_dict()])[feature_columns]
    shap_raw = explainer.shap_values(X_input)
    if isinstance(shap_raw, list):
        values = shap_raw[1][0]
    elif shap_raw.ndim == 3:
        values = shap_raw[0, :, 1]
    else:
        values = shap_raw[0]
    return list(zip(feature_columns, [float(v) for v in values]))


def _maybe_send_critical_alert(serial: str, row: pd.Series) -> str | None:
    """Drafts an AI incident summary and emails it for a drive that just
    crossed into critical risk -- at most once per drive per session (see
    alerted_serials). Returns an event-feed message on a successful send,
    None otherwise (not configured, or the send failed) so the caller can
    add a visible event only when something real happened.

    HONEST SCOPE: this only runs while someone has the app open and is
    stepping through the replay (or has Live mode on) -- see
    email_alerts.py's docstring. It is not a 24/7 background monitor."""
    if not (assistant.is_configured() and email_alerts.is_configured()):
        return None

    feature_columns = _load_feature_columns()
    readings = {c: row[c] for c in feature_columns}
    shap_contribs = _shap_contributions(row, feature_columns)

    summary = assistant.draft_incident_summary(serial, float(row["risk"]), readings, shap_contribs)
    if not summary:
        return None

    sent = email_alerts.send_alert_email(
        subject=f"[Drive Alert] {serial} flagged HIGH RISK ({row['risk']:.0%})",
        body=summary,
    )
    return "📧 AI alert emailed" if sent else None


def step_and_record():
    """Advance the clock one tick and update running history/events. Call this
    once per UI refresh tick when live mode is on."""
    advance_one_day()
    snapshot = get_fleet_snapshot()
    as_of = current_date()

    high_risk_now = set(snapshot.loc[snapshot["tier"] == "critical", "serial_number"])
    newly_high_risk = high_risk_now - st.session_state.previously_high_risk
    todays_failures = snapshot.loc[(snapshot["date"] == as_of) & (snapshot["failure"] == 1), "serial_number"]

    for serial in todays_failures:
        st.session_state.cumulative_failures += 1
        had_warning = serial in st.session_state.previously_high_risk
        if had_warning:
            st.session_state.failures_with_prior_warning += 1
        st.session_state.live_events.insert(0, {
            "date": str(as_of.date()),
            "message": f"Drive {serial} FAILED"
                       + (" — was flagged high-risk in advance" if had_warning else " — no prior warning"),
            "level": "critical",
        })

    alerted_this_tick = False  # at most one AI alert per tick -- bounds latency/cost even on a noisy day
    for serial in list(newly_high_risk)[:5]:  # cap noisy days
        row = snapshot[snapshot["serial_number"] == serial].iloc[0]
        st.session_state.live_events.insert(0, {
            "date": str(as_of.date()),
            "message": f"Drive {serial} crossed into HIGH RISK ({row['risk']:.0%})",
            "level": "elevated",
        })
        if not alerted_this_tick and serial not in st.session_state.alerted_serials:
            alert_msg = _maybe_send_critical_alert(serial, row)
            st.session_state.alerted_serials.add(serial)
            alerted_this_tick = True
            if alert_msg:
                st.session_state.live_events.insert(0, {
                    "date": str(as_of.date()),
                    "message": f"{alert_msg} for drive {serial}",
                    "level": "elevated",
                })

    st.session_state.previously_high_risk = high_risk_now
    st.session_state.live_events = st.session_state.live_events[:40]  # keep the feed bounded

    st.session_state.live_history.append({
        "date": as_of,
        "avg_risk": float(snapshot["risk"].mean()),
        "high_risk_count": int((snapshot["tier"] == "critical").sum()),
        "elevated_count": int((snapshot["tier"] == "elevated").sum()),
        "cumulative_failures": st.session_state.cumulative_failures,
        "failures_with_prior_warning": st.session_state.failures_with_prior_warning,
        "active_drives": len(snapshot),
    })
    st.session_state.live_history = st.session_state.live_history[-120:]  # bound history length


def get_history_df() -> pd.DataFrame:
    if not st.session_state.live_history:
        return pd.DataFrame(columns=["date", "avg_risk", "high_risk_count", "elevated_count", "cumulative_failures", "active_drives"])
    return pd.DataFrame(st.session_state.live_history)
