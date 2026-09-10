"""
scripts/check_and_alert.py -- the genuinely automated background check,
run on a real schedule by .github/workflows/fleet-check.yml (GitHub
Actions -- free and unlimited on a public repo, no card required).

Unlike live_feed.py's in-browser alert (which only fires while someone has
the Streamlit app open and is stepping through the replay), this script
runs headlessly and independently, on its own schedule, whether or not
anyone is looking at the dashboard. This is the piece that actually earns
the word "automated" -- see README.md for the full honest explanation of
why the in-browser version alone didn't.

State (which simulated day the check is on, which drives are currently
flagged critical, and running totals) persists in automation/state.json,
which this script updates and the workflow commits back to the repo after
each run -- so each scheduled run picks up exactly where the last one left
off, and loops back to the start of the quarter when it reaches the end
(same behavior as live_feed.py's replay, so it keeps running indefinitely
without manual intervention).

Uses report_generator.py's rule-based (no external API, free) summaries --
no LLM, no per-call cost, just Python string templates over real model +
SHAP output -- and email_alerts.py's SMTP sender, both already used by the
in-browser alert.

Usage:
    python scripts/check_and_alert.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import shap

# email_alerts.py / provenance.py / report_generator.py live in the project
# root, one level up from scripts/ -- running `python scripts/check_and_alert.py`
# only puts scripts/ itself on sys.path, so add the root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import email_alerts
import provenance
import report_generator

STATE_PATH = "automation/state.json"
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/feature_names.json"
SAMPLE_DATA_PATH = "data/sample_drive_stats.csv"
REAL_DATA_PATH = "data/real_drive_stats.csv"

RISK_HIGH = 0.66
DAYS_PER_CHECK = 2  # simulated days advanced per run -- tuned so an hourly GitHub Actions
                     # schedule replays the full ~90-day quarter in about 2 days


def _resolve_data_path() -> str:
    p = provenance.load_provenance()
    if p.get("source") == "real_backblaze_download" and os.path.exists(REAL_DATA_PATH):
        return REAL_DATA_PATH
    return SAMPLE_DATA_PATH


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "day_index": -1,  # -1 so the very first run starts at day 0
        "previously_high_risk": [],
        "cumulative_failures": 0,
        "cumulative_warned": 0,
    }


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _shap_contributions(explainer, row: pd.Series, feature_columns: list) -> list:
    X_input = pd.DataFrame([row[feature_columns].to_dict()])[feature_columns]
    shap_raw = explainer.shap_values(X_input)
    if isinstance(shap_raw, list):
        values = shap_raw[1][0]
    elif shap_raw.ndim == 3:
        values = shap_raw[0, :, 1]
    else:
        values = shap_raw[0]
    return list(zip(feature_columns, [float(v) for v in values]))


def main():
    df = pd.read_csv(_resolve_data_path(), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].unique())

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)
    explainer = shap.TreeExplainer(model)

    state = _load_state()
    previously_high_risk = set(state["previously_high_risk"])

    start_idx = state["day_index"] + 1
    looped = False
    if start_idx > len(dates) - 1:
        # Reached the end of the quarter -- loop back to the start, same
        # behavior as live_feed.py's replay, so this keeps running
        # indefinitely without anyone needing to reset it by hand.
        start_idx = 0
        previously_high_risk = set()
        state["cumulative_failures"] = 0
        state["cumulative_warned"] = 0
        looped = True
    end_idx = min(start_idx + DAYS_PER_CHECK - 1, len(dates) - 1)

    newly_critical_events = []
    failure_events = []

    for idx in range(start_idx, end_idx + 1):
        as_of = dates[idx]
        window = df[df["date"] <= as_of]
        snapshot = window.sort_values("date").groupby("serial_number", as_index=False).last()
        X = snapshot[feature_columns].fillna(0)
        snapshot["risk"] = model.predict_proba(X)[:, 1]

        high_risk_now = set(snapshot.loc[snapshot["risk"] >= RISK_HIGH, "serial_number"])
        newly_high_risk = high_risk_now - previously_high_risk

        todays_failures = snapshot.loc[(snapshot["date"] == as_of) & (snapshot["failure"] == 1), "serial_number"]
        for serial in todays_failures:
            had_warning = serial in previously_high_risk
            state["cumulative_failures"] += 1
            if had_warning:
                state["cumulative_warned"] += 1
            failure_events.append({"date": str(as_of.date()), "serial": serial, "had_warning": had_warning})

        for serial in list(newly_high_risk)[:5]:  # cap a noisy day, same as live_feed.py
            row = snapshot[snapshot["serial_number"] == serial].iloc[0]
            contribs = _shap_contributions(explainer, row, feature_columns)
            summary = report_generator.incident_summary(serial, float(row["risk"]), contribs)
            newly_critical_events.append({
                "date": str(as_of.date()), "serial": serial,
                "risk_pct": float(row["risk"]), "summary": summary,
            })

        previously_high_risk = high_risk_now

    period_label = f"{dates[start_idx].date()} to {dates[end_idx].date()}"
    if looped:
        period_label += " (looped back to start of quarter)"

    final_window = df[df["date"] <= dates[end_idx]]
    final_snapshot = final_window.sort_values("date").groupby("serial_number", as_index=False).last()

    subject, body = report_generator.status_report(
        period_label=period_label,
        drives_monitored=len(final_snapshot),
        newly_critical=newly_critical_events,
        failures=failure_events,
        cumulative_failures=state["cumulative_failures"],
        cumulative_warned=state["cumulative_warned"],
    )

    sent = email_alerts.send_alert_email(subject, body)
    print(f"Checked {period_label}: {len(newly_critical_events)} newly critical, "
          f"{len(failure_events)} failures. Email sent: {sent}")
    if not email_alerts.is_configured():
        print("(email not configured -- set SMTP_HOST/PORT/USER/PASSWORD + ALERT_EMAIL_TO "
              "as GitHub Actions secrets to actually send. State still updates below.)")

    state["day_index"] = end_idx
    state["previously_high_risk"] = sorted(previously_high_risk)
    _save_state(state)


if __name__ == "__main__":
    main()
