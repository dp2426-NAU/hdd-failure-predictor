"""
train_survival_model.py

Fits a Cox Proportional Hazards model (via lifelines) on the survival-analysis
dataset -- this is the "how long until failure" model, distinct from
train_model.py's "will it fail" classifier. Also fits an overall Kaplan-Meier
curve for the fleet as a reference baseline shown on the dashboard.

Evaluation metric is the concordance index (C-index), the standard metric for
survival models -- 0.5 means no better than random ranking of who fails
first, 1.0 means perfect ranking. Report this, not accuracy.

Usage:
    python scripts/train_survival_model.py
"""

import json

import joblib
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

DATA_PATH = "data/survival_data.csv"
COX_MODEL_PATH = "model/survival_model.pkl"
KM_CURVE_PATH = "model/km_curve.json"
METRICS_PATH = "model/survival_metrics.json"

ID_COLUMNS = ["serial_number", "model", "duration", "event_observed"]


def main():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in ID_COLUMNS]

    cox_df = df[feature_columns + ["duration", "event_observed"]].copy()

    cph = CoxPHFitter(penalizer=0.1)  # small penalty for stability with correlated SMART features
    cph.fit(cox_df, duration_col="duration", event_col="event_observed")

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], event_observed=df["event_observed"])
    km_df = kmf.survival_function_.reset_index()
    km_df.columns = ["day", "survival_probability"]

    joblib.dump({"model": cph, "feature_columns": feature_columns}, COX_MODEL_PATH)
    km_df.to_json(KM_CURVE_PATH, orient="records")

    metrics = {
        "concordance_index": float(cph.concordance_index_),
        "n_drives": len(df),
        "n_events": int(df["event_observed"].sum()),
        "median_fleet_survival_days": float(kmf.median_survival_time_) if kmf.median_survival_time_ != float("inf") else None,
        "hazard_ratios": json.loads(cph.hazard_ratios_.to_json()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Trained Cox model on {len(df):,} drives ({metrics['n_events']} events)")
    print(f"Concordance index: {metrics['concordance_index']:.3f} (0.5 = random, 1.0 = perfect ranking)")
    print(f"Fleet median survival: {metrics['median_fleet_survival_days']} days" if metrics['median_fleet_survival_days'] else
          "Fleet median survival: not reached within observation window (>50% of drives still healthy)")
    print("Hazard ratios (>1 = raises failure hazard, <1 = lowers it):")
    for k, v in metrics["hazard_ratios"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
