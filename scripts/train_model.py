"""
train_model.py

Trains a RandomForestClassifier on the balanced drive-failure dataset and
saves everything the Streamlit dashboard (app.py) needs to make predictions.

Usage:
    python scripts/train_model.py

Outputs:
    model/model.pkl          -- the trained scikit-learn model (joblib format)
    model/feature_names.json -- ordered list of feature columns the model expects
    model/metrics.json       -- precision/recall/F1/confusion matrix, shown in the dashboard

Note on metrics: because failures are rare events, accuracy alone is a
misleading metric here -- a model that always predicts "healthy" would still
score ~98%+ accuracy. Report and read precision/recall/F1 instead.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed_balanced.csv"
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/feature_names.json"
METRICS_PATH = "model/metrics.json"
TEST_PREDICTIONS_PATH = "model/test_predictions.csv"

ID_COLUMNS = ["date", "serial_number", "model"]
TARGET_COLUMN = "failure"


def main():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in ID_COLUMNS + [TARGET_COLUMN]]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = dict(zip(feature_columns, model.feature_importances_.tolist()))

    # Raw test-set predictions -- used by cost_simulator.py to build the
    # threshold x horizon cost surface from real held-out outcomes, not a
    # made-up assumption about model behavior.
    pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba}).to_csv(TEST_PREDICTIONS_PATH, index=False)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "classification_report": report,
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["healthy (0)", "will fail soon (1)"],
            "feature_importances": importances,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "horizon_days": 7,  # must match HORIZON_DAYS in scripts/prepare_data.py
        }, f, indent=2)

    print(f"Trained on {len(X_train):,} rows, tested on {len(X_test):,} rows")
    print(f"Failure-class precision: {report['1']['precision']:.2f} | "
          f"recall: {report['1']['recall']:.2f} | f1: {report['1']['f1-score']:.2f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURES_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved test-set predictions -> {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
