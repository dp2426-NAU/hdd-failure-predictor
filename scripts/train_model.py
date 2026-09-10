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

Note on the split -- grouped by drive, not by row: a failing drive
contributes multiple rows to the positive class (one per pre-failure day
within the early-warning window), and a drive's SMART readings are highly
autocorrelated day-to-day. A plain row-level train_test_split can put
different days from the SAME physical drive on both sides of the split,
letting the model partly recognize a specific drive's baseline rather than
generalize to an unseen one -- inflating held-out metrics. StratifiedGroupKFold
keeps every drive's rows entirely on one side, while still balancing the
fail/healthy ratio as closely as that grouping constraint allows.

Note on cross-fold variance -- report the range, not just one split: with
only ~43 real failing drives total, a single held-out fold's precision/
recall is itself a noisy estimate (which specific drives happen to land in
that one fold matters a lot at this sample size). Rather than present one
fold's numbers as if they were the whole picture, this script evaluates all
4 folds of the same StratifiedGroupKFold and saves the per-fold results and
their mean/std alongside the deployed model's own numbers (fold 0, the one
actually saved to model.pkl) -- so the honest uncertainty is disclosed in
model/metrics.json rather than hidden behind a single flattering split.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

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
    groups = df["serial_number"]

    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups=groups))

    # Evaluate all 4 folds first, purely to characterize how much the
    # metrics move depending on which drives land in the test set -- this
    # is diagnostic only and does not affect which model gets deployed.
    fold_results = []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        fold_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        fp = fold_model.predict(X.iloc[te_idx])
        fproba = fold_model.predict_proba(X.iloc[te_idx])[:, 1]
        frep = classification_report(y.iloc[te_idx], fp, output_dict=True, zero_division=0)
        fold_results.append({
            "fold": fold_i,
            "n_test_drives": int(groups.iloc[te_idx].nunique()),
            "n_failing_test_drives": int(groups.iloc[te_idx][y.iloc[te_idx] == 1].nunique()),
            "precision": frep["1"]["precision"],
            "recall": frep["1"]["recall"],
            "f1": frep["1"]["f1-score"],
            "roc_auc": float(roc_auc_score(y.iloc[te_idx], fproba)),
        })

    cv_summary = {
        "note": "Diagnostic only -- shows how much precision/recall/F1 move across the 4 "
                "StratifiedGroupKFold folds, given only ~43 real failing drives total. The "
                "deployed model below uses fold 0; this range is disclosed so fold 0's "
                "numbers aren't read as more certain than they are.",
        "folds": fold_results,
        "mean": {k: float(np.mean([f[k] for f in fold_results])) for k in ("precision", "recall", "f1", "roc_auc")},
        "std": {k: float(np.std([f[k] for f in fold_results])) for k in ("precision", "recall", "f1", "roc_auc")},
    }

    # Deployed model: fold 0, trained fresh (identical to the loop above,
    # kept as an explicit separate fit so it's clearly "the one that ships").
    train_idx, test_idx = splits[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_drives_train = groups.iloc[train_idx].nunique()
    n_drives_test = groups.iloc[test_idx].nunique()
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    assert not overlap, f"Group leakage: {len(overlap)} drives appear in both train and test"

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
    roc_auc = float(roc_auc_score(y_test, y_proba))

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
            "roc_auc": roc_auc,
            "feature_importances": importances,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_drives_train": int(n_drives_train),
            "n_drives_test": int(n_drives_test),
            "split_method": "StratifiedGroupKFold(n_splits=4) grouped by serial_number -- "
                             "no drive's rows appear on both sides of the split",
            "cross_fold_stability": cv_summary,
            "horizon_days": 7,  # must match HORIZON_DAYS in scripts/prepare_data.py
        }, f, indent=2)

    print(f"Trained on {len(X_train):,} rows ({n_drives_train:,} drives), "
          f"tested on {len(X_test):,} rows ({n_drives_test:,} drives) -- grouped split, zero drive overlap")
    print(f"Failure-class precision: {report['1']['precision']:.2f} | "
          f"recall: {report['1']['recall']:.2f} | f1: {report['1']['f1-score']:.2f} | roc_auc: {roc_auc:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")
    print(f"Cross-fold stability (4 folds) -- precision {cv_summary['mean']['precision']:.2f}±{cv_summary['std']['precision']:.2f}, "
          f"recall {cv_summary['mean']['recall']:.2f}±{cv_summary['std']['recall']:.2f}, "
          f"f1 {cv_summary['mean']['f1']:.2f}±{cv_summary['std']['f1']:.2f}, "
          f"roc_auc {cv_summary['mean']['roc_auc']:.2f}±{cv_summary['std']['roc_auc']:.2f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURES_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved test-set predictions -> {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
