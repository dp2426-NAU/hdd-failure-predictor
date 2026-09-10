"""
prepare_data.py

Turns raw Backblaze-schema drive-day rows into a small, balanced training set.

Works on two kinds of input:
  1. The bundled sample: data/sample_drive_stats.csv (default, no args needed)
  2. A REAL Backblaze quarterly download: a folder of daily CSVs, e.g.
     data/raw/2025-10-01.csv, data/raw/2025-10-02.csv, ...
     Download these yourself from:
     https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

Usage:
    python scripts/prepare_data.py                     # uses the bundled sample
    python scripts/prepare_data.py --raw-dir data/raw   # uses real downloaded CSVs
    python scripts/prepare_data.py --raw-dir data/raw --models ST16000NM001G

IMPORTANT -- this builds an EARLY-WARNING label, not a same-day diagnosis:
    A naive approach would label a drive's row on the day it fails as the
    positive class. Don't do that -- by definition, a drive's SMART readings
    on the day it actually fails already look bad, so a model trained that
    way isn't predicting anything, it's just recognizing "this looks like a
    failure event," which is trivial and not useful (you don't need ML to
    tell you a drive that's failing today is failing today). That kind of
    same-day leakage is also exactly the thing a professor checking for
    academic rigor would flag.

    Instead, this script labels each PRE-failure drive-day row as positive if
    the drive's actual failure occurs within the next HORIZON_DAYS days, and
    excludes the failure day itself from training entirely. That's a genuine
    forward-looking prediction task: "given today's SMART readings, will this
    drive fail in the next week" -- not "does today look like a failure."

Output:
    data/processed_balanced.csv -- all early-warning-positive rows + an equal
    random sample of negative rows, keeping only the SMART feature columns
    the model uses. This "undersampling" approach is a standard, defensible
    way to handle a rare-event target like drive failure (~1.36% annualized)
    -- say so plainly in your report rather than reporting misleadingly high
    raw accuracy.
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import pandas as pd

# The SMART attributes prior research and Backblaze's own engineering blog
# flag as predictive of impending failure. Real Backblaze files carry many
# more SMART columns than this; we only use the ones we need.
FEATURE_COLUMNS = [
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_9_raw",    # Power-On Hours
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_194_raw",  # Temperature (Celsius)
    "smart_197_raw",  # Current Pending Sector Count
    "smart_198_raw",  # Offline Uncorrectable Sector Count
]
TARGET_COLUMN = "failure"  # kept as the column name for compatibility with
                            # everything downstream (train_model.py, the
                            # dashboards) -- but see HORIZON_DAYS below: its
                            # MEANING is now "fails within the horizon", not
                            # "failed today". A raw same-day failure flag from
                            # the source data is never used as a training row.
ID_COLUMNS = ["date", "serial_number", "model"]
HORIZON_DAYS = 7  # the early-warning window: "will this drive fail in the next N days"


def load_raw(raw_dir: str | None, sample_path: str, models: list[str] | None = None) -> pd.DataFrame:
    if raw_dir:
        csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {raw_dir}. Download and unzip a quarter from "
                "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data first."
            )
        print(f"Loading {len(csv_files)} real Backblaze daily CSVs from {raw_dir} ...")
        # A real quarterly download is ~90 files x ~200 SMART/metadata columns
        # (multiple GB uncompressed). Reading full-width frames and only
        # filtering columns/models after concatenation runs out of memory on
        # an ordinary machine. Restrict each file to the columns this
        # pipeline actually uses -- and, when --models is given, to those
        # models -- while reading, so memory stays proportional to the data
        # we keep rather than Backblaze's full raw schema.
        wanted_cols = set(ID_COLUMNS) | set(FEATURE_COLUMNS) | {TARGET_COLUMN}
        frames = []
        for f in csv_files:
            header = pd.read_csv(f, nrows=0).columns
            usecols = [c for c in header if c in wanted_cols]
            chunk = pd.read_csv(f, usecols=usecols, low_memory=False)
            if models:
                chunk = chunk[chunk["model"].isin(models)]
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True)
    print(f"Loading bundled sample from {sample_path} ...")
    return pd.read_csv(sample_path)


def build_early_warning_labels(df: pd.DataFrame, available_features: list[str]) -> pd.DataFrame:
    """For each drive, find its failure date (if any), drop the failure-day
    row itself (that's an outcome, not a feature row), and label every
    remaining row 1 if it falls within HORIZON_DAYS of that failure, else 0.
    Drives that never fail contribute only label=0 rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    labeled_frames = []
    for serial, g in df.groupby("serial_number"):
        g = g.sort_values("date")
        fail_rows = g[g["failure"] == 1]
        if fail_rows.empty:
            g = g.copy()
            g["label"] = 0
            labeled_frames.append(g)
            continue

        fail_date = fail_rows["date"].iloc[0]
        pre_failure = g[g["date"] < fail_date].copy()
        if pre_failure.empty:
            continue  # nothing to learn from (failed on its first observed day)
        days_before = (fail_date - pre_failure["date"]).dt.days
        pre_failure["label"] = (days_before <= HORIZON_DAYS).astype(int)
        labeled_frames.append(pre_failure)

    return pd.concat(labeled_frames, ignore_index=True) if labeled_frames else df.iloc[0:0]


def build_balanced_set(df: pd.DataFrame, models: list[str] | None, seed: int = 42) -> pd.DataFrame:
    if models:
        df = df[df["model"].isin(models)].copy()
        print(f"Filtered to models {models}: {len(df):,} rows remain")

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        print(f"Note: columns not present in this data and skipped: {sorted(missing)}")

    keep_cols = [c for c in ID_COLUMNS if c in df.columns] + available_features + ["failure"]
    df = df[keep_cols].dropna(subset=available_features, how="all")
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=available_features)

    labeled = build_early_warning_labels(df, available_features)
    print(f"Early-warning labeling (horizon = {HORIZON_DAYS} days): "
          f"{len(df) - len(labeled):,} failure-day rows excluded from training as outcomes, not predictors")

    positives = labeled[labeled["label"] == 1].drop(columns=["failure"]).rename(columns={"label": TARGET_COLUMN})
    negatives = labeled[labeled["label"] == 0].drop(columns=["failure"]).rename(columns={"label": TARGET_COLUMN})
    if len(positives) == 0:
        raise ValueError(
            f"No rows fall within the {HORIZON_DAYS}-day early-warning window -- check --models, "
            "the input data, or lower HORIZON_DAYS."
        )

    negative_sample = negatives.sample(n=min(len(negatives), len(positives)), random_state=seed)
    balanced = pd.concat([positives, negative_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Balanced set: {len(positives):,} early-warning-positive rows + {len(negative_sample):,} "
          f"negative rows = {len(balanced):,} total")
    return balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None, help="Folder of real Backblaze daily CSVs")
    parser.add_argument("--sample-path", default="data/sample_drive_stats.csv")
    parser.add_argument("--models", nargs="*", default=None, help="Drive model(s) to keep, e.g. ST16000NM001G")
    parser.add_argument("--out", default="data/processed_balanced.csv")
    args = parser.parse_args()

    df = load_raw(args.raw_dir, args.sample_path, args.models)
    balanced = build_balanced_set(df, args.models)
    balanced.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    # Record where this data actually came from, so the dashboard can show an
    # accurate banner instead of a hardcoded claim that goes stale the moment
    # someone swaps in real data (or vice versa).
    provenance = {
        "source": "real_backblaze_download" if args.raw_dir else "simulated_sample",
        "raw_dir": args.raw_dir,
        "models_filtered": args.models,
        "n_failure_rows": int((balanced["failure"] == 1).sum()),
        "n_healthy_rows": int((balanced["failure"] == 0).sum()),
        "date_range": [str(df["date"].min()), str(df["date"].max())] if "date" in df.columns else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open("data/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Wrote data/provenance.json (source: {provenance['source']})")


if __name__ == "__main__":
    main()
