"""
prepare_live_feed_data.py

Builds the dataset that powers Fleet Overview's "live-feel" chronological
replay (live_feed.py) -- a day-by-day drive-day timeline, in the SAME schema
generate_sample_data.py produces (date, serial_number, model, capacity_bytes,
failure, + the 7 SMART feature columns).

Why this is a separate script from prepare_data.py / prepare_survival_data.py:
those two build TRAINING sets (an early-warning-labeled/balanced table, and a
one-row-per-drive survival table) -- neither is a full chronological
drive-day timeline, which is what the replay engine needs to advance
day-by-day and show a drive's history.

Without this script, live_feed.py has nothing to fall back on except the
bundled simulated sample -- meaning Fleet Overview would keep replaying fake
drives even after prepare_data.py / train_model.py were retrained on real
data. Running this script (and re-running the app) is what makes Fleet
Overview's replay switch to genuine drive-days, matching the Home page
banner.

The full real quarter (millions of drive-days) is too large to load in the
deployed app or commit to git, so this keeps ALL real failing drives (so the
event feed and "caught with advance warning" mechanic stay meaningful) plus a
random sample of healthy drives, capped at --max-drives (default 3,000 --
the same fleet size the bundled demo used), across the FULL real observation
window (not truncated to 20 days like the demo).

Usage:
    python scripts/prepare_live_feed_data.py --raw-dir data/raw --models ST16000NM001G
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw",
]
ID_COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def load_raw(raw_dir: str, models: list[str] | None, max_drives: int, seed: int) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    wanted_cols = set(ID_COLUMNS) | set(FEATURE_COLUMNS)
    frames = []
    for f in csv_files:
        header = pd.read_csv(f, nrows=0).columns
        usecols = [c for c in header if c in wanted_cols]
        chunk = pd.read_csv(f, usecols=usecols, low_memory=False)
        if models:
            chunk = chunk[chunk["model"].isin(models)]
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    failing_serials = set(df.loc[df["failure"] == 1, "serial_number"].unique())
    all_serials = df["serial_number"].unique()
    healthy_serials = np.array([s for s in all_serials if s not in failing_serials])

    n_healthy_to_keep = max(0, max_drives - len(failing_serials))
    rng = np.random.default_rng(seed)
    if n_healthy_to_keep < len(healthy_serials):
        keep_healthy = set(rng.choice(healthy_serials, size=n_healthy_to_keep, replace=False))
    else:
        keep_healthy = set(healthy_serials)

    keep_serials = failing_serials | keep_healthy
    return df[df["serial_number"].isin(keep_serials)].sort_values(["serial_number", "date"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Folder of real Backblaze daily CSVs")
    parser.add_argument("--models", nargs="*", default=None, help="Drive model(s) to keep, e.g. ST16000NM001G")
    parser.add_argument("--max-drives", type=int, default=3000,
                         help="Cap on total drives kept (all failing drives are always kept)")
    parser.add_argument("--out", default="data/real_drive_stats.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.raw_dir, args.models, args.max_drives, args.seed)

    keep_cols = ID_COLUMNS + FEATURE_COLUMNS
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    df.to_csv(args.out, index=False)

    n_drives = df["serial_number"].nunique()
    n_failing = df.loc[df["failure"] == 1, "serial_number"].nunique()
    print(f"Wrote {len(df):,} drive-day rows for {n_drives:,} drives ({n_failing:,} failing) to {args.out}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    main()
