"""
prepare_survival_data.py

Builds a ONE-ROW-PER-DRIVE survival-analysis dataset, which is a genuinely
different framing from prepare_data.py's one-row-per-drive-DAY classification
dataset. Instead of "will this drive fail" (yes/no), this supports "how much
longer will this drive likely run" (a remaining-useful-life estimate) --
the standard survival-analysis framing used in real industrial predictive
maintenance, via lifelines' Cox Proportional Hazards model.

For each drive:
    duration        -- how many days it was observed for
    event_observed  -- 1 if it failed during observation, 0 if it was still
                        healthy when observation ended ("censored" in
                        survival-analysis terms -- we don't know when/if it
                        would have failed after that)
    <SMART columns>  -- covariates, taken from the drive's LAST observed
                        reading (a simplification: a proper time-varying Cox
                        model would use the full history, which lifelines
                        also supports but adds real complexity for a course
                        project -- this simplification is disclosed here and
                        in the dashboard, not hidden)

Usage:
    python scripts/prepare_survival_data.py                     # bundled sample
    python scripts/prepare_survival_data.py --raw-dir data/raw  # real data
"""

import argparse
import glob
import os

import pandas as pd

FEATURE_COLUMNS = [
    "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw",
]


def load_raw(raw_dir, sample_path, models=None):
    if raw_dir:
        csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs in {raw_dir}")
        # Same memory concern as prepare_data.py: a real quarterly download
        # is too wide/long to read in full before filtering. Keep only the
        # columns this pipeline uses (plus failure/date), and filter to
        # --models while reading, not after concatenation.
        wanted_cols = {"date", "serial_number", "model", "failure"} | set(FEATURE_COLUMNS)
        frames = []
        for f in csv_files:
            header = pd.read_csv(f, nrows=0).columns
            usecols = [c for c in header if c in wanted_cols]
            chunk = pd.read_csv(f, usecols=usecols, low_memory=False)
            if models:
                chunk = chunk[chunk["model"].isin(models)]
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(sample_path)


def build_survival_table(df: pd.DataFrame, models=None) -> pd.DataFrame:
    if models:
        df = df[df["model"].isin(models)].copy()

    df["date"] = pd.to_datetime(df["date"])
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    rows = []
    for serial, g in df.sort_values("date").groupby("serial_number"):
        g = g.dropna(subset=available_features, how="all")
        if g.empty:
            continue
        first_day, last_day = g["date"].iloc[0], g["date"].iloc[-1]
        duration = max((last_day - first_day).days + 1, 1)
        event = int((g["failure"] == 1).any())
        last_reading = g.iloc[-1]
        row = {"serial_number": serial, "model": g["model"].iloc[-1],
               "duration": duration, "event_observed": event}
        for col in available_features:
            row[col] = pd.to_numeric(last_reading.get(col), errors="coerce")
        rows.append(row)

    out = pd.DataFrame(rows).dropna(subset=available_features)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--sample-path", default="data/sample_drive_stats.csv")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default="data/survival_data.csv")
    args = parser.parse_args()

    df = load_raw(args.raw_dir, args.sample_path, args.models)
    survival_df = build_survival_table(df, args.models)
    survival_df.to_csv(args.out, index=False)

    n_events = survival_df["event_observed"].sum()
    print(f"Wrote {len(survival_df):,} drives to {args.out}")
    print(f"Failure events: {n_events:,} ({n_events/len(survival_df):.1%}) | "
          f"Censored (still healthy at end of window): {len(survival_df) - n_events:,}")


if __name__ == "__main__":
    main()
