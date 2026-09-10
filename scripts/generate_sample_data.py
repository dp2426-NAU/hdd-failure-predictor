"""
generate_sample_data.py

Creates a SMALL, REALISTIC STAND-IN dataset that mirrors the real Backblaze
"Drive Stats" schema (https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data),
so you can develop and demo the full pipeline offline before plugging in the
real quarterly download.

Why this exists instead of the real data:
  The real dataset ships as ~1-1.3 GB quarterly ZIPs (~10 GB uncompressed) hosted
  directly by Backblaze. This project was scaffolded in a sandboxed environment
  that cannot reach arbitrary external file hosts, so the real ZIP could not be
  downloaded here. This script instead builds a small sample that:
    - uses the REAL column names from Backblaze's published schema
    - uses the REAL 2025 annualized failure rate (1.36%, per Backblaze's 2025
      Drive Stats report) to calibrate how often a simulated drive fails
    - gives failing drives realistically elevated SMART attributes (the same
      attributes Backblaze's own engineering research has flagged as predictive:
      reallocated sectors, reported uncorrectable errors, command timeout,
      current pending sector count) so the model has real signal to learn from

This is clearly a SIMULATED sample, not real Backblaze telemetry. Swap it for
the real thing by running prepare_data.py against the real download -- see
README.md, section "Using the real dataset".

IMPORTANT -- about the failure rate used here: Backblaze's real 2025 annualized
failure rate is 1.36% (Backblaze, 2026). At that true rate, a demo-sized sample
of a few thousand drives observed for a few weeks would realistically contain
only a handful of failures -- not enough to train or evaluate a classifier
offline. So this generator deliberately OVERSAMPLES the failure class (see
DEMO_FAIL_FRACTION below) purely so the pipeline has enough positive examples
to demonstrate end-to-end. This is a synthetic-data convenience, not a claim
about the real failure rate -- report the real 1.36% figure in your written
proposal, and note that a real quarterly download (with millions of drive-days)
would not need this oversampling at all.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_DRIVES = 3000            # number of unique physical drives simulated
DAYS_OBSERVED = 20         # days of history per drive (drive-days = rows)
REAL_ANNUAL_AFR = 0.0136   # Backblaze 2025 annualized failure rate (Backblaze, 2026) -- for reference/reporting only
DEMO_FAIL_FRACTION = 0.10  # oversampled rate used ONLY to make this demo dataset trainable

DRIVE_MODELS = [
    "ST16000NM001G",  # Seagate 16TB - a real model tracked in Backblaze's reports
    "WUH721816ALE6L",  # WDC/HGST 16TB
    "HUH721212ALE600",  # HGST 12TB
]


SUDDEN_FAILURE_FRACTION = 0.25  # ~1 in 4 failures gives little/no advance SMART warning -- realistic,
                                 # and an honest limitation to report: not every failure is predictable.
NOISY_HEALTHY_FRACTION = 0.06   # some healthy drives get a one-off spurious elevated reading (real SMART
                                 # data is noisy -- without this, a classifier trained on this data looks
                                 # implausibly perfect, which is itself a red flag worth avoiding.


def simulate_drive(serial_id: int):
    """Simulate one drive's ~DAYS_OBSERVED daily SMART readings."""
    model = RNG.choice(DRIVE_MODELS)
    capacity_bytes = 16_000_900_661_248 if "16" in model or "1816" in model else 12_000_138_625_024

    # Deliberately oversampled for demo trainability -- see module docstring.
    will_fail = RNG.random() < DEMO_FAIL_FRACTION
    fail_day = RNG.integers(DAYS_OBSERVED // 2, DAYS_OBSERVED) if will_fail else None
    is_sudden_failure = will_fail and (RNG.random() < SUDDEN_FAILURE_FRACTION)
    is_noisy_healthy = (not will_fail) and (RNG.random() < NOISY_HEALTHY_FRACTION)
    noisy_day = RNG.integers(0, DAYS_OBSERVED) if is_noisy_healthy else None

    power_on_base = RNG.integers(500, 45000)  # hours already on the clock
    rows = []
    for day in range(DAYS_OBSERVED):
        is_failure_day = will_fail and day == fail_day
        # Degradation ramps up in the days just before a real failure event --
        # unless this is a "sudden" failure, which gives almost no advance warning
        # (mirrors real drives that fail from a sudden mechanical/electrical event).
        days_to_fail = (fail_day - day) if will_fail else 999
        if is_sudden_failure:
            stress = 9 if days_to_fail <= 1 else 0
        else:
            stress = max(0, 10 - days_to_fail) if will_fail else 0

        # Random noise floor + occasional spurious spike on an otherwise-healthy
        # drive, so "elevated reading" isn't a perfect proxy for "about to fail."
        noise_spike = 6 if (is_noisy_healthy and day == noisy_day) else 0

        smart_5_reallocated = max(0, int(RNG.normal(stress * 8 + noise_spike, 2.5)))
        smart_187_uncorrectable = max(0, int(RNG.poisson(stress * 0.6 + noise_spike * 0.3)))
        smart_188_cmd_timeout = max(0, int(RNG.poisson(stress * 1.2 + noise_spike * 0.2)))
        smart_197_pending = max(0, int(RNG.normal(stress * 3 + noise_spike * 0.5, 1.3)))
        smart_198_offline_uncorrectable = max(0, int(RNG.poisson(stress * 0.4)))
        smart_9_power_on_hours = power_on_base + day * 24
        smart_194_temp_c = int(np.clip(RNG.normal(32 + stress * 0.4, 3.5), 18, 55))

        rows.append({
            "date": pd.Timestamp("2025-10-01") + pd.Timedelta(days=day),
            "serial_number": f"SIM{serial_id:06d}",
            "model": model,
            "capacity_bytes": capacity_bytes,
            "failure": int(is_failure_day),
            "smart_5_raw": smart_5_reallocated,
            "smart_9_raw": smart_9_power_on_hours,
            "smart_187_raw": smart_187_uncorrectable,
            "smart_188_raw": smart_188_cmd_timeout,
            "smart_194_raw": smart_194_temp_c,
            "smart_197_raw": smart_197_pending,
            "smart_198_raw": smart_198_offline_uncorrectable,
        })
        if is_failure_day:
            break  # a drive stops reporting once it fails
    return rows


def main():
    all_rows = []
    for serial_id in range(N_DRIVES):
        all_rows.extend(simulate_drive(serial_id))

    df = pd.DataFrame(all_rows)
    out_path = "data/sample_drive_stats.csv"
    df.to_csv(out_path, index=False)

    n_failure_rows = df["failure"].sum()
    n_drives_failed = df.loc[df["failure"] == 1, "serial_number"].nunique()
    print(f"Wrote {len(df):,} drive-day rows for {N_DRIVES:,} simulated drives to {out_path}")
    print(f"Simulated failures: {n_drives_failed:,} drives ({n_failure_rows:,} failure=1 rows)")
    print(f"Demo failure fraction used: {n_drives_failed / N_DRIVES:.2%} "
          f"(oversampled for trainability -- real annual AFR is {REAL_ANNUAL_AFR:.2%}, see docstring)")


if __name__ == "__main__":
    main()
