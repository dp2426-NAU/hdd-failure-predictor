"""
provenance.py -- reads data/provenance.json (written by scripts/prepare_data.py)
so the dashboard's data-authenticity banner is always accurate, instead of a
hardcoded claim that goes stale the moment someone swaps in real data.
"""

import json
import os

PROVENANCE_PATH = "data/provenance.json"


def load_provenance() -> dict:
    if not os.path.exists(PROVENANCE_PATH):
        # prepare_data.py hasn't been run with the new tracking yet -- assume
        # the original bundled simulated sample.
        return {"source": "simulated_sample", "date_range": None, "n_failure_rows": None}
    with open(PROVENANCE_PATH) as f:
        return json.load(f)


def banner_html() -> str:
    p = load_provenance()
    if p.get("source") == "real_backblaze_download":
        date_range = p.get("date_range")
        range_str = f" spanning {date_range[0][:10]} to {date_range[1][:10]}" if date_range else ""
        return (
            '<div class="disclosure" style="border-left-color:#4caf7d;">'
            f"<b>✅ Running on real Backblaze drive-day data</b>{range_str}. "
            f"{p.get('n_failure_rows', '?')} genuine failure records in the balanced training set. "
            "\"Live mode\" on the Fleet Overview page still means a chronological <i>replay</i> of these "
            "real records, not a live sensor feed — see that page for details."
            "</div>"
        )
    return (
        '<div class="disclosure">'
        "<b>⚠️ Running on simulated sample data</b> — not real Backblaze telemetry. Built to Backblaze's "
        "real schema and calibrated to their real published 2025 failure rate (1.36%), but the individual "
        "records are synthetic. Run <code>scripts/prepare_data.py --raw-dir data/raw</code> on a real "
        "downloaded quarter and retrain to replace this with genuine data — see README.md."
        "</div>"
    )
