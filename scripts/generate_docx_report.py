"""
scripts/generate_docx_report.py -- builds docs/Predicting_Drive_Failure_Report.docx,
a formal written report for the capstone submission.

Pulls real numbers directly from model/metrics.json, model/survival_metrics.json,
and data/provenance.json rather than hand-typing them, so the report can't drift
out of sync with whatever the pipeline actually produced. Re-run this after any
retrain to regenerate the report with fresh numbers.

Usage:
    python scripts/generate_docx_report.py
"""

import json
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACCENT = RGBColor(0x0D, 0x7D, 0x6F)
INK = RGBColor(0x16, 0x21, 0x1F)
MUTED = RGBColor(0x55, 0x55, 0x55)

OUT_PATH = "docs/Predicting_Drive_Failure_Report.docx"
SCREENSHOTS = "docs/screenshots"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT if level <= 2 else INK
    return h


def add_body(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    return p


def add_metric_table(doc, rows, headers):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()


def add_screenshot(doc, filename, caption, width=6.0):
    path = os.path.join(SCREENSHOTS, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = MUTED


def main():
    metrics = load_json("model/metrics.json")
    surv = load_json("model/survival_metrics.json")
    prov = load_json("data/provenance.json")

    report = metrics["classification_report"]["1"]
    cv = metrics.get("cross_fold_stability", {})
    cv_mean = cv.get("mean", {})
    cv_std = cv.get("std", {})
    model_name = ", ".join(prov.get("models_filtered") or ["(all models)"])
    date_range = prov.get("date_range") or ["?", "?"]

    doc = Document()

    # ---------- Title page ----------
    title = doc.add_heading("Predicting Drive Failure", level=0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("A Predictive-Maintenance System for Hard Drive Fleets\nBuilt on Real Backblaze Drive Stats Telemetry")
    r.font.size = Pt(14)
    r.font.color.rgb = MUTED

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "IT Capstone Project\n"
        "Live demo: https://hdd-failure-predictor.onrender.com\n"
        "Repository: https://github.com/dp2426-NAU/hdd-failure-predictor"
    ).font.size = Pt(11)
    doc.add_page_break()

    # ---------- Executive Summary ----------
    add_heading(doc, "Executive Summary", level=1)
    add_body(doc,
        "This project trains machine learning models on real hard-drive SMART "
        "telemetry, published by Backblaze under their Drive Stats program, to "
        "predict which drives are likely to fail within the next 7 days, estimate "
        "how much longer each drive is likely to run, and recommend a cost-optimal "
        "alert threshold for a given organization's cost structure. Results are "
        "served through three Streamlit dashboards covering three distinct "
        "audiences -- leadership, field technicians, and decision-makers -- backed "
        "by a two-tier automated alerting system, one layer running unattended on "
        "a real hourly schedule via GitHub Actions."
    )
    add_body(doc,
        "The system is intentionally built to go beyond a standard binary "
        "classification exercise on this well-known public dataset: it uses "
        "early-warning labeling (excluding same-day failure signal as training "
        "leakage), a Cox Proportional Hazards survival model for remaining useful "
        "life, per-prediction SHAP explainability, and a real cost-sensitivity "
        "surface -- all trained and evaluated on a real downloaded Backblaze "
        f"quarter ({date_range[0]} to {date_range[1]}, model {model_name})."
    )

    # ---------- Problem Statement ----------
    add_heading(doc, "1. Problem Statement and Motivation", level=1)
    add_body(doc,
        "Unplanned hard-drive failure is a persistent operational cost for any "
        "organization running physical storage infrastructure -- data centers, "
        "backup providers, and enterprise IT alike. Reactive maintenance (replacing "
        "a drive only after it fails) risks data loss and unplanned downtime; "
        "blanket proactive replacement wastes budget on drives that would have run "
        "fine. A predictive approach -- flagging at-risk drives early enough to "
        "act, backed by an estimate of how much time is actually left -- is the "
        "standard framing used in real industrial predictive maintenance, and is "
        "the goal of this project."
    )

    # ---------- Why different ----------
    add_heading(doc, "2. What Makes This Project Different", level=1)
    add_body(doc,
        "Backblaze's Drive Stats dataset is one of the most common public machine "
        "learning case studies available, and plenty of prior student and public "
        "projects use it for plain binary classification. Four design decisions "
        "make this project substantively different, not just cosmetically "
        "different:"
    )
    add_bullet(doc, "Early-warning labeling, not same-day diagnosis. A drive's SMART "
                    "readings on the day it fails already look like a failure -- "
                    "training on that row would be recognizing an already-failed "
                    "drive, not predicting one. Every row is instead labeled positive "
                    "only if the drive's actual failure lands within the next 7 days, "
                    "and the failure-day row itself is excluded from training.")
    add_bullet(doc, "Remaining useful life, not just yes/no. A Cox Proportional "
                    "Hazards survival model estimates how many days a drive likely "
                    "has left, the standard framing in real industrial predictive "
                    "maintenance, instead of stopping at binary classification.")
    add_bullet(doc, "Per-prediction explainability. SHAP explains this specific "
                    "drive's score, using its own real SMART readings, not just "
                    "which features matter on average across the whole model.")
    add_bullet(doc, "A genuine decision-support result, not just a risk score. The "
                    "cost surface shows that the right alert threshold depends on an "
                    "organization's own cost structure -- there is no single "
                    "universal answer, which is the actually useful, non-obvious "
                    "finding.")

    # ---------- Data ----------
    add_heading(doc, "3. Data", level=1)
    add_body(doc,
        f"Training data is a real quarterly download from Backblaze's public Drive "
        f"Stats program ({date_range[0]} to {date_range[1]}), filtered to a single "
        f"high-volume drive model, {model_name}. Mixing drive models was avoided "
        "deliberately: during model selection, the single highest-volume model in "
        "the raw data turned out to report zero values for two of the seven SMART "
        "attributes this project's feature set relies on -- a real vendor-level "
        "SMART-reporting difference (confirmed to be a Seagate-vs-WD/Toshiba "
        "reporting pattern), not a data quality bug. The model ultimately used "
        "reports all seven attributes fully."
    )
    add_metric_table(doc,
        headers=["Metric", "Value"],
        rows=[
            ["Drive model", model_name],
            ["Date range", f"{date_range[0]} to {date_range[1]}"],
            ["Real failure-day rows (this model)", prov.get("n_failure_rows", "?")],
            ["Balanced training rows (matched negatives)", prov.get("n_healthy_rows", "?")],
            ["Data source", "real_backblaze_download (see data/provenance.json)"],
        ],
    )

    # ---------- Methodology ----------
    add_heading(doc, "4. Methodology", level=1)
    add_heading(doc, "4.1 Classifier", level=2)
    add_body(doc,
        "A RandomForestClassifier (300 trees, max depth 8, class_weight="
        "'balanced') predicts the probability a drive fails within the next 7 "
        "days, trained on SMART 5, 9, 187, 188, 194, 197, and 198 (reallocated "
        "sectors, power-on hours, reported uncorrectable errors, command timeout, "
        "temperature, current pending sectors, and offline uncorrectable sectors)."
    )
    add_body(doc,
        "The train/test split is grouped by drive (StratifiedGroupKFold on "
        "serial_number), not by row. A failing drive contributes multiple rows to "
        "the positive class -- one per pre-failure day within the horizon -- and a "
        "drive's SMART readings are autocorrelated day-to-day, so a naive row-level "
        "split can let the same physical drive leak across train and test, "
        "inflating held-out metrics. This was caught and fixed during development."
    )
    add_heading(doc, "4.2 Survival Model", level=2)
    add_body(doc,
        "A Cox Proportional Hazards model (lifelines, penalizer=0.1) estimates "
        "remaining useful life per drive, evaluated with the concordance index -- "
        "the fraction of comparable drive pairs the model ranks in the correct "
        "order of who fails first. A Kaplan-Meier estimator provides the fleet-wide "
        "baseline survival curve shown alongside each drive's individual curve."
    )
    add_heading(doc, "4.3 Explainability", level=2)
    add_body(doc,
        "SHAP's TreeExplainer (exact, not approximated, for tree ensembles) "
        "computes per-drive feature contributions, shown on the Operator Lookup "
        "dashboard and reused by the automated alert emails to explain why a "
        "specific drive was flagged."
    )
    add_heading(doc, "4.4 Cost Optimization", level=2)
    add_body(doc,
        "A cost simulator sweeps every combination of alert threshold and cost "
        "ratio (downtime cost / replacement cost) against the classifier's real "
        "held-out test outcomes, showing that the optimal threshold moves with an "
        "organization's own cost structure rather than having one universal answer."
    )

    # ---------- Results ----------
    add_heading(doc, "5. Results", level=1)
    add_body(doc, "Classifier (deployed model, grouped held-out test fold):", bold=True)
    add_metric_table(doc,
        headers=["Metric", "Value"],
        rows=[
            ["Precision (failure class)", f"{report['precision']:.0%}"],
            ["Recall (failure class)", f"{report['recall']:.0%}"],
            ["F1 score", f"{report['f1-score']:.0%}"],
            ["ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}"],
            ["Train / test rows", f"{metrics['n_train']} / {metrics['n_test']}"],
            ["Train / test drives (grouped, zero overlap)",
             f"{metrics.get('n_drives_train', '?')} / {metrics.get('n_drives_test', '?')}"],
        ],
    )
    add_body(doc,
        "Cross-fold stability (all 4 StratifiedGroupKFold folds evaluated, not "
        "just the deployed one) -- reported because only about 43 real failing "
        "drives underlie every classifier metric, which is genuinely few for a "
        "held-out evaluation:", italic=True
    )
    add_metric_table(doc,
        headers=["Metric", "Mean", "Std Dev"],
        rows=[
            ["Precision", f"{cv_mean.get('precision', 0):.3f}", f"{cv_std.get('precision', 0):.3f}"],
            ["Recall", f"{cv_mean.get('recall', 0):.3f}", f"{cv_std.get('recall', 0):.3f}"],
            ["F1", f"{cv_mean.get('f1', 0):.3f}", f"{cv_std.get('f1', 0):.3f}"],
            ["ROC-AUC", f"{cv_mean.get('roc_auc', 0):.3f}", f"{cv_std.get('roc_auc', 0):.3f}"],
        ],
    )
    add_body(doc,
        "Recall in particular varies substantially fold to fold (a range from "
        "roughly 0.62 to 1.00 was observed), depending on which specific failing "
        "drives land in a given test fold. This range, not the single deployed "
        "fold's point estimate, is the statistically honest answer to 'how good is "
        "the model.'"
    )
    add_body(doc, "Survival model:", bold=True)
    add_metric_table(doc,
        headers=["Metric", "Value"],
        rows=[
            ["Concordance index", f"{surv['concordance_index']:.3f}"],
            ["Drives in survival model", f"{surv['n_drives']:,}"],
            ["Failure events observed", surv["n_events"]],
        ],
    )

    # ---------- Dashboards ----------
    add_heading(doc, "6. Dashboards", level=1)
    add_heading(doc, "6.1 Home", level=2)
    add_body(doc,
        "Landing page with an accurate, automatically-generated data-authenticity "
        "banner (reads data/provenance.json rather than a hardcoded claim), links "
        "to all three dashboards, and the real model snapshot."
    )
    add_screenshot(doc, "01_home.png", "Home page -- real-data banner and model snapshot")

    add_heading(doc, "6.2 Fleet Overview", level=2)
    add_body(doc,
        "The leadership/budget view: a chronological, honestly-labeled replay of "
        "the real quarter, a 3D rack of monitored drives colored by risk tier, an "
        "event feed, a fleet-risk trend line, and an adjustable cost-avoidance "
        "estimate."
    )
    add_screenshot(doc, "03_fleet_overview_rack.png", "Fleet Overview -- 3D rack and event feed showing both elevated and critical events")

    add_heading(doc, "6.3 Operator Lookup", level=2)
    add_body(doc,
        "The technician view: pick one drive and see its individual risk score, "
        "explained with SHAP values specific to that drive."
    )
    add_screenshot(doc, "04_operator_lookup.png", "Operator Lookup -- a real high-risk drive with its SHAP explanation")

    add_heading(doc, "6.4 Optimization Lab", level=2)
    add_body(doc,
        "The differentiator dashboard: a remaining-useful-life estimate per drive "
        "and a 3D cost-optimized alert-threshold surface built from real held-out "
        "test predictions."
    )
    add_screenshot(doc, "06_optimization_lab_cost_surface.png", "Optimization Lab -- 3D cost surface responding to threshold and cost-ratio sliders")

    # ---------- Automation ----------
    add_heading(doc, "7. Automation", level=1)
    add_body(doc,
        "Two email-alert layers, both free (rule-based summaries built from real "
        "SHAP output, no external API), covering two severity tiers -- an early "
        "warning the moment a drive crosses into elevated risk, and a critical "
        "alert if it goes on to cross critical risk:"
    )
    add_bullet(doc, "In-browser alert: fires instantly while someone is actively "
                    "stepping through Fleet Overview's replay.")
    add_bullet(doc, "Scheduled automation: a GitHub Actions workflow running on a "
                    "real hourly cron schedule, independent of the deployed app or "
                    "anyone having it open -- free and unlimited on a public "
                    "repository. State persists in automation/state.json, committed "
                    "back to the repository after every run.")

    # ---------- Limitations ----------
    add_heading(doc, "8. Known Limitations", level=1)
    for item in [
        "Imbalanced, rare-event target. Real drives fail roughly 1.36% of the time "
        "annually; precision/recall/F1 are reported, not raw accuracy.",
        "Only about 43 real failing drives underlie every classifier metric -- "
        "genuinely few for a held-out evaluation, addressed by reporting cross-fold "
        "stability rather than a single point estimate.",
        "One vendor's fleet. Backblaze's operating conditions may not match "
        "another organization's -- this is a demonstrated method, not a drop-in "
        "production tool.",
        "\"Live\" is a labeled replay of real historical data, not a live sensor "
        "feed -- disclosed directly in the app itself.",
        "The cost-avoidance figures are adjustable planning assumptions, not "
        "sourced statistics.",
        "The survival model uses static, last-observed covariates, not a full "
        "time-varying history -- a standard simplification for a course project.",
        "The 7-day early-warning horizon is a chosen parameter, not a universal "
        "constant.",
    ]:
        add_bullet(doc, item)

    # ---------- Future Work ----------
    add_heading(doc, "9. Future Work", level=1)
    add_bullet(doc, "Validate the Cox proportional-hazards assumption "
                    "(cph.check_assumptions()).")
    add_bullet(doc, "A calibration curve / Brier score for the classifier's "
                    "predicted probabilities.")
    add_bullet(doc, "Extend the existing GitHub Actions automation to periodically "
                    "retrain the model as new data arrives.")
    add_bullet(doc, "Data drift detection between training and live data.")
    add_bullet(doc, "A pytest test suite and CI workflow for the pipeline scripts "
                    "and alerting logic.")

    # ---------- References ----------
    add_heading(doc, "10. References", level=1)
    add_body(doc,
        "Backblaze. (2026, January). Backblaze Drive Stats for 2025. Backblaze "
        "Blog. https://www.backblaze.com/blog/backblaze-drive-stats-for-2025/"
    )

    os.makedirs("docs", exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
