"""
report_generator.py -- rule-based (no external API, free) incident and
status-report text. Shared by:
  - live_feed.py's in-browser alert (fires when someone is actively
    stepping through Fleet Overview's replay in the deployed app)
  - scripts/check_and_alert.py's scheduled GitHub Actions automation
    (fires on a real cron schedule, independent of anyone using the app --
    see that script and .github/workflows/fleet-check.yml)

Deliberately template-based, not an LLM: turns a drive's real SHAP
contributions into a short plain-English summary using plain Python string
formatting over real model output. No API key, no per-call cost, no
network dependency beyond SMTP for actually sending it.
"""

from __future__ import annotations


def incident_summary(
    serial: str,
    risk_pct: float,
    shap_contributions: list[tuple[str, float]],
) -> str:
    """A short plain-English incident summary for one newly-critical drive,
    built from its real SHAP contributions (positive = raises risk)."""
    positive = sorted(
        [(label, value) for label, value in shap_contributions if value > 0],
        key=lambda x: -x[1],
    )
    top = positive[:2]
    if top:
        drivers = " and ".join(f"{label} ({value:+.2f})" for label, value in top)
        driver_sentence = f"Main contributing factors: {drivers}."
    else:
        driver_sentence = "No single reading dominates -- the risk is a combination of several small elevations."

    return (
        f"{risk_pct:.0%} predicted probability of failing within the next 7 days. "
        f"{driver_sentence} Recommended action: schedule a proactive inspection or "
        "replacement within the next few days."
    )


def status_report(
    period_label: str,
    drives_monitored: int,
    newly_critical: list[dict],
    failures: list[dict],
    cumulative_failures: int,
    cumulative_warned: int,
) -> tuple[str, str]:
    """Returns (subject, body) for one automated check window.
    newly_critical: list of {"date", "serial", "risk_pct", "summary"}
    failures: list of {"date", "serial", "had_warning"}"""
    n_crit = len(newly_critical)
    n_fail = len(failures)

    if n_fail > 0:
        subject = f"🚨 Fleet check ({period_label}): {n_fail} failure(s), {n_crit} new critical drive(s)"
    elif n_crit > 0:
        subject = f"⚠️ Fleet check ({period_label}): {n_crit} new critical drive(s)"
    else:
        subject = f"✅ Fleet check ({period_label}): no new critical drives"

    lines = [
        f"Automated fleet check -- {period_label}",
        f"Drives monitored: {drives_monitored:,}",
        f"Cumulative failures this run: {cumulative_failures} ({cumulative_warned} caught with advance warning)",
        "",
    ]

    if failures:
        lines.append(f"FAILURES ({n_fail}):")
        for f in failures:
            lines.append(
                f"- {f['date']}: drive {f['serial']} failed"
                + (" (was flagged high-risk in advance)" if f.get("had_warning") else " (no prior warning)")
            )
        lines.append("")

    if newly_critical:
        lines.append(f"NEWLY CRITICAL ({n_crit}):")
        for c in newly_critical:
            lines.append(f"- {c['date']}: drive {c['serial']} -- {c['risk_pct']:.0%}")
            lines.append(f"  {c['summary']}")
        lines.append("")

    if n_crit == 0 and n_fail == 0:
        lines.append("No new critical drives or failures since the last check. Fleet nominal.")
        lines.append("")

    lines.append(
        "-- This is an automated, scheduled check of a real historical Backblaze quarter, "
        "replayed chronologically. There is no live sensor behind this project -- see the "
        "README for the full honest explanation of what this automation does and doesn't mean."
    )

    return subject, "\n".join(lines)
