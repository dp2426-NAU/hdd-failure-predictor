# Automated fleet check

`state.json` in this folder is the persisted state for `scripts/check_and_alert.py`,
run on a schedule by `.github/workflows/fleet-check.yml` (GitHub Actions).

This is the genuinely unattended automation in this project: it runs
independently of the deployed Streamlit app, on its own schedule, whether
or not anyone has the dashboard open. It's a different thing from
`live_feed.py`'s in-browser critical-alert email, which only fires while
someone is actively stepping through Fleet Overview's replay — see the
"AI-drafted critical alerts" note on that page, and `email_alerts.py`'s
docstring, for that distinction.

## How it works

Each scheduled run:
1. Reads `state.json` to see which simulated day it left off on.
2. Advances a couple of simulated days forward through the real Q1 2026
   Backblaze quarter.
3. Checks for any drive that newly crossed into critical risk, or
   genuinely failed, in that window.
4. Builds a plain-English status report (`report_generator.py` — rule-based
   string templates over real SHAP output, no external API, no cost) and
   emails it via SMTP (`email_alerts.py`).
5. Writes the updated `state.json` back, which the workflow commits to the
   repo so the next scheduled run continues from here.

When it reaches the end of the 90-day quarter, it loops back to the start
(same behavior as the in-app replay), so it keeps running indefinitely
without anyone needing to reset it.

## Setting it up

Add these as **GitHub Actions secrets** (repo → Settings → Secrets and
variables → Actions → New repository secret) — not as Render environment
variables, which are a separate system for the deployed app:

- `SMTP_HOST` — e.g. `smtp.gmail.com`
- `SMTP_PORT` — e.g. `587`
- `SMTP_USER` — the sending account's email address
- `SMTP_PASSWORD` — an **app password**, not your regular account password
  (Gmail and most providers reject plain passwords for SMTP)
- `ALERT_EMAIL_TO` — where reports are sent

Leaving any of these unset doesn't break the workflow — `check_and_alert.py`
still runs and updates `state.json` on schedule, it just won't send an
email (the run's log says so explicitly).

## Verifying it's actually running

The most convincing evidence isn't in this repo's files, it's in **GitHub's
own Actions tab** — the run history, with real timestamps, shows the
workflow executing on its own schedule with nobody at a keyboard. Trigger
one manually first with the "Run workflow" button (via `workflow_dispatch`)
to confirm the secrets are set up correctly before waiting for the
schedule.

## Known limitations (same honesty standard as the rest of this project)

- Scheduled GitHub Actions runs can be delayed a few minutes during high
  platform load — the cron time is a target, not a guarantee.
- GitHub automatically disables a scheduled workflow after 60 days with no
  repo activity. Any push (including this automation's own state commits)
  resets that clock, so an actively-used repo doesn't hit it in practice.
- This replays a fixed historical quarter, not live telemetry — same
  disclosure as everywhere else "live" or "automated" appears in this
  project. See the README's "What live means" section.
