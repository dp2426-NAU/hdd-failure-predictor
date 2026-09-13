"""
email_alerts.py -- sends the critical-risk incident summary via SMTP (see
report_generator.py for the rule-based drafting, live_feed.py and
scripts/check_and_alert.py for where this is triggered).

HONEST SCOPE: this fires during the interactive replay -- when someone has
the app open and is stepping through days or running Live mode on Fleet
Overview. Render's free tier has no persistent background worker, and
there's no live sensor behind this project (see the "what live means"
disclosure on the Home page), so this is NOT a 24/7 monitor that emails you
while your laptop is closed. It's the same honest framing as everything
else "live" in this project, applied to the new alert feature.

All credentials come from environment variables, never hardcoded (the repo
is public):
    SMTP_HOST      -- e.g. smtp.gmail.com
    SMTP_PORT      -- e.g. 587
    SMTP_USER      -- the sending account's email address
    SMTP_PASSWORD  -- an APP password, not your regular account password --
                      Gmail (and most providers) reject plain account
                      passwords for SMTP; generate a dedicated app password
    ALERT_EMAIL_TO -- where alerts are sent

If any of these five are unset, send_alert_email() is a no-op that returns
False -- so local dev and the AppTest checks are unaffected.
"""

import os
import smtplib
from email.mime.text import MIMEText

_REQUIRED_VARS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO")


def is_configured() -> bool:
    return all(os.environ.get(k) for k in _REQUIRED_VARS)


def send_alert_email(subject: str, body: str) -> bool:
    """Returns True only on a confirmed send. Never raises -- a bad SMTP
    config shouldn't crash the dashboard mid-replay or the scheduled
    GitHub Actions check, it should just mean no email went out (callers
    surface that to the event feed / run log).

    On failure, prints the exception to stdout -- smtplib error messages
    (e.g. "535 Username and Password not accepted") never include the
    password itself, only the server's rejection reason, so this is safe
    to log and is what actually makes a bad SMTP config diagnosable
    instead of a silent, unexplained False."""
    if not is_configured():
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    to_addr = os.environ["ALERT_EMAIL_TO"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        return True
    except Exception as e:
        print(f"send_alert_email failed: {type(e).__name__}: {e}")
        return False
