"""
assistant.py -- Claude-powered features layered on top of the existing
classifier + SHAP explainability, not a replacement for either:

  1. draft_incident_summary() -- turns one drive's SHAP explanation into a
     short plain-English incident summary for the critical-alert email
     (see email_alerts.py, wired in via live_feed.py's step_and_record()).
  2. answer_drive_question() -- answers a free-text question about one
     drive, grounded in its real risk score / readings / SHAP contributions
     (used by the chat panel on Operator Lookup).

Both are gated on ANTHROPIC_API_KEY -- never hardcoded, since the repo is
public. Set it in Render's Environment tab (or locally as an env var) to
turn these on; leave it unset and both functions return None, same
no-op-when-unconfigured pattern as auth.py and email_alerts.py.

Model: defaults to Claude Opus 5 (claude-opus-5), Anthropic's current
flagship, run at low effort -- these are short, well-scoped generation
tasks (a few sentences of grounded prose), not open-ended reasoning, so low
effort keeps latency and cost down without giving up quality. Override with
the ASSISTANT_MODEL env var (e.g. "claude-sonnet-5" or "claude-haiku-4-5")
if you want a cheaper/faster model for a live classroom demo -- every call
spends real money against your own Anthropic API key, there is no shared
or subsidized key here.
"""

from __future__ import annotations

import os

import streamlit as st

MODEL = os.environ.get("ASSISTANT_MODEL", "claude-opus-5")


@st.cache_resource
def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def is_configured() -> bool:
    return _client() is not None


def draft_incident_summary(
    serial: str,
    risk_pct: float,
    readings: dict,
    shap_contributions: list[tuple[str, float]],
) -> str | None:
    """None if not configured; a short incident-summary paragraph otherwise
    (or a visible '(AI summary unavailable: ...)' string on an API error --
    surfaced rather than silently dropped, so a failed call is diagnosable)."""
    client = _client()
    if client is None:
        return None

    shap_lines = "\n".join(f"- {label}: {value:+.3f}" for label, value in shap_contributions)
    reading_lines = "\n".join(f"- {k}: {v}" for k, v in readings.items())

    prompt = (
        f"A hard-drive failure-prediction model just flagged drive {serial} as "
        f"HIGH RISK ({risk_pct:.0%} predicted probability of failing within 7 days).\n\n"
        f"Current SMART readings:\n{reading_lines}\n\n"
        "SHAP contributions (how much each reading pushed the risk score up (+) "
        f"or down (-) from the model's baseline):\n{shap_lines}\n\n"
        "Write a short (3-5 sentence) plain-English incident summary suitable "
        "for an IT technician's inbox: what's wrong, which readings are driving "
        "it, and one concrete recommended next step. No preamble, no markdown "
        "headers -- just the summary text."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception as e:
        return f"(AI summary unavailable: {e})"


def answer_drive_question(question: str, context: dict, history: list[dict]) -> str | None:
    """None if not configured; the assistant's reply otherwise (or a visible
    error string, same reasoning as draft_incident_summary). `history` is
    the running chat as a list of {"role": "user"|"assistant", "content": str}."""
    client = _client()
    if client is None:
        return None

    system = (
        "You are a predictive-maintenance assistant embedded in a hard-drive "
        "failure-prediction dashboard. Answer questions about the ONE drive "
        "described below, grounded in its real model output -- don't invent "
        "numbers that aren't given to you. Be concise (a few sentences unless "
        "asked for more detail). If asked something outside drive health, "
        "SMART data, or this dashboard's scope, say so briefly rather than "
        "guessing.\n\n"
        f"Drive: {context.get('serial')}\n"
        f"Predicted risk of failure within 7 days: {context.get('risk_pct', 0):.0%}\n"
        f"Current SMART readings: {context.get('readings')}\n"
        f"SHAP contributions (feature -> push on risk score, + raises risk): "
        f"{context.get('shap')}"
    )

    messages = history + [{"role": "user", "content": question}]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=system,
            output_config={"effort": "low"},
            messages=messages,
        )
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception as e:
        return f"(AI assistant error: {e})"
