"""
auth.py -- an optional password gate for the deployed demo.

This is NOT real security -- it doesn't protect the GitHub repo, the model
files, or the data; it only keeps the public onrender.com URL from being
casually opened by search crawlers or randoms before a presentation.
Streamlit's session_state resets per browser session, so this is a soft
"keep the link private until I'm ready to show it" gate, not authentication.

The password is read from an environment variable (APP_PASSWORD), never
hardcoded here -- the repo is public, so a hardcoded password would be
visible to anyone reading the source. Set APP_PASSWORD in Render's
dashboard (Environment tab) to turn the gate on; leave it unset (as it is
by default, including for local development and the AppTest checks) and
the app opens with no gate at all.

Call require_password() as the very first thing after st.set_page_config()
on every page (app.py and every file in pages/) -- Streamlit's multipage
routing lets someone open a page's URL directly without going through
app.py first, so each page needs its own check. Session state is shared
across pages within one browser session, so a viewer only has to enter the
password once per visit, not once per page.
"""

import os

import streamlit as st


def require_password():
    app_password = os.environ.get("APP_PASSWORD")
    if not app_password:
        return  # no password configured -- app stays open

    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Predicting Drive Failure")
    st.caption("This demo is password-protected. Enter the password to continue.")
    entered = st.text_input("Password", type="password", key="auth_password_input")
    if st.button("Enter"):
        if entered == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
