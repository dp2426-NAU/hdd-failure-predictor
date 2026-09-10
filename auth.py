"""
auth.py -- an optional sign-in gate for the deployed demo.

This is NOT real security -- it doesn't protect the GitHub repo, the model
files, or the data; it only keeps the public onrender.com URL from being
casually opened by search crawlers or randoms before a presentation.
Streamlit's session_state resets per browser session/tab, so this is a soft
"keep the link private until I'm ready to show it" gate, not authentication.
There is exactly one shared login (not per-user accounts).

Credentials are read from environment variables (APP_LOGIN_ID,
APP_PASSWORD), never hardcoded here -- the repo is public, so a hardcoded
credential would be visible to anyone reading the source. Set both in
Render's dashboard (Environment tab) to turn the gate on; leave either
unset (as both are by default, including for local development and the
AppTest checks) and the app opens with no gate at all.

Call require_login() as the first thing after st.set_page_config() and
inject_base_css() on every page (app.py and every file in pages/) --
Streamlit's multipage routing lets someone open a page's URL directly
without going through app.py first, so each page needs its own check.
Session state is shared across pages within one browser session, so a
viewer only has to sign in once per visit (per tab), not once per page.
"""

import os

import streamlit as st

from styles import COLOR_BORDER, COLOR_CRITICAL, COLOR_INK, COLOR_MUTED, COLOR_SURFACE


def require_login():
    login_id = os.environ.get("APP_LOGIN_ID")
    password = os.environ.get("APP_PASSWORD")
    if not login_id or not password:
        return  # no credentials configured -- app stays open

    if st.session_state.get("authenticated"):
        return

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
        }}
        .signin-card {{
            width: 100%;
            max-width: 380px;
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 32px 30px 26px;
            text-align: center;
        }}
        .signin-icon {{ font-size: 32px; margin-bottom: 4px; }}
        .signin-title {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {COLOR_INK};
            margin: 4px 0 2px;
        }}
        .signin-sub {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 13px;
            color: {COLOR_MUTED};
            margin: 0 0 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(
            """
            <div class="signin-card">
                <div class="signin-icon">🔒</div>
                <p class="signin-title">Predicting Drive Failure</p>
                <p class="signin-sub">Sign in to view this demo</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("signin_form"):
            entered_id = st.text_input("Login ID")
            entered_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if entered_id == login_id and entered_password == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.markdown(
                    f'<p style="color:{COLOR_CRITICAL};font-family:\'IBM Plex Sans\',sans-serif;'
                    f'font-size:13.5px;text-align:center;">Incorrect ID or password.</p>',
                    unsafe_allow_html=True,
                )

    st.stop()
