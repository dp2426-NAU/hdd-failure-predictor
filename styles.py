"""
styles.py -- shared visual theme for every page.

Palette mirrors the project's planning doc: IBM Plex type family, a teal
"signal" accent, and a reserved green/amber/red status system for
healthy/elevated/critical drive risk (status color is never reused as a
generic series color -- see dataviz notes in README).
"""

import streamlit as st

COLOR_BG = "#12181a"
COLOR_SURFACE = "#1a2225"
COLOR_SURFACE_2 = "#212b2e"
COLOR_BORDER = "#2b3538"
COLOR_INK = "#e9efed"
COLOR_MUTED = "#93a19d"
COLOR_ACCENT = "#57d6c4"

COLOR_GOOD = "#4caf7d"
COLOR_WARNING = "#e0ab54"
COLOR_CRITICAL = "#e0705a"


def inject_base_css():
    # NOTE: no blank lines are allowed inside this <style> block. Streamlit's
    # markdown renderer follows CommonMark's raw-HTML-block rule, which ends
    # an HTML block at the first blank line -- any blank line here would
    # split the block and cause the remaining CSS to be emitted as literal
    # visible paragraph text instead of being parsed as part of <style>.
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        }}
        code, .mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; }}
        [data-testid="stAppViewContainer"] {{ background: {COLOR_BG}; }}
        [data-testid="stSidebar"] {{ background: {COLOR_SURFACE}; border-right: 1px solid {COLOR_BORDER}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .stat-card {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 16px 18px;
        }}
        .stat-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: {COLOR_MUTED};
            margin: 0 0 6px;
        }}
        .stat-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 28px;
            font-weight: 600;
            color: {COLOR_INK};
            margin: 0;
        }}
        .stat-sub {{ font-size: 12.5px; color: {COLOR_MUTED}; margin-top: 4px; }}
        .pill {{
            display: inline-flex; align-items: center; gap: 6px;
            font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
            letter-spacing: .04em; text-transform: uppercase;
            padding: 3px 10px; border-radius: 100px;
        }}
        .pill.good {{ background: rgba(76,175,125,0.14); color: {COLOR_GOOD}; }}
        .pill.warning {{ background: rgba(224,171,84,0.14); color: {COLOR_WARNING}; }}
        .pill.critical {{ background: rgba(224,112,90,0.14); color: {COLOR_CRITICAL}; }}
        .pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
        .event-row {{
            display: flex; gap: 10px; align-items: baseline;
            padding: 7px 0; border-bottom: 1px solid {COLOR_BORDER}; font-size: 13px;
        }}
        .event-row:last-child {{ border-bottom: none; }}
        .event-date {{ font-family: 'IBM Plex Mono', monospace; color: {COLOR_MUTED}; font-size: 11.5px; white-space: nowrap; }}
        .live-dot {{
            display:inline-block; width:8px; height:8px; border-radius:50%;
            background:{COLOR_CRITICAL}; margin-right:6px;
            animation: pulse 1.4s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }} 50% {{ opacity: .35; }} 100% {{ opacity: 1; }}
        }}
        .disclosure {{
            border: 1px solid {COLOR_BORDER};
            border-left: 3px solid {COLOR_ACCENT};
            background: {COLOR_SURFACE};
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 13px;
            color: {COLOR_MUTED};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="stat-card">
            <p class="stat-label">{label}</p>
            <p class="stat-value">{value}</p>
            {f'<p class="stat-sub">{sub}</p>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(level: str, text: str) -> str:
    return f'<span class="pill {level}"><span class="dot"></span>{text}</span>'
