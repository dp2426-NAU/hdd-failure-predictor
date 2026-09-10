"""
Fleet Overview -- the leadership / budget dashboard.

Shows fleet-wide risk trend, a live event feed, a 3D rack visualization, and
a cost-avoidance estimate tied to the ITIC downtime-cost research cited in
the project proposal. See app.py / README.md for the honest explanation of
what "live" means in this project (a labeled replay of real-schema drive-day
data, not a live sensor).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import live_feed
from styles import inject_base_css, stat_card, pill, COLOR_ACCENT, COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL
from components.three_d_rack import render_rack

st.set_page_config(page_title="Fleet Overview · Drive Failure", page_icon="🖥️", layout="wide")
inject_base_css()
live_feed.init_session_state()

st.title("🖥️ Fleet Overview")

top_l, top_r = st.columns([3, 1])
with top_l:
    st.caption("Chronological replay of drive-day telemetry, one simulated day per tick — a labeled "
               "stand-in for a live monitoring feed. See the note on the Home page for what this does and doesn't mean.")
with top_r:
    live_on = st.toggle("▶ Live mode", value=False, help="Auto-advance the simulated clock")

control_l, control_r, control_reset = st.columns([1, 1, 1])
with control_l:
    st.button("⏭ Step forward one day", on_click=live_feed.step_and_record, use_container_width=True)
with control_r:
    speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast"], value="Normal", label_visibility="collapsed")
with control_reset:
    st.button("↺ Restart simulation", on_click=live_feed.reset, use_container_width=True)

if live_on:
    interval_ms = {"Slow": 3000, "Normal": 1500, "Fast": 600}[speed]
    st_autorefresh(interval=interval_ms, key="fleet_autorefresh")
    live_feed.step_and_record()
    st.markdown(f'<span class="live-dot"></span> **LIVE** — day {live_feed.current_date().date()}', unsafe_allow_html=True)
else:
    st.markdown(f"**Paused** — day {live_feed.current_date().date()}")

snapshot = live_feed.get_fleet_snapshot()
history = live_feed.get_history_df()

st.write("")
k1, k2, k3, k4 = st.columns(4)
with k1:
    stat_card("Drives monitored", f"{len(snapshot):,}")
with k2:
    n_crit = int((snapshot["tier"] == "critical").sum())
    stat_card("High-risk right now", f"{n_crit}", sub=pill("critical", "needs attention") if n_crit else pill("good", "all clear"))
with k3:
    stat_card("Failures observed", f"{st.session_state.cumulative_failures}")
with k4:
    warned = st.session_state.failures_with_prior_warning
    total_fail = max(st.session_state.cumulative_failures, 1)
    stat_card("Caught with advance warning", f"{warned}/{st.session_state.cumulative_failures}",
               sub=f"{warned/total_fail:.0%} of failures were flagged high-risk before they happened")

st.write("")
st.markdown("##### Estimated cost avoided")
cc1, cc2 = st.columns([1, 2])
with cc1:
    cost_per_incident = st.slider(
        "Assumed cost per avoided emergency incident ($)",
        min_value=100, max_value=5000, value=500, step=50,
        help="Illustrative planning assumption — replace with your organization's real figures. "
             "Not sourced from ITIC's enterprise downtime-cost figures, which measure full-outage hours, not per-drive incidents.",
    )
with cc2:
    avoided = cost_per_incident * warned
    stat_card("Illustrative avoided cost", f"${avoided:,.0f}",
               sub="= (drives caught before failure) × (assumed cost per incident, adjustable at left)")

st.divider()

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Fleet risk over time")
    if len(history) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["avg_risk"] * 100,
            mode="lines", name="Avg. fleet risk", line=dict(color=COLOR_ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(87,214,196,0.12)",
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e9efed",
            yaxis=dict(title="Avg. risk %", gridcolor="#2b3538", ticksuffix="%"),
            xaxis=dict(gridcolor="#2b3538"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Step forward a few days (or turn on Live mode) to build up the trend line.")

    st.subheader("Event feed")
    if st.session_state.live_events:
        rows = "".join(
            f'<div class="event-row"><span class="event-date">{e["date"]}</span>'
            f'{pill(e["level"] if e["level"] in ("critical",) else "warning", e["level"])} {e["message"]}</div>'
            for e in st.session_state.live_events[:15]
        )
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.caption("No events yet — step forward to start the replay.")

with right:
    st.subheader("Fleet, in 3D")
    st.caption("Each block is one monitored drive, colored by current risk tier. Drag to rotate, hover for detail.")
    drives_payload = [
        {"serial": row.serial_number, "tier": row.tier, "risk": float(row.risk)}
        for row in snapshot.itertuples()
    ]
    render_rack(drives_payload, height=420)
    st.markdown(
        f'{pill("good", "healthy")} {pill("warning", "elevated")} {pill("critical", "critical")}',
        unsafe_allow_html=True,
    )
