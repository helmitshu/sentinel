"""Sentinel dashboard: fleet health, telemetry, and alert explanations."""
from __future__ import annotations

import os

import requests
import streamlit as st
import plotly.graph_objects as go

API = st.sidebar.text_input(
    "API URL", os.environ.get("SENTINEL_API_URL", "http://localhost:8000")
)

STATE_COLORS = {"ok": "green", "watch": "orange", "advisory": "orangered", "alert": "red"}


def api(path: str, method: str = "get", **kw):
    r = requests.request(method, f"{API}{path}", timeout=60, **kw)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Sentinel", layout="wide")
st.title("Sentinel: predictive maintenance")

if st.sidebar.button("Seed demo fleet"):
    with st.spinner("Generating 30 days for 3 pumps..."):
        api("/demo/seed", method="post")
    st.sidebar.success("Fleet seeded")

try:
    machines = api("/machines")
except Exception as e:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn api.main:app ({e})")
    st.stop()

if not machines:
    st.info("No machines yet. Seed the demo fleet from the sidebar.")
    st.stop()

mid = st.sidebar.selectbox("Machine", [m["machine_id"] for m in machines])
status = api(f"/machines/{mid}")

c1, c2, c3 = st.columns(3)
c1.metric("Health", status["health"])
c2.metric("State", status["state"].upper())
c3.metric("Alerts", status["alerts"])
st.markdown(f"State: :{STATE_COLORS.get(status['state'], 'gray')}[{status['state'].upper()}]")

history = api(f"/machines/{mid}/history")
if history:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h["timestamp"] for h in history],
        y=[h["health"] for h in history],
        mode="lines", name="health",
    ))
    for level, name in ((90, "watch"), (80, "advisory"), (60, "alert")):
        fig.add_hline(y=level, line_dash="dash", line_color="gray",
                      annotation_text=name)
    fig.update_layout(title="Machine health over time", yaxis_range=[0, 105],
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

telemetry = api(f"/machines/{mid}/telemetry", params={"n": 2880})
if telemetry:
    sensor = st.selectbox("Sensor", [k for k in telemetry[0] if k != "timestamp"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[t["timestamp"] for t in telemetry], y=[t[sensor] for t in telemetry],
        mode="lines", name=sensor,
    ))
    fig2.update_layout(title=f"{sensor} (last {len(telemetry)} readings)",
                       height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

st.subheader("Alerts")
alerts = api(f"/machines/{mid}/alerts")
if not alerts:
    st.write("No alerts.")
for a in reversed(alerts):
    with st.expander(f"{a['timestamp'][:16]}  {a['state'].upper()}  health {a['health']}"):
        st.write(a["explanation"])
        if a.get("findings"):
            st.write("Evidence:")
            for f in a["findings"]:
                st.write(f"- {f}")
