"""Sentinel dashboard: predictive maintenance demo, told as a story."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

API = os.environ.get("SENTINEL_API_URL", "http://localhost:8000")

HERE = Path(__file__).resolve().parent
try:
    VALIDATION = json.loads((HERE / "validation_results.json").read_text())
except Exception:
    VALIDATION = None

STATE_COLORS = {"ok": "green", "watch": "orange", "advisory": "orangered", "alert": "red"}

# The demo fleet: each machine has a story the visitor should know.
FLEET_STORIES = {
    "P-101": ("The healthy one", "Runs normally for all 30 days. This is the baseline."),
    "P-102": ("Bearing wear", "Starts degrading on day 14. Watch the health score fall."),
    "P-103": ("Overheating", "Runs hot from day 20. The temperature tells the story."),
}

SENSOR_LABELS = {
    "vibration": "Vibration (mm/s)",
    "bearing_temp": "Bearing temperature (C)",
    "discharge_pressure": "Discharge pressure (bar)",
    "rpm": "Speed (rpm)",
    "motor_current": "Motor current (A)",
}

REQUIRED_CSV_COLUMNS = ["timestamp", "machine_id", *SENSOR_LABELS]


def api(path: str, method: str = "get", **kw):
    r = requests.request(method, f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Sentinel", layout="wide")
st.title("Sentinel")
st.write(
    "A predictive maintenance engine. It watches equipment telemetry, learns what "
    "normal looks like, and raises alerts before failures happen, with a plain "
    "English explanation for every alert."
)

with st.expander("How it works: the pipeline"):
    st.write(
        "1. **Ingest.** Raw sensor readings arrive (one row per minute per machine).\n"
        "2. **Features.** Readings are grouped into 60 reading windows. Each window "
        "becomes statistics: mean, spread, min, max, trend per sensor.\n"
        "3. **Detect.** A z-score detector and an Isolation Forest score each window "
        "for abnormality against the machine's own baseline.\n"
        "4. **Health.** Scores feed an EWMA health tracker, 0 to 100, calibrated per "
        "machine: ok, watch, advisory, alert.\n"
        "5. **Explain.** When an alert fires, diagnostics find which sensors deviated "
        "and classify the likely fault, then write a grounded explanation."
    )

if VALIDATION:
    with st.expander("Validation on real data"):
        st.write(VALIDATION["method"])
        cmapss = VALIDATION["cmapss_fd001"]
        cwru = VALIDATION["cwru"]
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**NASA turbofan engines (C-MAPSS FD001)**")
            st.metric("Detection rate",
                      f"{cmapss['detection_rate']:.0%}",
                      f"{cmapss['detected_before_eol']} of {cmapss['engines']} engines")
            st.metric("Median alert lead time",
                      f"{cmapss['median_lead_time_alert_cycles']:.0f} cycles")
            st.metric("Alerts that self-cleared",
                      cmapss["alerts_that_self_cleared"])
            for c in cmapss["caveats"]:
                st.caption("Note: " + c)
        with vc2:
            st.markdown("**Bearing vibration (CWRU)**")
            st.metric("AUROC", f"{cwru['auroc']:.2f}")
            for fault, rate in cwru["detection_rate_per_file"].items():
                st.write(f"{fault.replace('_', ' ').title()}: {rate}")
            st.caption("Held out normal file: " + cwru["heldout_normal"])
            for c in cwru["caveats"]:
                st.caption("Note: " + c)
        st.caption("Reproduce it: `python scripts/prepare_data.py` downloads the "
                   "public datasets, then `python scripts/validate.py` scores them.")

try:
    machines = api("/machines")
except Exception as e:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn api.main:app ({e})")
    st.stop()

if not machines:
    st.subheader("Try the demo")
    st.write(
        "Load 30 days of pump telemetry for three machines: one stays healthy, "
        "one develops bearing wear, one starts overheating. Then watch Sentinel "
        "catch the failures before they happen."
    )
    if st.button("Load the demo fleet"):
        with st.spinner("Generating 30 days of telemetry for 3 pumps..."):
            api("/demo/seed", method="post")
        st.rerun()
    st.stop()

st.subheader("The fleet")
st.write(
    "Three pumps, 30 days of readings each. Pick one to see its story, or compare "
    "them below."
)
cols = st.columns(3)
by_id = {m["machine_id"]: m for m in machines}
for col, mid in zip(cols, sorted(by_id)):
    m = by_id[mid]
    name, story = FLEET_STORIES.get(mid, (mid, "Your uploaded machine."))
    color = STATE_COLORS.get(m["state"], "gray")
    if not m["ready"]:
        col.metric(f"{mid}: {name}", "learning...",
                   f"{m['readings']:,} readings so far")
        col.caption("Still learning its baseline. Upload more data.")
        continue
    col.metric(f"{mid}: {name}", f"{m['health']:.0f} / 100",
               f"{m['state'].upper()} ({m['alerts']} alerts)")
    col.caption(f":{color}[{m['state'].upper()}] " + story)

st.subheader("Test it with your own data")
st.write(
    "Upload a CSV and Sentinel will learn that machine's baseline and watch it "
    "for anomalies, just like the demo fleet."
)
with st.expander("CSV format"):
    st.write(
        "Columns: `timestamp, machine_id, vibration, bearing_temp, "
        "discharge_pressure, rpm, motor_current`. One row per reading, "
        "timestamps parseable by pandas. Sentinel needs about 10 days of "
        "minute-resolution data to learn a baseline before it can score health."
    )
    sample = pd.DataFrame([{
        "timestamp": "2026-01-01 00:00:00", "machine_id": "MY-PUMP",
        "vibration": 2.1, "bearing_temp": 62.0, "discharge_pressure": 4.2,
        "rpm": 2950, "motor_current": 42.0,
    }])
    st.download_button("Download a sample CSV", sample.to_csv(index=False),
                       "sample.csv", "text/csv")

uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df["timestamp"] = df["timestamp"].astype(str)
            payload = {"readings": df.to_dict("records")}
            with st.spinner(f"Ingesting {len(df):,} readings..."):
                result = api("/ingest", method="post", json=payload)
            for s in result["machines"]:
                if s["ready"]:
                    st.success(
                        f"{s['machine_id']}: health {s['health']:.0f}, "
                        f"state {s['state'].upper()}, {s['alerts']} alerts."
                    )
                else:
                    st.warning(
                        f"{s['machine_id']}: received {s['readings']:,} readings, "
                        "still learning its baseline. Upload more history to "
                        "start scoring."
                    )
            st.rerun()
    except Exception as e:
        st.error(f"Could not ingest that file: {e}")

options = [f"{mid}: {FLEET_STORIES.get(mid, (mid, 'Custom upload'))[0]}"
           for mid in sorted(by_id)]
choice = st.selectbox("Inspect a machine", options)
mid = choice.split(":")[0]
status = api(f"/machines/{mid}")
name, story = FLEET_STORIES.get(mid, (mid, "Uploaded by you."))

st.subheader(f"{mid}: {name}")
st.write(story)
if not status["ready"]:
    st.info(
        f"This machine has {status['readings']:,} readings but needs about "
        "14,400 (10 days at one reading per minute) before Sentinel can score it."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Current health", f"{status['health']:.0f} / 100",
          help="100 is perfect. Below 90 is watch, below 80 advisory, below 60 alert.")
c2.metric("State", status["state"].upper())
c3.metric("Alerts raised", status["alerts"])

history = api(f"/machines/{mid}/history")
if history:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h["timestamp"] for h in history],
        y=[h["health"] for h in history],
        mode="lines", name="health",
    ))
    for level, label in ((90, "watch"), (80, "advisory"), (60, "alert")):
        fig.add_hline(y=level, line_dash="dash", line_color="gray",
                      annotation_text=label)
    fig.update_layout(title="Health score over time", yaxis_range=[0, 105],
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

telemetry = api(f"/machines/{mid}/telemetry", params={"n": 2880})
if telemetry:
    sensor_keys = [k for k in telemetry[0] if k != "timestamp"]
    labels = [SENSOR_LABELS.get(k, k) for k in sensor_keys]
    sensor = st.selectbox("Sensor readings (last 2 days)", labels)
    key = sensor_keys[labels.index(sensor)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[t["timestamp"] for t in telemetry], y=[t[key] for t in telemetry],
        mode="lines", name=sensor,
    ))
    fig2.update_layout(title=sensor, height=280,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

st.subheader("Alerts")
st.write("Each alert comes with a plain English explanation of what Sentinel saw.")
alerts = api(f"/machines/{mid}/alerts")
if not alerts:
    st.write("No alerts. This machine stayed healthy.")
for a in reversed(alerts):
    with st.expander(f"{a['timestamp'][:16]}  {a['state'].upper()}  health {a['health']:.0f}"):
        st.write(a["explanation"])
        if a.get("findings"):
            st.write("Evidence:")
            for f in a["findings"]:
                st.write(f"- {f}")
