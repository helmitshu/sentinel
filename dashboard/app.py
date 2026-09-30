"""Sentinel dashboard: predictive maintenance, told as a story."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from sentinel.industries import INDUSTRY_TEMPLATES

API = os.environ.get("SENTINEL_API_URL", "http://localhost:8000")
HERE = Path(__file__).resolve().parent


def load_json(name: str):
    try:
        return json.loads((HERE / name).read_text())
    except Exception:
        return None


VALIDATION = load_json("validation_results.json")
NASA_DEMO = load_json("nasa_demo.json")

st.set_page_config(page_title="Sentinel", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.badge { display:inline-block; padding:3px 12px; border-radius:999px;
         font-size:11px; font-weight:700; letter-spacing:0.08em; }
.badge-real { background:#23402B; color:#F5F2E9; }
.badge-demo { background:#C15F3C; color:#FFF9F2; }
.hero-card { background:#FFFFFF; border:1px solid #E7E2D4; border-radius:16px;
             padding:26px 26px 20px 26px; height:100%; }
.hero-card h3 { margin-top:10px; }
.verdict { background:#FFFFFF; border:1px solid #E7E2D4; border-left:5px solid #C15F3C;
           border-radius:12px; padding:16px 20px; margin:12px 0; }
.verdict-good { border-left-color:#23402B; }
.big-health { font-size:52px; font-weight:700; line-height:1; }
.state-pill { display:inline-block; padding:4px 14px; border-radius:999px;
              font-weight:700; font-size:14px; }
.stButton>button { border-radius:10px; }
</style>
""",
    unsafe_allow_html=True,
)


def api(path: str, method: str = "get", **kw):
    r = requests.request(method, f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def badge(kind: str) -> str:
    if kind == "real":
        return '<span class="badge badge-real">REAL DATA</span>'
    return '<span class="badge badge-demo">SIMULATED DEMO</span>'


def sensor_label(name: str, unit: str) -> str:
    return f"{name.replace('_', ' ').title()} ({unit})"


# ---------------------------------------------------------------- header ---
st.title("Sentinel")
st.write(
    "Software that watches your machines and warns you before they break. "
    "Pick a demo below: a real jet engine, or a mining shovel."
)

hero_a, hero_b = st.columns(2)
with hero_a:
    st.markdown(
        f"""<div class="hero-card">{badge("real")}
<h3>Watch a jet engine fail</h3>
<p>Real run-to-failure data from NASA. Scrub through an engine's life and
see Sentinel flag the failure before it happens. No labels, no training
on failures: it learns healthy, then watches.</p></div>""",
        unsafe_allow_html=True,
    )
    go_nasa = st.button("Run the jet engine demo", key="go_nasa", use_container_width=True)
with hero_b:
    st.markdown(
        f"""<div class="hero-card">{badge("demo")}
<h3>Watch a mining shovel fail</h3>
<p>A rope shovel on shift rotation, simulated. The hoist bearing wears out
over 30 days: vibration climbs, the motor runs hot, and Sentinel raises
the flag before the shovel goes offline.</p></div>""",
        unsafe_allow_html=True,
    )
    go_shovel = st.button("Run the shovel demo", key="go_shovel", use_container_width=True)

st.divider()

# ------------------------------------------------------------------ demos ---
tab_nasa, tab_shovel = st.tabs(["Jet engine demo", "Shovel demo"])

with tab_nasa:
    st.markdown(badge("real"), unsafe_allow_html=True)
    st.subheader("A real engine, from healthy to failed")
    if not NASA_DEMO:
        st.error(
            "Demo data not found. Generate it with: "
            "`python scripts/build_nasa_demo.py`"
        )
    else:
        st.write(NASA_DEMO["method"])
        labels = NASA_DEMO["labels"]
        engines = {e["engine"]: e for e in NASA_DEMO["engines"]}
        order = [labels["early"], labels["typical"], labels["late"]]
        nice = {
            labels["early"]: f"Engine {labels['early']}: the early catch",
            labels["typical"]: f"Engine {labels['typical']}: the typical catch",
            labels["late"]: f"Engine {labels['late']}: the late catch",
        }
        pick = st.radio("Pick an engine", order, format_func=lambda x: nice[x],
                        horizontal=True, key="nasa_engine")
        eng = engines[pick]
        traj = eng["trajectory"]
        eol = eng["eol"]

        cycle = st.slider("Scrub through the engine's life (cycles)", 5, eol,
                          eol, step=5, key="nasa_cycle")
        shown = [p for p in traj if p["cycle"] <= cycle]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p["cycle"] for p in shown], y=[p["health"] for p in shown],
            mode="lines", name="engine health",
            line=dict(color="#C15F3C", width=3),
        ))
        for lvl, name, color in ((90, "watch", "#C77E2A"), (80, "advisory", "#B4552D"),
                                 (60, "alert", "#A33327")):
            fig.add_hline(y=lvl, line_dash="dash", line_color=color,
                          annotation_text=name)
        if eng["first_warning_cycle"] and cycle >= eng["first_warning_cycle"]:
            fig.add_vline(x=eng["first_warning_cycle"], line_dash="dot",
                          line_color="#C77E2A", annotation_text="first warning")
        if eng["first_alert_cycle"] and cycle >= eng["first_alert_cycle"]:
            fig.add_vline(x=eng["first_alert_cycle"], line_dash="dot",
                          line_color="#A33327", annotation_text="alert raised")
        fig.add_vline(x=eol, line_color="#23211C", annotation_text="failure")
        fig.update_layout(yaxis_range=[0, 105], height=360,
                          margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="operating cycles", yaxis_title="health (0-100)")
        st.plotly_chart(fig, width="stretch")

        if eng["first_alert_cycle"] and cycle >= eng["first_alert_cycle"]:
            st.markdown(
                f"""<div class="verdict verdict-good"><b>Sentinel raised the flag
                {eng["lead_alert"]} operating cycles before this engine failed.</b>
                It had never seen a failure. It just knew what healthy looked
                like, and this stopped looking healthy.</div>""",
                unsafe_allow_html=True,
            )
        elif eng["first_warning_cycle"] and cycle >= eng["first_warning_cycle"]:
            st.markdown(
                f"""<div class="verdict"><b>Something is off.</b> Sentinel is
                watching this engine closely, {eng["lead_warning"]} cycles before
                failure. Keep scrubbing.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="verdict"><b>Healthy.</b> The engine is running
                normally and Sentinel agrees. Scrub forward to watch it
                degrade.</div>""",
                unsafe_allow_html=True,
            )
        if pick == labels["late"]:
            st.caption(
                "Honest note: this one was caught late, only "
                f"{eng['lead_alert']} cycles before failure. Early detection "
                "is the goal, not a guarantee on every engine."
            )

with tab_shovel:
    st.markdown(badge("demo"), unsafe_allow_html=True)
    st.subheader("A shovel's hoist bearing wears out")
    st.write(
        "Three acts. Days 1 to 14: the shovel works normally and Sentinel "
        "learns what healthy looks like. Day 14: the hoist bearing starts "
        "wearing. Watch the health score fall and the alert fire, with a "
        "plain English explanation of what Sentinel saw."
    )
    tpl = INDUSTRY_TEMPLATES["mining_shovel"]
    st.caption(tpl["demo_story"])

    scol1, scol2 = st.columns(2)
    if scol1.button("Simulate: bearing wear", key="shovel_bw", use_container_width=True):
        with st.spinner("Simulating 30 days on the shovel..."):
            api("/demo/scenario", method="post",
                json={"fault": "bearing_wear", "industry": "mining_shovel"})
        st.rerun()
    if scol2.button("Simulate: healthy shovel", key="shovel_ok", use_container_width=True):
        with st.spinner("Simulating 30 days on the shovel..."):
            api("/demo/scenario", method="post",
                json={"fault": "healthy", "industry": "mining_shovel"})
        st.rerun()

    try:
        machines = {m["machine_id"]: m for m in api("/machines")}
    except Exception as e:
        st.error(f"Cannot reach the API at {API} ({e})")
        machines = {}

    shovel_ids = ["SHOVEL-BW", "SHOVEL-OH", "SHOVEL-SD", "SHOVEL-OK"]
    present = [mid for mid in shovel_ids if mid in machines]
    if present:
        mid = present[0]
        status = api(f"/machines/{mid}")
        state_colors = {"ok": "#23402B", "watch": "#C77E2A",
                        "advisory": "#B4552D", "alert": "#A33327"}
        color = state_colors.get(status["state"], "#666")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<div class="big-health">{status["health"]:.0f}</div>',
                        unsafe_allow_html=True)
            st.caption("health score (0-100)")
            st.markdown(
                f'<span class="state-pill" style="background:{color}22;'
                f'color:{color}">{status["state"].upper()}</span>',
                unsafe_allow_html=True,
            )
            st.write(f"{status['alerts']} alerts raised")
        with c2:
            history = api(f"/machines/{mid}/history")
            if history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[h["timestamp"] for h in history],
                    y=[h["health"] for h in history],
                    mode="lines", line=dict(color="#C15F3C", width=2.5),
                ))
                for lvl, name in ((90, "watch"), (80, "advisory"), (60, "alert")):
                    fig.add_hline(y=lvl, line_dash="dash", line_color="gray",
                                  annotation_text=name)
                fig.update_layout(yaxis_range=[0, 105], height=260,
                                  margin=dict(l=10, r=10, t=20, b=10),
                                  xaxis_title="day", yaxis_title="health")
                st.plotly_chart(fig, width="stretch")
        alerts = api(f"/machines/{mid}/alerts")
        if alerts:
            st.subheader("What Sentinel saw")
            latest = alerts[-1]
            st.markdown(
                f"""<div class="verdict"><b>{latest["timestamp"][:16]}: 
                {latest["state"].upper()}</b><br>{latest["explanation"]}</div>""",
                unsafe_allow_html=True,
            )
            if len(alerts) > 1:
                with st.expander(f"Earlier alerts ({len(alerts) - 1})"):
                    for a in reversed(alerts[:-1]):
                        st.write(f"**{a['timestamp'][:16]} {a['state'].upper()}**: "
                                 f"{a['explanation']}")
        else:
            st.markdown(
                """<div class="verdict verdict-good"><b>No alerts.</b> This shovel
                stayed healthy for all 30 days.</div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("Press a simulate button above to put a shovel to work.")

# ------------------------------------------------------- validation strip ---
st.divider()
st.subheader("Does it actually work?")
st.write("Tested on public datasets. No marketing words, just the numbers.")
if VALIDATION:
    cmapss = VALIDATION["cmapss_fd001"]
    cwru = VALIDATION["cwru"]
    v1, v2, v3 = st.columns(3)
    v1.metric("Engines caught before failure",
              f"{cmapss['detected_before_eol']} of {cmapss['engines']}",
              "NASA turbofan data")
    v2.metric("Typical early warning",
              f"{cmapss['median_lead_time_alert_cycles']:.0f} operating cycles",
              "before failure")
    v3.metric("Bearing faults told apart",
              f"{cwru['auroc']:.0%}", "vibration data")
    with st.expander("How this was tested, and the limits"):
        st.write(VALIDATION["method"])
        for c in cmapss["caveats"] + cwru["caveats"]:
            st.caption("Note: " + c)
        st.caption("Reproduce it: `python scripts/prepare_data.py` downloads the "
                   "public datasets, then `python scripts/validate.py` scores them.")
else:
    st.caption("Run `python scripts/validate.py` to generate the validation numbers.")

# ------------------------------------------------------------- engineers ---
st.divider()
st.header("For engineers")
st.write("The machinery behind the demos. Jargon lives here, nowhere else.")

with st.expander("How it works: the pipeline"):
    st.write(
        """1. **Ingest.** Sensor readings arrive, one row per minute per machine.
2. **Features.** Readings are grouped into windows of statistics: mean, spread, min, max, trend per sensor.
3. **Detect.** Two detectors score each window against the machine's own baseline: an Isolation Forest for the multivariate fingerprint, and per-sensor z-scores for sharp spikes.
4. **Health.** Scores feed a health tracker, 0 to 100: ok, watch, advisory, alert.
5. **Explain.** When an alert fires, diagnostics find which sensors deviated, classify the likely fault, and write a grounded explanation.
6. **Limitations.** A new machine needs about 10 days of minute-resolution data before it is ready. Baselines are per machine, so different operating modes get blended. The z threshold is fixed, not learned per sensor. V2 is addressing these."""
    )

with st.expander("Inspect any machine"):
    try:
        eng_machines = api("/machines")
    except Exception:
        eng_machines = []
    if not eng_machines:
        if st.button("Load the demo pump fleet"):
            with st.spinner("Generating 30 days of pump telemetry..."):
                api("/demo/seed", method="post")
            st.rerun()
    else:
        by_id = {m["machine_id"]: m for m in eng_machines}
        mids = sorted(by_id)
        ready = [m for m in eng_machines if m["ready"]]
        rank = {"ok": 0, "watch": 1, "advisory": 2, "alert": 3}
        ranked = sorted(ready, key=lambda m: (rank.get(m["state"], 0), -m["health"]),
                        reverse=True)
        default = ranked[0]["machine_id"] if ranked else mids[0]
        mid = st.selectbox("Machine", mids, index=mids.index(default), key="eng_mid")
        status = api(f"/machines/{mid}")
        st.write(f"Health **{status['health']:.0f} / 100**, "
                 f"state **{status['state'].upper()}**, {status['alerts']} alerts.")
        hist = api(f"/machines/{mid}/history")
        if hist:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["timestamp"] for h in hist], y=[h["health"] for h in hist],
                mode="lines", line=dict(color="#C15F3C", width=2)))
            fig.update_layout(yaxis_range=[0, 105], height=260,
                              margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        eng_alerts = api(f"/machines/{mid}/alerts")
        for a in reversed(eng_alerts[-5:]):
            st.write(f"**{a['timestamp'][:16]} {a['state'].upper()}**: {a['explanation']}")
        with st.expander("How is this score calculated?"):
            try:
                bd = api(f"/machines/{mid}/score_breakdown")
                h = bd["health"]
                st.code(
                    f"""health = 100 * (1 - ewma)
ewma   = (1 - alpha) * previous_ewma + alpha * normalized_anomaly
alpha = {h['alpha']}
current ewma = {h['current_ewma']}
health = {h['health']:.0f}""",
                    language="text",
                )
                st.write(f"Trend: **{h['trend']}**. Watch below "
                         f"{h['thresholds']['watch']}, advisory below "
                         f"{h['thresholds']['advisory']}, alert below "
                         f"{h['thresholds']['alert']}.")
            except Exception as e:
                st.error(f"Could not load the score breakdown: {e}")

# ------------------------------------------------------------------ ingest ---
st.divider()
st.header("Try your own data")
st.markdown(badge("demo"), unsafe_allow_html=True)
st.write(
    "Upload telemetry and Sentinel will learn that machine's baseline and "
    "watch it for anomalies. **This is a demo uploader.** In production, "
    "ingest is configured per industry and per operation: sensor schemas, "
    "units, baseline windows, and alert thresholds are set to match the "
    "equipment, not assumed."
)

ind_key = st.selectbox(
    "Industry template",
    list(INDUSTRY_TEMPLATES),
    format_func=lambda k: f"{INDUSTRY_TEMPLATES[k]['label']} "
                          f"({INDUSTRY_TEMPLATES[k]['industry']})",
    key="ingest_industry",
)
tpl = INDUSTRY_TEMPLATES[ind_key]
st.write(f"**{tpl['label']}** expects these sensors:")
st.table(pd.DataFrame(
    [{"sensor": name, "unit": spec["unit"]} for name, spec in tpl["sensors"].items()]
))
if tpl["baseline_days"]:
    st.caption(f"Needs about {tpl['baseline_days']} days of minute-resolution data "
               "to learn a baseline before scoring.")
else:
    st.caption("Baseline window for this template is configured per deployment.")

with st.expander("CSV format"):
    st.write(
        "Columns: `timestamp, machine_id` plus every sensor above. "
        "One row per reading, timestamps parseable by pandas."
    )
    sample = pd.DataFrame([{
        "timestamp": "2026-01-01 00:00:00", "machine_id": "MY-MACHINE",
        **{name: spec["base"] for name, spec in tpl["sensors"].items()},
    }])
    st.download_button("Download a sample CSV", sample.to_csv(index=False),
                       f"sample_{ind_key}.csv", "text/csv")

uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"], key="ingest_csv")
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        required = ["timestamp", "machine_id", *tpl["sensors"]]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(
                f"Missing columns for the {tpl['label']} template: "
                f"{', '.join(missing)}"
            )
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
