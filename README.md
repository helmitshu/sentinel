# Sentinel


Predictive maintenance engine. It watches equipment telemetry, learns what normal looks like, and flags failures before they happen, with plain English explanations for every alert.


Live demo: [dashboard](https://fantastic-miracle-production-65d8.up.railway.app) | [API](https://sentinel-production-ab25.up.railway.app)


## How it works


1. **Simulate or ingest.** A seeded pump simulator generates realistic telemetry (vibration, temperature, pressure, flow) with injectable faults, or load your own CSV.
2. **Extract features.** Rolling 60 reading tumbling windows produce statistical features per sensor.
3. **Detect.** Z-score and Isolation Forest detectors score each window for abnormality.
4. **Score health.** An EWMA health tracker (0 to 100) with per machine baseline calibration turns detections into ok, watch, advisory, and alert states.
5. **Explain.** A diagnostics layer finds which features deviated and classifies the likely fault, then a template explainer (with optional LLM) writes a grounded explanation.


## Live telemetry ingest

The next ingest layer speaks MQTT with Sparkplug B, the standard field protocol for industrial telemetry. A managed MQTT broker carries device messages, a Sparkplug B subscriber inside the API decodes them into the existing ingest schema, and the detectors, alerts, and dashboard work unchanged.

- **Automatic device registration.** Birth certificates announce each device's sensors and units, so new machines register themselves with no manual provisioning.
- **Known state, not stale data.** Death certificates mark a device offline the moment its connection drops, instead of letting the models score stale readings.
- **Ordered, timestamped payloads** with store and forward buffering, so the tumbling windows train on complete, correctly sequenced history even over unreliable site links.

Rollout starts with a simulated edge device publishing live telemetry, proving the full chain online before any field hardware is connected. Broker credentials are provided via environment variables and never stored in the repo.


## Validation on real data


Tested unsupervised on two public datasets, labels used only for scoring.


**NASA C-MAPSS (turbofan engines, FD001):** 98% detection rate, median alert lead time 47.5 cycles, zero alerts self cleared. Caveats: FD001 covers a single operating condition and a single fault mode. Lead times are in cycles, not wall clock time.


**CWRU bearing vibration:** AUROC 0.97. Inner race and outer race faults detected in 4 of 4 files each. Ball faults detected at 85 to 95% per file. One held out normal file flagged 1 of 20 windows.


## The windowing bug


An early version used overlapping windows (step 1), which produced 14,000 highly correlated training rows from 10 days of data. The Isolation Forest miscalibrated on the correlated rows and false alerted on healthy pumps. Switching to tumbling windows (step equals window size) fixed it. Lesson: effective sample size, not row count, calibrates the detector.


## Run it


```bash
pip install -r requirements.txt
pip install -e .
python scripts/demo.py        # the P-104 pump story
python -m pytest tests/ -q    # 14 tests
```


API: `uvicorn api.main:app`. Dashboard: `streamlit run dashboard/app.py`.


## Structure


- `src/sentinel/`: simulator, ingestion, features, detectors, health, diagnostics, explainer, monitor
- `api/`: FastAPI backend
- `dashboard/`: Streamlit dashboard
- `scripts/`: demo, data preparation, validation
- `tests/`: unit tests
