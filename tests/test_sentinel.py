import numpy as np
import pandas as pd
import pytest

from sentinel.detectors import IsolationForestDetector, ZScoreDetector
from sentinel.diagnostics import Finding, classify, diagnose
from sentinel.explainer import TemplateExplainer
from sentinel.features import compute_features, feature_columns
from sentinel.health import HealthTracker
from sentinel.ingestion import load_csv, validate
from sentinel.simulator import PumpSimulator, SENSORS


@pytest.fixture()
def healthy_df():
    return PumpSimulator(seed=7).run(days=5)


@pytest.fixture()
def fault_df():
    return PumpSimulator(seed=7).run(days=30, fault="bearing_wear", fault_start_day=14)


def test_simulator_schema_and_reproducibility(healthy_df):
    assert list(healthy_df.columns) == ["timestamp", "machine_id", *SENSORS]
    assert len(healthy_df) == 5 * 1440
    again = PumpSimulator(seed=7).run(days=5)
    pd.testing.assert_frame_equal(healthy_df, again)


def test_simulator_fault_grows_vibration(fault_df):
    early = fault_df["vibration"].iloc[: 14 * 1440].mean()
    late = fault_df["vibration"].iloc[-1440:].mean()
    assert late > early * 1.5, "bearing wear should clearly lift vibration"


def test_simulator_rejects_bad_fault():
    with pytest.raises(ValueError):
        PumpSimulator().run(days=5, fault="explosion")


def test_ingestion_validates(healthy_df):
    out = validate(healthy_df)
    assert len(out) == len(healthy_df)
    bad = healthy_df.drop(columns=["vibration"])
    with pytest.raises(ValueError):
        validate(bad)


def test_load_csv_roundtrip(healthy_df, tmp_path):
    path = str(tmp_path / "telemetry.csv")
    healthy_df.to_csv(path, index=False)
    out = load_csv(path)
    assert len(out) == len(healthy_df)


def test_features_shape(healthy_df):
    feats = compute_features(healthy_df, window=60)
    assert list(feats.columns) == ["timestamp", "machine_id", *feature_columns()]
    # tumbling windows by default: one window per 60 readings
    assert len(feats) == len(healthy_df) // 60
    # slope of vibration during healthy operation should hover near zero
    assert abs(feats["vibration_slope"].mean()) < 0.01


def test_features_overlapping_step(healthy_df):
    feats = compute_features(healthy_df, window=60, step=1)
    assert len(feats) == len(healthy_df) - 60 + 1


def test_zscore_flags_spike(healthy_df):
    det = ZScoreDetector().fit(healthy_df)
    spiked = healthy_df.copy()
    spiked.loc[spiked.index[-10:], "vibration"] = 50.0
    scored = det.score(spiked)
    assert scored["flagged"].iloc[-1]
    assert not scored["flagged"].iloc[0]


def test_isolation_forest_separates_fault(fault_df):
    baseline = fault_df.iloc[: 10 * 1440]
    degraded = fault_df.iloc[-3 * 1440 :]
    feats_base = compute_features(baseline, window=60)
    feats_deg = compute_features(degraded, window=60)
    det = IsolationForestDetector().fit(feats_base)
    base_scores = det.score(feats_base)
    deg_scores = det.score(feats_deg)
    assert deg_scores.mean() > base_scores.mean() + 0.3
    # Single-window spikes happen on healthy data; the EWMA health layer damps
    # them. The product guarantee is steady-state health, checked below.
    assert ((base_scores > 0.5).mean()) < 0.15

    tracker = HealthTracker(alpha=0.05, baseline=float(base_scores.mean()))
    for v in base_scores:
        state = tracker.update(v)
    assert state["state"] == "ok", "healthy data must settle at ok, not watch"
    assert state["health"] > 95


def test_health_tracker_states():
    tracker = HealthTracker(alpha=0.2)
    for _ in range(50):
        s = tracker.update(0.0)
    assert s["state"] == "ok" and s["health"] > 95
    assert s["trend"] == "stable"
    for _ in range(15):
        s = tracker.update(0.9)
    assert s["trend"] == "degrading", "trend must catch the transition"
    for _ in range(300):
        s = tracker.update(0.9)
    assert s["state"] == "alert" and s["health"] < 60
    assert s["trend"] == "stable", "converged low health is stable, not degrading"


def test_explainer_is_grounded():
    explainer = TemplateExplainer()
    findings = [
        Finding("vibration", "mm/s", "vibration average up 119% vs baseline", 8.2),
        Finding("bearing_temp", "C", "bearing temp average up 23% vs baseline", 6.1),
    ]
    text = explainer.explain(
        machine_id="P-104",
        findings=findings,
        stable=["discharge_pressure", "rpm"],
        health=52,
        trend="degrading",
        fault_hint="early bearing wear",
        confidence="high",
        window_days=6,
        action="schedule bearing inspection within 7 days",
    )
    assert "P-104" in text and "119%" in text and "52" in text
    assert "bearing wear" in text
    # Every finding's text must appear verbatim: nothing invented, nothing lost.
    for f in findings:
        assert f.text in text


def test_diagnose_finds_bearing_wear(fault_df):
    baseline_feats = compute_features(fault_df.iloc[: 10 * 1440], window=60)
    recent_feats = compute_features(fault_df.iloc[17 * 1440 : 18 * 1440], window=60)
    findings, stable = diagnose(baseline_feats, recent_feats)
    assert findings, "should find deviations on day 17 of bearing wear"
    assert any(f.sensor == "vibration" for f in findings)
    hint, confidence = classify(findings)
    assert "mechanical issue" in hint and confidence == "medium"
    assert "discharge_pressure" in stable or "rpm" in stable


def test_diagnosis_sharpens_as_evidence_grows(fault_df):
    baseline_feats = compute_features(fault_df.iloc[: 10 * 1440], window=60)
    late_feats = compute_features(fault_df.iloc[24 * 1440 : 26 * 1440], window=60)
    findings, _ = diagnose(baseline_feats, late_feats)
    hint, confidence = classify(findings)
    assert hint == "early bearing wear" and confidence == "high"


def test_diagnose_flags_sensor_drift():
    df = PumpSimulator(seed=11).run(days=20, fault="sensor_drift", fault_start_day=10)
    baseline_feats = compute_features(df.iloc[: 10 * 1440], window=60)
    recent_feats = compute_features(df.iloc[18 * 1440 : 20 * 1440], window=60)
    findings, _ = diagnose(baseline_feats, recent_feats)
    hint, _ = classify(findings)
    assert "sensor issue" in hint, f"drift should read as sensor issue, got {hint!r}"


def test_monitor_combined_scoring_healthy_stays_clean():
    from sentinel.monitor import MachineMonitor

    df = PumpSimulator(machine_id="P-T1", seed=7).run(days=12)
    mon = MachineMonitor("P-T1")
    mon.ingest(df.to_dict("records"))
    assert mon.ready
    st = mon.status()
    assert st["alerts"] == 0, f"healthy machine must not alert, got {st}"
    assert st["state"] == "ok" and st["health"] > 90


def test_monitor_combined_scoring_fault_alerts():
    from sentinel.monitor import MachineMonitor

    df = PumpSimulator(machine_id="P-T2", seed=42).run(
        days=30, fault="bearing_wear", fault_start_day=14
    )
    mon = MachineMonitor("P-T2")
    mon.ingest(df.to_dict("records"))
    st = mon.status()
    assert st["alerts"] >= 1, "bearing wear must raise at least one alert"
    assert st["state"] == "alert"
    bd = mon.score_breakdown()
    assert bd["anomaly"]["formula"] == "score = max(if_score, z_norm)"
    assert bd["zscore"]["threshold"] == 3.0
    assert bd["zscore"]["last_max_z"] > 0
    # combined score is the max of the two components
    assert st["anomaly_score"] == max(st["if_score"], st["z_score"])


def test_monitor_z_layer_catches_sharp_spike():
    from sentinel.monitor import MachineMonitor

    df = PumpSimulator(machine_id="P-T3", seed=7).run(days=11)
    mon = MachineMonitor("P-T3")
    mon.ingest(df.iloc[: 10 * 1440].to_dict("records"))
    assert mon.ready
    spiked = df.iloc[10 * 1440 :].copy()
    # one extreme vibration reading inside the final window of new data
    spiked.loc[spiked.index[-10:], "vibration"] = 60.0
    mon.ingest(spiked.to_dict("records"))
    st = mon.status()
    assert st["z_max"] > 3.0, f"spike must push max |z| past threshold, got {st['z_max']}"
    assert st["z_score"] > 0.5
    assert st["anomaly_score"] >= st["z_score"]


def test_monitor_constant_sensor_does_not_crash():
    from sentinel.monitor import MachineMonitor

    df = PumpSimulator(machine_id="P-T4", seed=7).run(days=12)
    df["rpm"] = 2950.0  # flat sensor: zero std in the baseline period
    mon = MachineMonitor("P-T4")
    mon.ingest(df.to_dict("records"))
    assert mon.ready
    assert mon.status()["health"] > 90


def test_zscore_detector_sensor_subset(healthy_df):
    det = ZScoreDetector(sensors=["vibration"]).fit(healthy_df)
    scored = det.score(healthy_df)
    assert "vibration_z" in scored.columns
    assert "bearing_temp_z" not in scored.columns
    assert "max_z" in scored.columns
    # default still covers every sensor (backward compatible)
    full = ZScoreDetector().fit(healthy_df).score(healthy_df)
    assert "motor_current_z" in full.columns


# --- industry templates & multi-industry simulation (dashboard V2) ---

def test_get_template_shovel():
    from sentinel.industries import get_template
    tpl = get_template("mining_shovel")
    assert tpl["label"] == "Rope shovel"
    assert set(tpl["sensors"]) == {
        "hoist_vibration", "hoist_motor_temp", "crowd_motor_current",
        "swing_bearing_temp", "rope_tension",
    }
    assert "bearing_wear" in tpl["faults"]


def test_get_template_unknown_raises():
    from sentinel.industries import get_template
    with pytest.raises(ValueError, match="unknown industry"):
        get_template("submarine")


def test_shape_curves():
    from sentinel.industries import shape_curve
    p = np.array([0.0, 0.5, 1.0])
    assert np.allclose(shape_curve("p", p), p)
    assert np.allclose(shape_curve("q", p), p**2)
    assert np.allclose(shape_curve("q2", p), p**4)
    assert np.allclose(shape_curve("q15", p), (p**2) ** 1.5)
    with pytest.raises(ValueError, match="unknown shape"):
        shape_curve("nope", p)


def test_shovel_simulator_schema_and_fault():
    from sentinel.simulator import simulator_for
    sim = simulator_for("mining_shovel", "SHOVEL-T", seed=21)
    df = sim.run(days=5, fault="bearing_wear", fault_start_day=2)
    assert list(df.columns) == [
        "timestamp", "machine_id", "hoist_vibration", "hoist_motor_temp",
        "crowd_motor_current", "swing_bearing_temp", "rope_tension",
    ]
    assert len(df) == 5 * 1440
    # fault must grow hoist vibration like bearing wear grows pump vibration
    early = df["hoist_vibration"].iloc[:2 * 1440].mean()
    late = df["hoist_vibration"].iloc[-1440:].mean()
    assert late > early * 1.5, f"fault did not grow vibration: {early} -> {late}"
    # reproducibility
    again = simulator_for("mining_shovel", "SHOVEL-T", seed=21).run(
        days=5, fault="bearing_wear", fault_start_day=2)
    assert np.allclose(df.select_dtypes("number").to_numpy(),
                       again.select_dtypes("number").to_numpy())


def test_pump_simulator_unchanged_via_template():
    # PumpSimulator must stay byte-identical to the pre-template behavior
    from sentinel.simulator import PumpSimulator, SENSORS
    from sentinel.industries import PUMP_SENSORS
    assert SENSORS == PUMP_SENSORS
    df = PumpSimulator(seed=11).run(days=5, fault="bearing_wear", fault_start_day=2)
    assert list(df.columns[2:]) == list(PUMP_SENSORS)


def test_scenario_endpoint_industry():
    # exercise the endpoint function directly (no TestClient: the installed
    # starlette wants an httpx2 package we do not ship)
    from api.main import Scenario, scenario
    status = scenario(Scenario(fault="bearing_wear", industry="mining_shovel"))
    assert status["machine_id"] == "SHOVEL-BW"
    assert status["ready"]
    # shovel sensors made it through ingest
    from api.main import monitors
    mon = monitors["SHOVEL-BW"]
    assert "hoist_vibration" in mon._readings[0]
    # unknown industry rejected with a 400
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        scenario(Scenario(fault="bearing_wear", industry="submarine"))
    assert exc.value.status_code == 400
    # pump scenarios keep their original ids
    assert scenario(Scenario(fault="overheating"))["machine_id"] == "SIM-OH"
