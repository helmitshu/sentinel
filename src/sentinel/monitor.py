"""Live machine monitor: the piece that runs in production.


A MachineMonitor owns one machine's full pipeline: ingest readings, maintain
a rolling feature buffer, score with the Isolation Forest plus per-sensor
z-scores, track health, and log alerts with grounded explanations when the
state changes.

The detector is fitted on the first `baseline_readings` readings (the
commissioning period), exactly like the validation harness.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sentinel.detectors import IsolationForestDetector, ZScoreDetector
from sentinel.diagnostics import diagnose
from sentinel.explainer import TemplateExplainer
from sentinel.features import FEATURES, compute_features, feature_columns
from sentinel.health import HealthTracker
from sentinel.ingestion import validate
from sentinel.simulator import SENSORS


class MachineMonitor:
    def __init__(
        self,
        machine_id: str,
        window: int = 60,
        baseline_readings: int = 10 * 1440,
        sensors: list[str] | None = None,
        sensor_units: dict[str, str] | None = None,
    ):
        self.machine_id = machine_id
        self.window = window
        self.baseline_readings = baseline_readings
        self.sensors = list(sensors) if sensors else list(SENSORS)
        self.sensor_units = dict(sensor_units) if sensor_units else {}
        self._readings: deque[dict] = deque(maxlen=100_000)  # ~69 days at 1/min
        self._processed = 0  # readings consumed into feature windows
        self._detector: IsolationForestDetector | None = None
        self._zdetector: ZScoreDetector | None = None
        self._tracker: HealthTracker | None = None
        self._baseline_features: pd.DataFrame | None = None
        self._last_state = "ok"
        self._last_health = 100.0
        self._last_trend = "stable"
        self._worst_rank = 0  # worst state rank seen since the last ok (0=ok..3=alert)
        self._last_score = 0.0  # normalized anomaly score of the most recent window
        self._last_raw_score = 0.0  # unclipped IF score before median/p99 scaling
        self._last_z = 0.0  # max |z| across sensors in the most recent window
        self._last_z_norm = 0.0  # z component of the combined score, in [0, 1]
        self._last_if_score = 0.0  # IF component of the combined score, in [0, 1]
        self._last_features: dict | None = None  # feature row of the most recent window
        self._history: list[dict] = []  # per-window {timestamp, health, state}
        self.alerts: list[dict] = []
        self._explainer = TemplateExplainer()

    @property
    def ready(self) -> bool:
        return self._detector is not None

    def ingest(self, readings: list[dict]) -> dict:
        """Add raw readings; returns the machine's current status."""
        df = validate(pd.DataFrame(readings), sensors=self.sensors)
        if (df["machine_id"] != self.machine_id).any():
            raise ValueError("all readings must belong to this monitor's machine")
        before = len(self._readings)
        self._readings.extend(df.to_dict("records"))
        # the deque evicts from the left when full: keep _processed aligned
        # with the buffer's new left edge, and trim history to match
        dropped = max(0, before + len(df) - len(self._readings))
        if dropped:
            self._processed = max(0, self._processed - dropped)
            del self._history[:dropped]

        if not self.ready and len(self._readings) >= self.baseline_readings:
            self._calibrate()

        if self.ready:
            self._drain()

        return self.status()

    def _window_z(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per tumbling window: (z_norm in [0,1], max |z| across sensors).

        The z layer is the fast layer: it catches sharp per-reading spikes
        that a windowed multivariate model can smooth over. z_norm maps the
        detector threshold to 0.5 and twice the threshold to 1.0.
        """
        assert self._zdetector is not None
        max_z = self._zdetector.score(df)["max_z"].to_numpy()
        n_windows = len(max_z) // self.window
        # df always holds whole windows here (calibration baseline and drain
        # chunks are window multiples); trim a ragged tail defensively.
        trimmed = max_z[: n_windows * self.window].reshape(n_windows, self.window)
        per_window = trimmed.max(axis=1)
        z_norm = np.clip(per_window / (2 * self._zdetector.threshold), 0, 1)
        return z_norm, per_window

    def _calibrate(self) -> None:
        df = pd.DataFrame(list(self._readings)[: self.baseline_readings])
        feats = compute_features(df, window=self.window, sensors=self.sensors)
        cols = feature_columns(self.sensors)
        self._baseline_features = feats
        self._detector = IsolationForestDetector().fit(feats, columns=cols)
        self._zdetector = ZScoreDetector(sensors=self.sensors).fit(df)
        if_scores = self._detector.score(feats)
        z_norms, z_max = self._window_z(df)
        n = min(len(if_scores), len(z_norms))
        combined = np.maximum(if_scores[:n], z_norms[:n])
        baseline = float(combined.mean())
        self._tracker = HealthTracker(alpha=0.05, baseline=baseline)
        last = {"health": 100.0, "state": "ok"}
        raw_scored = -self._detector.model.decision_function(feats[cols])
        for v, raw in zip(combined, raw_scored):
            last = self._tracker.update(float(v))
        self._last_score = float(combined[-1])
        self._last_if_score = float(if_scores[n - 1])
        self._last_raw_score = float(raw_scored[n - 1])
        self._last_z = float(z_max[n - 1])
        self._last_z_norm = float(z_norms[n - 1])
        last_row = feats.iloc[-1].to_dict()
        self._last_features = {k: v for k, v in last_row.items()
                               if k not in ("timestamp", "machine_id")}
        self._processed = self.baseline_readings
        self._last_health = last["health"]
        self._last_state = last["state"]

    def _drain(self) -> None:
        assert self._detector is not None and self._tracker is not None
        df = pd.DataFrame(list(self._readings))
        # process whole new windows only (tumbling: one window per `window` readings)
        new_windows = (len(df) - self._processed) // self.window
        if new_windows < 1:
            return
        end = self._processed + new_windows * self.window
        chunk = df.iloc[self._processed : end]
        feats = compute_features(chunk, window=self.window, sensors=self.sensors)
        assert len(feats) == new_windows, f"{len(feats)} != {new_windows}"
        cols = self._detector._columns or feature_columns(self.sensors)
        raw_scores = -self._detector.model.decision_function(feats[cols])
        if_scores = self._detector.score(feats)
        z_norms, z_max = self._window_z(chunk)
        assert len(z_norms) == new_windows, f"{len(z_norms)} != {new_windows}"
        # either layer can raise the alarm independently: the health tracker
        # sees the worst credible signal from the two detectors
        scores = np.maximum(if_scores, z_norms)
        feat_rows = feats.to_dict("records")
        for i, (ts, score, if_s, raw, zm, zn, frow) in enumerate(
            zip(feats["timestamp"], scores, if_scores, raw_scores, z_max, z_norms, feat_rows)
        ):
            res = self._tracker.update(float(score))
            self._last_health, self._last_state = res["health"], res["state"]
            self._last_trend = res["trend"]
            self._last_score = float(score)
            self._last_if_score = float(if_s)
            self._last_raw_score = float(raw)
            self._last_z = float(zm)
            self._last_z_norm = float(zn)
            self._last_features = {k: v for k, v in frow.items()
                                   if k not in ("timestamp", "machine_id")}
            self._history.append(
                {"timestamp": str(ts), "health": res["health"], "state": res["state"]}
            )
            # check for a state transition on every window, not once per
            # ingest call: bulk ingestion (replay, CSV upload) must produce
            # the same alerts as a live stream.
            window_end = self._processed + (i + 1) * self.window
            self._maybe_alert(df, ts, window_end)
        self._processed = end

    def _maybe_alert(self, df: pd.DataFrame, ts, window_end: int) -> None:
        from sentinel.diagnostics import classify

        rank = {"ok": 0, "watch": 1, "advisory": 2, "alert": 3}
        state = self._last_state
        if state == "ok":
            # recovery: mention it once if we had warned before
            if self._worst_rank >= 2:
                self._record_alert(df, ts, window_end, state, recovered=True)
            self._worst_rank = 0
            return
        if state == "watch":
            return
        if rank[state] <= self._worst_rank:
            return  # already warned at this level or worse; no re-paging
        self._worst_rank = rank[state]
        self._record_alert(df, ts, window_end, state)

    def _record_alert(
        self, df: pd.DataFrame, ts, window_end: int, state: str,
        recovered: bool = False,
    ) -> None:
        from sentinel.diagnostics import classify

        if recovered:
            self.alerts.append({
                "timestamp": str(ts),
                "machine_id": self.machine_id,
                "state": state,
                "health": round(self._last_health, 1),
                "recovered": True,
                "explanation": f"{self.machine_id} has recovered to full health.",
                "findings": [],
            })
            return
        # state changed into advisory|alert: explain it from data
        start = max(0, window_end - 24 * self.window)
        recent = compute_features(
            df.iloc[start:window_end], window=self.window, sensors=self.sensors
        )
        findings, stable = diagnose(self._baseline_features, recent,
                                     sensors=self.sensors,
                                     sensor_units=self.sensor_units)
        hint, confidence = classify(findings)
        action = "inspect at the next planned stop"
        if "bearing" in hint:
            action = "schedule a bearing inspection within 7 days"
        elif "thermal" in hint or "overheat" in hint:
            action = "check cooling and loading, inspect within 48 hours"
        entry = {
            "timestamp": str(ts),
            "machine_id": self.machine_id,
            "state": state,
            "health": round(self._last_health, 1),
            "recovered": False,
            "explanation": self._explainer.explain(
                machine_id=self.machine_id,
                findings=findings,
                stable=stable,
                health=int(self._last_health),
                trend=self._last_trend,
                fault_hint=hint,
                confidence=confidence,
                window_days=1,
                action=action,
            ),
            "findings": [f.text for f in findings],
        }
        self.alerts.append(entry)

    def score_breakdown(self) -> dict:
        """Explain how the current scores were calculated, with real values.

        Returns the formulas plus the actual numbers that went into them,
        so a dashboard can show the working behind each score.
        """
        if not self.ready:
            raise ValueError("machine is still learning its baseline")
        det = self._detector
        trk = self._tracker
        assert det is not None and trk is not None
        # per-sensor baseline stats from the calibration period
        df = pd.DataFrame(list(self._readings)[: self.baseline_readings])
        sensors = {}
        for s in self.sensors:
            unit = SENSORS.get(s, {}).get("unit", "")
            sensors[s] = {
                "unit": unit,
                "baseline_mean": round(float(df[s].mean()), 3),
                "baseline_std": round(float(df[s].std()), 3),
                "last_value": round(float(df[s].iloc[-1]), 3)
                if len(df) else None,
            }
        # baseline vs current feature values for the most deviant features
        feat_base = {}
        feat_now = {}
        if self._baseline_features is not None and self._last_features:
            for col in feature_columns(self.sensors):
                feat_base[col] = round(float(self._baseline_features[col].mean()), 3)
                v = self._last_features.get(col)
                feat_now[col] = round(float(v), 3) if v is not None else None
        return {
            "health": {
                "formula": "health = 100 * (1 - ewma)",
                "ewma_formula": "ewma = (1 - alpha) * previous_ewma + alpha * normalized_anomaly",
                "normalization": "normalized = (anomaly - baseline) / (1 - baseline), capped at 1",
                "alpha": trk.alpha,
                "baseline_anomaly": round(trk.baseline, 4),
                "last_anomaly_score": round(self._last_score, 4),
                "current_ewma": round(trk._ewma, 4),
                "health": round(self._last_health, 1),
                "trend": self._last_trend,
                "thresholds": {"watch": 90, "advisory": 80, "alert": 60},
            },
            "anomaly": {
                "formula": "score = max(if_score, z_norm)",
                "description": "Two detectors vote; the health tracker sees the worst "
                               "credible signal. if_score is the Isolation Forest: how "
                               "far this window's feature fingerprint sits outside the "
                               "baseline distribution. z_norm is the fast z-score layer: "
                               "the sharpest per-reading spike inside the window.",
                "baseline_median": round(det._median, 4),
                "baseline_p99": round(det._p99, 4),
                "last_raw_score": round(self._last_raw_score, 4),
                "last_if_score": round(self._last_if_score, 4),
                "last_normalized_score": round(self._last_score, 4),
                "contamination": det.model.contamination,
                "window_readings": self.window,
            },
            "zscore": {
                "formula": "z_norm = clip(max_z / (2 * threshold), 0, 1)",
                "description": "Per-sensor z against the baseline mean/std; max |z| "
                               "across sensors and readings in the window. "
                               "z = threshold scores 0.5, z = 2 * threshold scores 1.0.",
                "threshold": self._zdetector.threshold if self._zdetector else None,
                "last_max_z": round(self._last_z, 3),
                "last_z_norm": round(self._last_z_norm, 4),
            },
            "sensors": sensors,
            "features": {
                "description": "Each 60-reading window becomes mean, std, min, max, slope per sensor.",
                "baseline": feat_base,
                "current_window": feat_now,
            },
        }

    def status(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "ready": self.ready,
            "health": round(self._last_health, 1),
            "state": self._last_state,
            "readings": len(self._readings),
            "alerts": len(self.alerts),
            "trend": self._last_trend,
            "anomaly_score": round(self._last_score, 4),
            "if_score": round(self._last_if_score, 4),
            "z_max": round(self._last_z, 3),
            "z_score": round(self._last_z_norm, 4),
        }

    def telemetry(self, n: int = 1440) -> list[dict]:
        rows = list(self._readings)[-n:]
        return [
            {"timestamp": str(r["timestamp"]), **{s: r[s] for s in self.sensors}}
            for r in rows
        ]

    def history(self) -> list[dict]:
        return list(self._history)
