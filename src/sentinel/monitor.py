"""Live machine monitor: the piece that runs in production.

A MachineMonitor owns one machine's full pipeline: ingest readings, maintain
a rolling feature buffer, score with the Isolation Forest, track health, and
log alerts with grounded explanations when the state changes.

The detector is fitted on the first `baseline_readings` readings (the
commissioning period), exactly like the validation harness.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sentinel.detectors import IsolationForestDetector
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
    ):
        self.machine_id = machine_id
        self.window = window
        self.baseline_readings = baseline_readings
        self.sensors = list(sensors) if sensors else list(SENSORS)
        self._readings: deque[dict] = deque(maxlen=100_000)  # ~69 days at 1/min
        self._processed = 0  # readings consumed into feature windows
        self._detector: IsolationForestDetector | None = None
        self._tracker: HealthTracker | None = None
        self._baseline_features: pd.DataFrame | None = None
        self._last_state = "ok"
        self._last_health = 100.0
        self._last_trend = "stable"
        self._history: list[dict] = []  # per-window {timestamp, health, state}
        self.alerts: list[dict] = []
        self._explainer = TemplateExplainer()

    @property
    def ready(self) -> bool:
        return self._detector is not None

    def ingest(self, readings: list[dict]) -> dict:
        """Add raw readings; returns the machine's current status."""
        df = validate(pd.DataFrame(readings))
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

    def _calibrate(self) -> None:
        df = pd.DataFrame(list(self._readings)[: self.baseline_readings])
        feats = compute_features(df, window=self.window, sensors=self.sensors)
        cols = feature_columns(self.sensors)
        self._baseline_features = feats
        self._detector = IsolationForestDetector().fit(feats)
        baseline = float(self._detector.score(feats).mean())
        self._tracker = HealthTracker(alpha=0.05, baseline=baseline)
        last = {"health": 100.0, "state": "ok"}
        for v in self._detector.score(feats):
            last = self._tracker.update(v)
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
        scores = self._detector.score(feats)
        for ts, score in zip(feats["timestamp"], scores):
            res = self._tracker.update(float(score))
            self._last_health, self._last_state = res["health"], res["state"]
            self._last_trend = res["trend"]
            self._history.append(
                {"timestamp": str(ts), "health": res["health"], "state": res["state"]}
            )
        self._processed = end
        self._maybe_alert(df)

    def _maybe_alert(self, df: pd.DataFrame) -> None:
        from sentinel.diagnostics import classify

        state = self._last_state
        prev = self.alerts[-1]["state"] if self.alerts else "ok"
        if state == prev or state == "watch":
            return
        # state changed into/out of advisory|alert: explain it from data
        recent = compute_features(
            df.iloc[-24 * self.window :], window=self.window, sensors=self.sensors
        )
        findings, stable = diagnose(self._baseline_features, recent)
        hint, confidence = classify(findings)
        action = "inspect at the next planned stop"
        if "bearing" in hint:
            action = "schedule a bearing inspection within 7 days"
        elif "thermal" in hint or "overheat" in hint:
            action = "check cooling and loading, inspect within 48 hours"
        entry = {
            "timestamp": str(df["timestamp"].max()),
            "machine_id": self.machine_id,
            "state": state,
            "health": round(self._last_health, 1),
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

    def status(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "ready": self.ready,
            "health": round(self._last_health, 1),
            "state": self._last_state,
            "readings": len(self._readings),
            "alerts": len(self.alerts),
        }

    def telemetry(self, n: int = 1440) -> list[dict]:
        rows = list(self._readings)[-n:]
        return [
            {"timestamp": str(r["timestamp"]), **{s: r[s] for s in self.sensors}}
            for r in rows
        ]

    def history(self) -> list[dict]:
        return list(self._history)
