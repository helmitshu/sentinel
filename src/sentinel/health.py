"""Health scoring: turn anomaly scores into a 0-100 machine health score.

A single anomalous window means little; persistent anomaly means a lot. An
exponentially weighted moving average over the anomaly stream gives a smooth
score, and the slope of recent scores gives the trend a manager reads at a
glance.
"""
from __future__ import annotations

import numpy as np

WATCH = 90
ADVISORY = 80
ALERT = 60


class HealthTracker:
    """Stateful per-machine health. Feed it anomaly scores in [0, 1].

    baseline is the mean anomaly score on the machine's own training data.
    Health measures deviation from that baseline, so a healthy machine rests
    near 100 instead of drifting into a permanent yellow.
    """

    def __init__(self, alpha: float = 0.05, history_len: int = 200, baseline: float = 0.0):
        self.alpha = alpha
        self.history_len = history_len
        self.baseline = min(max(float(baseline), 0.0), 0.99)
        self._ewma = 0.0
        self._history: list[float] = []

    def update(self, anomaly: float) -> dict:
        anomaly = min(max(float(anomaly), 0.0), 1.0)
        # No lower clip: scores below baseline pull health back up, so the
        # baseline mean maps to health 100 in expectation. Upper clip only.
        norm = min((anomaly - self.baseline) / (1 - self.baseline), 1.0)
        self._ewma = (1 - self.alpha) * self._ewma + self.alpha * norm
        health = round(100 * (1 - self._ewma))
        health = min(max(health, 0), 100)
        self._history.append(float(health))
        self._history = self._history[-self.history_len :]

        if health >= WATCH:
            state = "ok"
        elif health >= ADVISORY:
            state = "watch"
        elif health >= ALERT:
            state = "advisory"
        else:
            state = "alert"

        return {"health": health, "trend": self._trend(), "state": state}

    def _trend(self) -> str:
        if len(self._history) < 10:
            return "stable"
        y = np.array(self._history[-20:])
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
        if slope < -0.5:
            return "degrading"
        if slope > 0.5:
            return "improving"
        return "stable"

    @property
    def history(self) -> list[float]:
        return list(self._history)
