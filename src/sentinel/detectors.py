"""Two-layer anomaly detection.

Fast layer: per-sensor rolling z-scores against baseline stats. Catches sudden
spikes within minutes.

Smart layer: an Isolation Forest over the multivariate feature vector. Learns
the fingerprint of healthy operation from baseline data and scores new windows
by how far they deviate. Unsupervised on purpose: real plants rarely have
labeled failure history, so a supervised classifier is not an honest option.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import feature_columns
from .simulator import SENSORS


class ZScoreDetector:
    """Per-sensor z-scores against baseline mean/std."""

    def __init__(self, threshold: float = 3.0, sensors: list[str] | None = None):
        self.threshold = threshold
        self.sensors = list(sensors) if sensors is not None else list(SENSORS)
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "ZScoreDetector":
        for sensor in self.sensors:
            self._mean[sensor] = float(df[sensor].mean())
            std = float(df[sensor].std())
            self._std[sensor] = std if std > 0 else 1e-9
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per reading: max |z| across sensors, plus per-sensor z."""
        out = pd.DataFrame(
            {"timestamp": df["timestamp"], "machine_id": df["machine_id"]}
        )
        for sensor in self.sensors:
            out[f"{sensor}_z"] = (
                df[sensor].to_numpy() - self._mean[sensor]
            ) / self._std[sensor]
        z_cols = [f"{s}_z" for s in self.sensors]
        out["max_z"] = out[z_cols].abs().max(axis=1)
        out["flagged"] = out["max_z"] > self.threshold
        return out


class IsolationForestDetector:
    """Multivariate anomaly scores in [0, 1], calibrated on baseline data."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state
        )
        self._median = 0.0
        self._p99 = 1.0
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, columns: list[str] | None = None) -> "IsolationForestDetector":
        """Fit on baseline feature rows.

        columns selects the feature columns; defaults to the pump feature set.
        Pass explicit columns to run on other telemetry (e.g. C-MAPSS sensors,
        CWRU vibration features).
        """
        self._columns = columns or feature_columns()
        self.model.fit(X[self._columns])
        raw = -self.model.decision_function(X[self._columns])  # higher = more anomalous
        self._median = float(np.median(raw))
        self._p99 = float(np.percentile(raw, 99))
        if self._p99 <= self._median:
            self._p99 = self._median + 1e-9
        return self

    def score(self, X: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        """Anomaly score per row: 0 = typical baseline, 1 = far outside it."""
        cols = columns or self._columns or feature_columns()
        raw = -self.model.decision_function(X[cols])
        return np.clip((raw - self._median) / (self._p99 - self._median), 0, 1)
