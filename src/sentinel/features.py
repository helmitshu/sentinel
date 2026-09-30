"""Feature pipeline: turn raw sensor streams into model-ready windows.

For each sensor and each rolling window we compute mean, std, min, max and
slope. Slope matters most for degradation: a slow climb in vibration is the
signature the fixed thresholds miss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SENSORS

FEATURES = ("mean", "std", "min", "max", "slope")


def _slope(window: np.ndarray) -> float:
    x = np.arange(len(window))
    return float(np.polyfit(x, window, 1)[0])


def compute_features(
    df: pd.DataFrame,
    window: int = 60,
    sensors: list[str] | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Rolling features per sensor. One row per window, timestamped at window end.

    window is in readings (60 = one hour at minute resolution).
    sensors defaults to the pump sensor set; pass any list of numeric columns
    to run the pipeline on other telemetry.
    step is the stride between window starts in readings; None (default) means
    tumbling windows (step == window). Use step=1 for overlapping windows, but
    note overlapping windows are highly correlated: training an Isolation
    Forest on them miscalibrates it (the effective sample size is ~1/window
    of the row count), causing false positives on fresh healthy data.
    """
    sensors = list(sensors) if sensors is not None else list(SENSORS)
    if len(df) < window:
        raise ValueError(f"need at much as {window} readings, got {len(df)}")
    step = window if step is None else step

    rows: list[dict] = []
    values = {s: df[s].to_numpy() for s in sensors}
    for end in range(window, len(df) + 1, step):
        row: dict = {
            "timestamp": df["timestamp"].iloc[end - 1],
            "machine_id": df["machine_id"].iloc[end - 1],
        }
        for sensor in sensors:
            w = values[sensor][end - window : end]
            row[f"{sensor}_mean"] = float(np.mean(w))
            row[f"{sensor}_std"] = float(np.std(w))
            row[f"{sensor}_min"] = float(np.min(w))
            row[f"{sensor}_max"] = float(np.max(w))
            row[f"{sensor}_slope"] = _slope(w)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(sensors: list[str] | None = None) -> list[str]:
    sensors = list(sensors) if sensors is not None else list(SENSORS)
    return [f"{s}_{f}" for s in sensors for f in FEATURES]
