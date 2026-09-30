"""Ingestion: validate and normalize telemetry into the engine's schema.

Every source, simulator, CSV upload, or live feed, lands in the same shape:
timestamp, machine_id, then one numeric column per sensor.
"""
from __future__ import annotations

import pandas as pd

from .simulator import SENSORS

REQUIRED_COLUMNS = ["timestamp", "machine_id", *SENSORS]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema and types. Returns a cleaned copy or raises ValueError."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("timestamp column contains unparseable values")

    for sensor in SENSORS:
        out[sensor] = pd.to_numeric(out[sensor], errors="coerce")
        if out[sensor].isna().any():
            raise ValueError(f"sensor column {sensor!r} contains non-numeric values")

    out = out.sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError("no readings provided")
    return out


def load_csv(path: str) -> pd.DataFrame:
    """Load telemetry from a CSV file and validate it."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}")
    return validate(df)
