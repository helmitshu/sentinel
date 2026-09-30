"""Realistic sensor stream simulator for a centrifugal pump.

Generates minute-resolution telemetry with daily operating cycles and noise,
plus injectable fault signatures (bearing wear, overheating, sensor drift).
Seedable for reproducibility. The demo and the test suite both run on this;
production use replaces it with real telemetry through the same schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS: dict[str, dict[str, float]] = {
    "vibration": {"unit": "mm/s", "base": 2.1, "noise": 0.15, "cycle_amp": 0.20},
    "bearing_temp": {"unit": "C", "base": 62.0, "noise": 0.80, "cycle_amp": 2.0},
    "discharge_pressure": {"unit": "bar", "base": 4.2, "noise": 0.05, "cycle_amp": 0.10},
    "rpm": {"unit": "rpm", "base": 2950.0, "noise": 8.0, "cycle_amp": 15.0},
    "motor_current": {"unit": "A", "base": 42.0, "noise": 0.60, "cycle_amp": 1.0},
}

FAULTS = ("bearing_wear", "overheating", "sensor_drift")


class PumpSimulator:
    """Simulates one pump's telemetry. All randomness is seeded."""

    def __init__(self, machine_id: str = "P-104", seed: int = 42):
        self.machine_id = machine_id
        self.seed = seed

    def run(
        self,
        days: int = 30,
        fault: str | None = None,
        fault_start_day: int = 14,
    ) -> pd.DataFrame:
        """Return a DataFrame with timestamp, machine_id and one column per sensor.

        fault: one of FAULTS or None. The fault ramps from fault_start_day to
        the end of the run, mimicking progressive degradation. p goes 0 (healthy)
        to 1 (failed) across the fault window.
        """
        if fault is not None:
            if fault not in FAULTS:
                raise ValueError(f"unknown fault {fault!r}, expected one of {FAULTS}")
            if not 0 <= fault_start_day < days:
                raise ValueError("fault_start_day must be within the run")

        rng = np.random.default_rng(self.seed)
        n = days * 1440  # minute resolution
        t = np.arange(n)
        cycle = np.sin(2 * np.pi * t / 1440)  # daily operating cycle

        p = np.zeros(n)
        if fault is not None:
            start = fault_start_day * 1440
            p[start:] = np.linspace(0, 1, n - start)
        # Degradation is slow then fast: q stays subtle for most of the fault
        # window and accelerates toward failure, like real bearing wear.
        q = p**2

        signal: dict[str, np.ndarray] = {}
        noise_amp: dict[str, np.ndarray] = {}
        for name, spec in SENSORS.items():
            signal[name] = spec["base"] + spec["cycle_amp"] * cycle
            noise_amp[name] = np.ones(n)

        if fault == "bearing_wear":
            signal["vibration"] = signal["vibration"] * (1 + 1.3 * q**2)
            signal["bearing_temp"] = signal["bearing_temp"] + 15 * q**1.5
            noise_amp["vibration"] = 1 + 1.5 * q
            noise_amp["motor_current"] = 1 + 2.5 * q
        elif fault == "overheating":
            signal["bearing_temp"] = signal["bearing_temp"] + 25 * q
            signal["vibration"] = signal["vibration"] * (1 + 0.25 * q)
        elif fault == "sensor_drift":
            # A failing sensor, not a failing machine. Only one channel moves.
            # Drift is linear by nature, so it uses p, not the shaped q.
            signal["vibration"] = signal["vibration"] + 1.5 * p

        data: dict[str, np.ndarray] = {}
        for name, spec in SENSORS.items():
            data[name] = signal[name] + rng.normal(0, spec["noise"], n) * noise_amp[name]

        df = pd.DataFrame(data)
        df.insert(0, "machine_id", self.machine_id)
        df.insert(
            0,
            "timestamp",
            pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="m"),
        )
        return df
