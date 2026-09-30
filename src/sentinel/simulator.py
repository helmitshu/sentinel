"""Sensor stream simulator driven by industry templates.

Generates minute-resolution telemetry with daily operating cycles and noise,
plus injectable fault signatures. Seedable for reproducibility. The demo and
the test suite run on this; production use replaces it with real telemetry
through the same schema.

PumpSimulator keeps the original pump behavior exactly (same signals, same
defaults) so existing tests and demos are unaffected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.industries import (
    FAULTS,
    PUMP_FAULTS,
    PUMP_SENSORS,
    get_template,
    shape_curve,
)

# Backward-compatible alias: the original module-level sensor spec.
SENSORS = PUMP_SENSORS


class MachineSimulator:
    """Simulates one machine's telemetry from an industry template."""

    def __init__(
        self,
        machine_id: str = "M-001",
        seed: int = 42,
        sensors: dict[str, dict[str, float]] | None = None,
        fault_effects: dict[str, list[tuple[str, str, float, str]]] | None = None,
    ):
        self.machine_id = machine_id
        self.seed = seed
        self.sensors = sensors if sensors is not None else PUMP_SENSORS
        self.fault_effects = fault_effects if fault_effects is not None else PUMP_FAULTS

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
        for name, spec in self.sensors.items():
            signal[name] = spec["base"] + spec["cycle_amp"] * cycle
            noise_amp[name] = np.ones(n)

        if fault is not None:
            for sensor, kind, magnitude, shape in self.fault_effects.get(fault, []):
                curve = shape_curve(shape, p)
                if kind == "mult":
                    signal[sensor] = signal[sensor] * (1 + magnitude * curve)
                elif kind == "add":
                    signal[sensor] = signal[sensor] + magnitude * curve
                elif kind == "noise_mult":
                    noise_amp[sensor] = noise_amp[sensor] * (1 + magnitude * curve)
                else:  # pragma: no cover - guarded by template authoring
                    raise ValueError(f"unknown effect kind {kind!r}")

        data: dict[str, np.ndarray] = {}
        for name, spec in self.sensors.items():
            data[name] = signal[name] + rng.normal(0, spec["noise"], n) * noise_amp[name]

        df = pd.DataFrame(data)
        df.insert(0, "machine_id", self.machine_id)
        df.insert(
            0,
            "timestamp",
            pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="m"),
        )
        return df


class PumpSimulator(MachineSimulator):
    """Simulates one pump's telemetry. All randomness is seeded."""

    def __init__(self, machine_id: str = "P-104", seed: int = 42):
        super().__init__(
            machine_id=machine_id,
            seed=seed,
            sensors=PUMP_SENSORS,
            fault_effects=PUMP_FAULTS,
        )


def simulator_for(industry: str, machine_id: str, seed: int) -> MachineSimulator:
    """Build a simulator for an industry template id."""
    tpl = get_template(industry)
    return MachineSimulator(
        machine_id=machine_id,
        seed=seed,
        sensors=tpl["sensors"],
        fault_effects=tpl["faults"],
    )
