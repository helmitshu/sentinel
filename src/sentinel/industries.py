"""Industry templates: sensor schemas and fault stories per industry.

A template describes what a machine in an industry looks like to Sentinel:
which sensors it streams, in what units, and how the demo faults express
themselves on those sensors. Production ingest is configured per industry
(and per operation inside an industry); these templates are the demo
starting points, clearly labeled as such in the dashboard.
"""
from __future__ import annotations

# Fault effect shapes, keyed by name. q is the shaped degradation curve
# (slow then fast, 0..1 across the fault window); p is the raw linear ramp.
# Each effect is (sensor, kind, magnitude, shape) where kind is one of:
#   "mult"       -> signal *= (1 + magnitude * shape)
#   "add"        -> signal += magnitude * shape
#   "noise_mult" -> noise amplitude *= (1 + magnitude * shape)
PUMP_SENSORS: dict[str, dict[str, float]] = {
    "vibration": {"unit": "mm/s", "base": 2.1, "noise": 0.15, "cycle_amp": 0.20},
    "bearing_temp": {"unit": "C", "base": 62.0, "noise": 0.80, "cycle_amp": 2.0},
    "discharge_pressure": {"unit": "bar", "base": 4.2, "noise": 0.05, "cycle_amp": 0.10},
    "rpm": {"unit": "rpm", "base": 2950.0, "noise": 8.0, "cycle_amp": 15.0},
    "motor_current": {"unit": "A", "base": 42.0, "noise": 0.60, "cycle_amp": 1.0},
}

PUMP_FAULTS: dict[str, list[tuple[str, str, float, str]]] = {
    "bearing_wear": [
        ("vibration", "mult", 1.3, "q2"),
        ("bearing_temp", "add", 15.0, "q15"),
        ("vibration", "noise_mult", 1.5, "q"),
        ("motor_current", "noise_mult", 2.5, "q"),
    ],
    "overheating": [
        ("bearing_temp", "add", 25.0, "q"),
        ("vibration", "mult", 0.25, "q"),
    ],
    "sensor_drift": [
        ("vibration", "add", 1.5, "p"),
    ],
}

# Rope shovel (surface mining): hoist and crowd machinery. Same statistical
# shapes as the pump template, relabeled to shovel equipment. Simulated.
SHOVEL_SENSORS: dict[str, dict[str, float]] = {
    "hoist_vibration": {"unit": "mm/s", "base": 3.4, "noise": 0.25, "cycle_amp": 0.35},
    "hoist_motor_temp": {"unit": "C", "base": 78.0, "noise": 1.20, "cycle_amp": 3.0},
    "crowd_motor_current": {"unit": "A", "base": 210.0, "noise": 3.0, "cycle_amp": 6.0},
    "swing_bearing_temp": {"unit": "C", "base": 65.0, "noise": 1.0, "cycle_amp": 2.5},
    "rope_tension": {"unit": "kN", "base": 420.0, "noise": 8.0, "cycle_amp": 20.0},
}

SHOVEL_FAULTS: dict[str, list[tuple[str, str, float, str]]] = {
    "bearing_wear": [
        ("hoist_vibration", "mult", 1.3, "q2"),
        ("hoist_motor_temp", "add", 18.0, "q15"),
        ("hoist_vibration", "noise_mult", 1.5, "q"),
        ("crowd_motor_current", "noise_mult", 2.0, "q"),
    ],
    "overheating": [
        ("hoist_motor_temp", "add", 28.0, "q"),
        ("swing_bearing_temp", "add", 20.0, "q"),
        ("hoist_vibration", "mult", 0.25, "q"),
    ],
    "sensor_drift": [
        ("rope_tension", "add", 30.0, "p"),
    ],
}

# Turbofan (aviation): the 14 informative C-MAPSS sensors. Used as the ingest
# template for aviation; the interactive NASA demo runs on precomputed
# trajectories, not on this template.
TURBOFAN_SENSORS: dict[str, dict[str, float]] = {
    f"sensor_{i}": {"unit": "raw", "base": 0.0, "noise": 1.0, "cycle_amp": 0.0}
    for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
}
TURBOFAN_FAULTS: dict[str, list[tuple[str, str, float, str]]] = {
    "bearing_wear": [],
    "overheating": [],
    "sensor_drift": [],
}

INDUSTRY_TEMPLATES: dict[str, dict] = {
    "pump": {
        "label": "Centrifugal pump",
        "industry": "Manufacturing / processing",
        "sensors": PUMP_SENSORS,
        "faults": PUMP_FAULTS,
        "baseline_days": 10,
        "demo_story": (
            "A process pump running around the clock. One stays healthy, "
            "one develops bearing wear, one starts overheating."
        ),
    },
    "mining_shovel": {
        "label": "Rope shovel",
        "industry": "Surface mining",
        "sensors": SHOVEL_SENSORS,
        "faults": SHOVEL_FAULTS,
        "baseline_days": 10,
        "demo_story": (
            "A rope shovel on a 12-hour shift rotation. Watch the hoist "
            "machinery: vibration climbs, the motor runs hot, and Sentinel "
            "flags the bearing before it takes the shovel offline."
        ),
    },
    "turbofan": {
        "label": "Turbofan engine",
        "industry": "Aviation",
        "sensors": TURBOFAN_SENSORS,
        "faults": TURBOFAN_FAULTS,
        "baseline_days": None,  # 60-cycle early-life fit, see validation
        "demo_story": (
            "Real run-to-failure data from NASA. Pick an engine and scrub "
            "through its life: Sentinel learns the first 60 cycles, then "
            "watches for the degradation it has never seen."
        ),
    },
}

FAULTS = ("bearing_wear", "overheating", "sensor_drift")


def get_template(industry: str) -> dict:
    """Return the template for an industry id, or raise ValueError."""
    try:
        return INDUSTRY_TEMPLATES[industry]
    except KeyError:
        raise ValueError(
            f"unknown industry {industry!r}, "
            f"expected one of {sorted(INDUSTRY_TEMPLATES)}"
        )


def shape_curve(kind: str, p: "np.ndarray"):
    """Map a shape name to a degradation curve over the linear ramp p."""
    q = p**2
    if kind == "p":
        return p
    if kind == "q":
        return q
    if kind == "q2":
        return q**2
    if kind == "q15":
        return q**1.5
    raise ValueError(f"unknown shape {kind!r}")
