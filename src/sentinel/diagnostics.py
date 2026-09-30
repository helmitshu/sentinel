"""Diagnostics: turn feature deviations into plain-English findings.

The detector reacts to feature-level changes (a rising slope, growing
variability), not just raw averages. The explanation must describe what the
detector actually saw, or it will be vacuous early, exactly when it matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FEATURES
from .simulator import SENSORS


@dataclass
class Finding:
    sensor: str
    unit: str
    text: str
    severity: float  # |z| of the underlying feature deviation


def _phrase(sensor: str, unit: str, feature: str, recent: float, base: float, direction: str) -> str:
    name = sensor.replace("_", " ")
    if feature == "slope":
        # Slope is per reading; x60 gives per hour at minute resolution.
        return f"{name} trending {direction} ({recent * 60:+.2f} {unit}/h)"
    if feature in ("max", "min"):
        label = "peaks" if feature == "max" else "lows"
        return f"{name} {label} reaching {recent:.2f} {unit} (baseline {base:.2f})"
    label = "average" if feature == "mean" else "variability"
    pct = 100 * (recent - base) / base if base != 0 else 0.0
    return f"{name} {label} {direction} {abs(pct):.0f}% vs baseline"


def diagnose(
    baseline_feats: pd.DataFrame,
    recent_feats: pd.DataFrame,
    z_floor: float = 0.5,
    max_findings: int = 4,
) -> tuple[list[Finding], list[str]]:
    """Compare recent feature windows against baseline.

    Returns the most deviant features as plain-English findings, plus the
    sensors with nothing to report. The decision to explain comes from the
    health state machine; this function just describes the strongest evidence
    factually. Features below z_floor count as stable.
    """
    candidates: list[Finding] = []
    for sensor in SENSORS:
        unit = SENSORS[sensor]["unit"]
        for feature in FEATURES:
            col = f"{sensor}_{feature}"
            base = baseline_feats[col]
            base_mean, base_std = float(base.mean()), float(base.std())
            if base_std == 0:
                continue
            recent = float(recent_feats[col].mean())
            z = (recent - base_mean) / base_std
            if abs(z) >= z_floor:
                if feature == "slope":
                    direction = "upward" if z > 0 else "downward"
                else:
                    direction = "up" if z > 0 else "down"
                candidates.append(
                    Finding(sensor, unit, _phrase(sensor, unit, feature, recent, base_mean, direction), abs(z))
                )
    candidates.sort(key=lambda f: f.severity, reverse=True)
    findings = candidates[:max_findings]
    # Guarantee representation: if a flagged sensor didn't make the top-N,
    # include its strongest finding so the classification evidence is visible.
    represented = {f.sensor for f in findings}
    for f in candidates[max_findings:]:
        if f.sensor not in represented and len(findings) < 6:
            findings.append(f)
            represented.add(f.sensor)
    findings.sort(key=lambda f: f.severity, reverse=True)
    flagged = {f.sensor for f in candidates}
    stable = [s for s in SENSORS if s not in flagged]
    return findings, stable


def classify(findings: list[Finding]) -> tuple[str, str]:
    """Heuristic fault classification from the finding pattern.

    Returns (hint, confidence). The hint describes the evidence pattern and
    sharpens as evidence accumulates; it is a heuristic, not a diagnosis, and
    the findings are always shown alongside it.
    """
    if not findings:
        return "no clear fault", "low"
    sensors = [f.sensor for f in findings]
    unique = set(sensors)
    top = sensors[0]
    if len(unique) == 1 and len(findings) >= 2:
        return (
            f"possible {top.replace('_', ' ')} sensor issue rather than machine degradation",
            "medium",
        )
    if "vibration" in unique and "bearing_temp" in unique:
        return "early bearing wear", "high"
    if top in ("vibration", "motor_current"):
        return f"developing mechanical issue led by {top.replace('_', ' ')}", "medium"
    if top == "bearing_temp":
        return "developing thermal issue", "medium"
    return "developing anomaly", "low"
