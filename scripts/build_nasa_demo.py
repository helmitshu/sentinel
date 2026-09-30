"""Precompute interactive NASA demo trajectories.

Runs the same unsupervised C-MAPSS pipeline as scripts/validate.py, but
records the per-window health trajectory for three representative engines
(earliest catch, median catch, latest catch) so the dashboard can replay
them interactively without recomputing.

Output: dashboard/nasa_demo.json
Usage: .venv/bin/python scripts/build_nasa_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.detectors import IsolationForestDetector
from sentinel.features import compute_features
from sentinel.health import HealthTracker
from validate import (
    CMAPSS_COLS,
    CMAPSS_SENSORS,
    PERSIST,
    RAW,
    WINDOW_CYCLES,
    _first_persistent,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dashboard" / "nasa_demo.json"


def engine_trajectory(eng: int) -> dict:
    train = pd.read_csv(
        RAW / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLS
    )
    df = train[train["engine"] == eng].copy()
    df["timestamp"] = df["cycle"]
    df["machine_id"] = f"engine_{eng}"
    feats = compute_features(df, window=WINDOW_CYCLES, sensors=CMAPSS_SENSORS)
    cols = [c for c in feats.columns if c not in ("timestamp", "machine_id")]

    n_fit = 60 // WINDOW_CYCLES
    fit_feats = feats.iloc[:n_fit]
    det = IsolationForestDetector().fit(fit_feats, columns=cols)
    tracker = HealthTracker(
        alpha=0.05, baseline=float(det.score(fit_feats, columns=cols).mean())
    )
    scores = det.score(feats, columns=cols)
    traj = []
    states = []
    for i, v in enumerate(scores):
        r = tracker.update(v)
        cycle_end = (i + 1) * WINDOW_CYCLES
        traj.append({"cycle": cycle_end, "health": r["health"], "state": r["state"]})
        states.append(r["state"])

    eol = int(df["cycle"].max())
    w = _first_persistent(states, {"advisory", "alert"}, n_fit)
    a = _first_persistent(states, {"alert"}, n_fit)
    return {
        "engine": int(eng),
        "eol": eol,
        "trajectory": traj,
        "first_warning_cycle": (w + 1) * WINDOW_CYCLES if w is not None else None,
        "first_alert_cycle": (a + 1) * WINDOW_CYCLES if a is not None else None,
        "lead_warning": eol - (w + 1) * WINDOW_CYCLES if w is not None else None,
        "lead_alert": eol - (a + 1) * WINDOW_CYCLES if a is not None else None,
    }


def main() -> None:
    train = pd.read_csv(
        RAW / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLS
    )
    engines = sorted(train["engine"].unique())
    print(f"scoring {len(engines)} engines for lead times...")
    leads = {}
    for eng in engines:
        t = engine_trajectory(eng)
        if t["lead_alert"] is not None:
            leads[eng] = t["lead_alert"]
    ordered = sorted(leads, key=leads.get)
    picks = {
        "early": int(ordered[-1]),
        "typical": int(ordered[len(ordered) // 2]),
        "late": int(ordered[0]),
    }
    print("picks (engine: alert lead cycles): " +
          ", ".join(f"{k}={v} ({leads[v]} cycles)" for k, v in picks.items()))

    demo = {
        "dataset": "NASA C-MAPSS FD001 (turbofan run-to-failure, real data)",
        "method": (
            "Unsupervised: Isolation Forest fit on cycles 1-60 only, then "
            "streamed cycle by cycle through the health tracker. No labels "
            "used for training."
        ),
        "engines": [engine_trajectory(eng) for eng in picks.values()],
        "labels": picks,
    }
    OUT.write_text(json.dumps(demo, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
