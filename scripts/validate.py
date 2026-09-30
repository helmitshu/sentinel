"""Honest validation of Sentinel's detection core on real-world datasets.

No labels are used for training: the Isolation Forest learns healthy operation
from baseline data only, exactly as in production. Labels are used solely to
measure what happened.

C-MAPSS (NASA turbofan run-to-failure, FD001):
  Fit per engine on cycles 1..60 (early life, degradation negligible). Then
  stream the rest cycle by cycle through the health tracker and record:
    - detection lead time: cycles between first persistent warning/alert
      and end of life
    - wolf cries: alerts that later self-clear for 10+ cycles (an alert that
      fires on real degradation stays on until end of life)
  Persistence = 3 consecutive cycles, to avoid counting flicker.

CWRU (real bearing vibration, .mat):
  Fit on normal-baseline windows (files 97+98). Operating threshold = 99th
  percentile of baseline scores. Report held-out false positive rate on files
  99+100, detection rate per fault file, and AUROC over all windows.

Usage:
  .venv/bin/python scripts/validate.py [--quick] [--cmapss] [--cwru]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

from sentinel.detectors import IsolationForestDetector
from sentinel.features import compute_features
from sentinel.health import HealthTracker

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CMAPSS_COLS = (
    ["engine", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
# The 14 informative C-MAPSS sensors (standard practice: drop the constant or
# near-constant sensors 1, 5, 6, 10, 16, 18, 19).
CMAPSS_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
# C-MAPSS uses 5-cycle tumbling windows: with only ~150-300 cycles per
# engine, 30-cycle tumbling windows leave too few training samples for the
# Isolation Forest (2 windows from cycles 1..60). 5-cycle windows give 12
# independent training windows while keeping windows non-overlapping, which
# the pump-simulator experiments showed is required for a calibrated detector
# on noisy data.
WINDOW_CYCLES = 5
PERSIST = 3

# CWRU sample rates per the data center spec: normal baselines at 48 kHz,
# 12k drive-end fault data at 12 kHz. (99.mat skipped: the server returns
# 98.mat's content at that URL, so held-out normal is file 100 only.)
CWRU_FS = {str(n): 48000 for n in (97, 98, 100)}
CWRU_FS.update({str(n): 12000 for n in (105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133)})
CWRU_FAULTS = {
    "inner_race": ["105", "106", "107", "108"],
    "ball": ["118", "119", "120", "121"],
    "outer_race": ["130", "131", "132", "133"],
}


def _first_persistent(states: list[str], kinds: set[str], start: int) -> int | None:
    run = 0
    for i in range(start, len(states)):
        run = run + 1 if states[i] in kinds else 0
        if run >= PERSIST:
            return i
    return None


def validate_cmapss(quick: bool = False) -> dict:
    train = pd.read_csv(RAW / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLS)
    sensors = CMAPSS_SENSORS
    print(f"C-MAPSS sensors used ({len(sensors)}): {sensors}")

    engines = sorted(train["engine"].unique())
    if quick:
        engines = engines[:10]

    lead_warn, lead_alert, self_cleared, detected = [], [], 0, 0
    for eng in engines:
        df = train[train["engine"] == eng].copy()
        df["timestamp"] = df["cycle"]
        df["machine_id"] = f"engine_{eng}"
        feats = compute_features(df, window=WINDOW_CYCLES, sensors=sensors)
        cols = [c for c in feats.columns if c not in ("timestamp", "machine_id")]
        # tumbling windows: index i ends at cycle (i+1)*WINDOW_CYCLES
        n_fit = 60 // WINDOW_CYCLES  # windows ending cycles 5..60
        fit_feats = feats.iloc[:n_fit]
        det = IsolationForestDetector().fit(fit_feats, columns=cols)
        tracker = HealthTracker(alpha=0.05, baseline=float(det.score(fit_feats, columns=cols).mean()))
        states = [tracker.update(v)["state"] for v in det.score(feats, columns=cols)]
        eol = int(df["cycle"].max())

        # first persistent warning / alert anywhere after the fit period
        w = _first_persistent(states, {"advisory", "alert"}, n_fit)
        a = _first_persistent(states, {"alert"}, n_fit)
        if w is not None:
            lead_warn.append(eol - (w + 1) * WINDOW_CYCLES)
        if a is not None:
            detected += 1
            lead_alert.append(eol - (a + 1) * WINDOW_CYCLES)
            # did the alert ever self-clear for 10+ cycles (cry wolf)?
            rest = states[a + PERSIST :]
            run = 0
            for s in rest:
                run = run + 1 if s in ("ok", "watch") else 0
                if run >= 10:
                    self_cleared += 1
                    break

    res = {
        "engines": len(engines),
        "detected_before_eol": detected,
        "detection_rate": round(detected / len(engines), 3),
        "median_lead_time_warning_cycles": round(float(np.median(lead_warn)), 1) if lead_warn else None,
        "median_lead_time_alert_cycles": round(float(np.median(lead_alert)), 1) if lead_alert else None,
        "min_lead_time_alert_cycles": round(float(np.min(lead_alert)), 1) if lead_alert else None,
        "alerts_that_self_cleared": self_cleared,
    }
    print("\nC-MAPSS FD001 (unsupervised, fit on cycles 1..60 only):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


def _cwru_windows(f: str, window_s: float = 0.5) -> pd.DataFrame:
    """1-D vibration -> time-domain feature rows, one per window_s seconds."""
    mat = loadmat(RAW / "cwru" / f"{f}.mat")
    key = next(k for k in mat if k.endswith("DE_time"))
    sig = mat[key].ravel().astype(float)
    fs = CWRU_FS[f]
    wlen = int(fs * window_s)
    n = len(sig) // wlen
    rows = []
    for i in range(n):
        w = sig[i * wlen : (i + 1) * wlen]
        rms = float(np.sqrt(np.mean(w**2)))
        rows.append(
            {
                "rms": rms,
                "std": float(np.std(w)),
                "peak": float(np.max(np.abs(w))),
                "crest": float(np.max(np.abs(w)) / rms) if rms else 0.0,
                "kurtosis": float(kurtosis(w)),
                "skew": float(skew(w)),
            }
        )
    return pd.DataFrame(rows)


def validate_cwru(quick: bool = False) -> dict:
    cols = ["rms", "std", "peak", "crest", "kurtosis", "skew"]
    base = pd.concat([_cwru_windows(f) for f in ("97", "98")], ignore_index=True)
    det = IsolationForestDetector().fit(base, columns=cols)
    thresh = float(np.percentile(det.score(base, columns=cols), 99))

    heldout = _cwru_windows("100")
    fpr = float(np.mean(det.score(heldout, columns=cols) > thresh))

    per_file, all_scores, all_labels = {}, [], []
    fault_files = [f for group in CWRU_FAULTS.values() for f in group]
    if quick:
        fault_files = fault_files[:4]
    for group, files in CWRU_FAULTS.items():
        for f in files:
            if f not in fault_files:
                continue
            w = _cwru_windows(f)
            s = det.score(w, columns=cols)
            tpr = float(np.mean(s > thresh))
            per_file[f"{group}_{f}"] = round(tpr, 3)
            all_scores.extend(s.tolist())
            all_labels.extend([1] * len(s))
    hn = det.score(heldout, columns=cols)
    all_scores.extend(hn.tolist())
    all_labels.extend([0] * len(hn))
    auc = float(roc_auc_score(all_labels, all_scores))

    res = {
        "window_seconds": 0.5,
        "baseline_files": ["97", "98"],
        "heldout_normal_files": ["100"],
        "heldout_normal_windows": len(heldout),
        "heldout_false_positives": int(np.sum(det.score(heldout, columns=cols) > thresh)),
        "heldout_false_positive_rate": round(fpr, 4),
        "auroc": round(auc, 4),
        "detection_rate_per_file": per_file,
    }
    print("\nCWRU bearing vibration (unsupervised, fit on normal only):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cmapss", action="store_true")
    ap.add_argument("--cwru", action="store_true")
    args = ap.parse_args()
    run_all = not (args.cmapss or args.cwru)

    results = {}
    results_path = OUT / "validation_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    if args.cmapss or run_all:
        results["cmapss_fd001"] = validate_cmapss(args.quick)
    if args.cwru or run_all:
        results["cwru"] = validate_cwru(args.quick)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
