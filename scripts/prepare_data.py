"""Prepare real-world validation data for Sentinel.

Downloads (if missing) and verifies:
  - NASA C-MAPSS turbofan degradation data (FD001 subset): run-to-failure
    sensor histories for 100 engines. Public S3 mirror, no login needed.
  - CWRU bearing vibration recordings (.mat): normal baselines plus seeded
    inner-race, ball, and outer-race faults at 0.007" across motor loads.

Raw data lives under data/raw/ (gitignored). Nothing here trains a model;
see scripts/validate.py for the honest-metrics evaluation.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CWRU_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
# Note: 99.mat is skipped: the server returns 98.mat's content at that URL.
CWRU_FILES = [str(n) for n in (97, 98, 100, 105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133)]

CMAPSS_COLS = (
    ["engine", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _download(url: str, dest: Path) -> None:
    print(f"downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def prepare_cmapss() -> Path:
    d = RAW / "cmapss"
    d.mkdir(parents=True, exist_ok=True)
    needed = [d / f"{split}_FD001.txt" for split in ("train", "test")] + [d / "RUL_FD001.txt"]
    if not all(p.exists() for p in needed):
        zpath = d / "cmapss.zip"
        if not zpath.exists():
            _download(CMAPSS_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            inner = [n for n in z.namelist() if n.endswith("CMAPSSData.zip")][0]
            z.extract(inner, d)
        with zipfile.ZipFile(d / inner) as z2:
            for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
                src = [n for n in z2.namelist() if n.endswith(name)][0]
                with z2.open(src) as fh, open(d / name, "wb") as out:
                    out.write(fh.read())
    train = pd.read_csv(d / "train_FD001.txt", sep=r"\s+", header=None, names=CMAPSS_COLS)
    n_engines = train["engine"].nunique()
    print(f"C-MAPSS FD001 train: {len(train)} cycles across {n_engines} engines")
    const = [c for c in CMAPSS_COLS if train[c].std() == 0]
    print(f"constant columns (dropped from validation): {const}")
    return d


def prepare_cwru() -> Path:
    d = RAW / "cwru"
    d.mkdir(parents=True, exist_ok=True)
    for f in CWRU_FILES:
        p = d / f"{f}.mat"
        if not p.exists():
            _download(CWRU_URL.format(f), p)
    for f in CWRU_FILES:
        p = d / f"{f}.mat"
        keys = [k for k in loadmat(p) if k.endswith("DE_time")]
        assert keys, f"{f}.mat has no drive-end signal"
        sig = loadmat(p)[keys[0]].ravel()
        print(f"{f}.mat: {keys[0]} ({len(sig)} samples)")
    return d


def main() -> None:
    prepare_cmapss()
    print()
    prepare_cwru()
    print("\ndata ready under data/raw/ (gitignored)")


if __name__ == "__main__":
    main()
