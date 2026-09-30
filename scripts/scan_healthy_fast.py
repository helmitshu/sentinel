"""Fast healthy-seed scan: calibrate, score, track health, print stats. One seed per argv."""
import sys
sys.path.insert(0, "src")
import numpy as np
from sentinel.simulator import PumpSimulator
from sentinel.features import compute_features
from sentinel.detectors import IsolationForestDetector
from sentinel.health import HealthTracker


def quality(seed: int):
    sim = PumpSimulator(seed=seed)
    df = sim.run(days=30, fault=None, fault_start_day=None)
    base = compute_features(df.iloc[:10 * 1440], window=60)
    det = IsolationForestDetector().fit(base)
    tracker = HealthTracker(alpha=0.05, baseline=float(det.score(base).mean()))
    # warm up on training scores like the monitor does
    for v in det.score(base):
        tracker.update(float(v))
    worst, non_ok, n = 100.0, 0, 0
    state_at_end = "ok"
    for day in range(10, 30):
        chunk = compute_features(df.iloc[day * 1440:(day + 1) * 1440], window=60)
        for v in det.score(chunk):
            r = tracker.update(float(v))
            worst = min(worst, r["health"])
            non_ok += r["state"] != "ok"
            n += 1
            state_at_end = r["state"]
    return worst, non_ok / n, state_at_end


for seed in [int(a) for a in sys.argv[1:]]:
    w, f, s = quality(seed)
    print(f"seed {seed}: min_health={w:.0f} non_ok={f:.3f} final={s}", flush=True)
