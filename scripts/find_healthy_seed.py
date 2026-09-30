"""Find a healthy-machine seed with a calm health chart.

Criteria: zero alerts, min health >= 65, <5% of time outside 'ok',
final state 'ok'. Prints first seed that passes.
"""
import sys
sys.path.insert(0, "src")
from sentinel.monitor import MachineMonitor
from sentinel.simulator import PumpSimulator


def quality(seed: int):
    sim = PumpSimulator(seed=seed)
    df = sim.run(days=30, fault=None, fault_start_day=None)
    df["machine_id"] = "T"
    df["timestamp"] = df["timestamp"].astype(str)
    mon = MachineMonitor("T")
    for start in range(30):
        mon.ingest(df.iloc[start * 1440:(start + 1) * 1440].to_dict("records"))
    h = mon.history()
    hs = [x["health"] for x in h]
    non_ok = sum(1 for x in h if x["state"] != "ok") / len(h)
    return {
        "alerts": len(mon.alerts),
        "min_health": min(hs),
        "non_ok_frac": non_ok,
        "final": mon.status()["state"],
    }


for seed in range(4, 250):
    q = quality(seed)
    print(f"seed {seed}: alerts={q['alerts']} min={q['min_health']:.0f} "
          f"non_ok={q['non_ok_frac']:.3f} final={q['final']}", flush=True)
    if (q["alerts"] == 0 and q["min_health"] >= 65
            and q["non_ok_frac"] < 0.05 and q["final"] == "ok"):
        print(f"PASS seed {seed}")
        break
