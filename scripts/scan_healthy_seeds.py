"""Scan healthy seeds: how calm is each one's health chart? Prints compact stats."""
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
    return len(mon.alerts), min(hs), non_ok, mon.status()["state"]


seeds = [int(a) for a in sys.argv[1:]] or range(1, 21)
for seed in seeds:
    alerts, mn, non_ok, final = quality(seed)
    print(f"seed {seed}: alerts={alerts} min_health={mn:.0f} non_ok={non_ok:.3f} final={final}", flush=True)
