"""Find demo seeds with clean alert stories (run once, then hardcode)."""
import sys
sys.path.insert(0, "src")
from sentinel.monitor import MachineMonitor
from sentinel.simulator import PumpSimulator


def run(seed, fault, fault_day):
    sim = PumpSimulator(seed=seed)
    df = sim.run(days=30, fault=fault, fault_start_day=fault_day)
    df["machine_id"] = "T"
    df["timestamp"] = df["timestamp"].astype(str)
    mon = MachineMonitor("T")
    for start in range(30):
        mon.ingest(df.iloc[start * 1440:(start + 1) * 1440].to_dict("records"))
    return [(a["timestamp"][:10], a["state"]) for a in mon.alerts]


def clean(alerts, fault_day_label):
    """No alert before the fault, no self-clear back to ok afterwards."""
    states = [s for _, s in alerts]
    if fault_day_label == "healthy":
        return len(alerts) == 0
    days = [int(d[8:10]) for d, _ in alerts]
    if any(d < fault_day_label for d in days):
        return False
    seen_bad = False
    for s in states:
        if s in ("advisory", "alert"):
            seen_bad = True
        elif s == "ok" and seen_bad:
            return False
    return seen_bad and states[-1] == "alert"


for role, fault, fday in (("P-101", None, 0), ("P-102", "bearing_wear", 14), ("P-103", "overheating", 20)):
    found = None
    for seed in range(1, 40):
        alerts = run(seed, fault, fday)
        if clean(alerts, "healthy" if fault is None else fday + 1):
            found = (seed, alerts)
            break
    print(role, "seed", found[0] if found else None)
    if found:
        for d, s in found[1]:
            print(f"    {d} {s}")
