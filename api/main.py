
"""Sentinel API: ingest telemetry, read machine health, review alerts."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from sentinel.industries import FAULTS, get_template
from sentinel.monitor import MachineMonitor
from sentinel.simulator import SENSORS, PumpSimulator, simulator_for
from sentinel.store import ReadingStore

store = ReadingStore()

monitors: dict[str, MachineMonitor] = {}


def _build_fleet() -> None:
    """Seed the demo fleet: 30 days of history for 3 pumps, one healthy,
    one developing bearing wear, one overheating. Idempotent."""
    # fixed seed 3 for all three: verified clean stories (P-101 no alerts;
    # P-102/P-103 alert after fault onset, no self-clearing). see
    # scripts/find_demo_seeds.py. hash() is randomized per process, so it
    # would give a different (possibly noisy) fleet on every restart.
    fleet = [
        ("P-101", 3, None, None),
        ("P-102", 3, "bearing_wear", 14),
        ("P-103", 3, "overheating", 20),
    ]
    for machine_id, seed, fault, day in fleet:
        # clear any previous readings so re-seeding never duplicates rows
        store.delete_machine(machine_id)
        if machine_id in monitors:
            del monitors[machine_id]
        sim = PumpSimulator(seed=seed)
        df = sim.run(days=30, fault=fault, fault_start_day=day)
        df["machine_id"] = machine_id
        df["timestamp"] = df["timestamp"].astype(str)
        mon = MachineMonitor(machine_id)
        monitors[machine_id] = mon
        # ingest day by day, like a real stream, so alerts tell the story
        for start in range(0, 30):
            chunk = df.iloc[start * 1440 : (start + 1) * 1440]
            records = chunk.to_dict("records")
            mon.ingest(records)
            store.save(records)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild monitors from durable storage. The replay is deterministic:
    # same readings in the same order produce the same health and alerts.
    for machine_id in store.machine_ids():
        mon = MachineMonitor(machine_id)
        monitors[machine_id] = mon
        mon.ingest(store.load(machine_id))
    if not monitors:
        # first boot (or the database was wiped): seed the demo fleet so
        # the dashboard is never blank.
        _build_fleet()
    yield


app = FastAPI(
    title="Sentinel",
    description="Predictive maintenance engine",
    lifespan=lifespan,
)


class Reading(BaseModel):
    """One telemetry reading. Sensor fields are free-form: the industry
    template on the batch decides which sensors are required."""

    model_config = ConfigDict(extra="allow")

    timestamp: str
    machine_id: str


class IngestBatch(BaseModel):
    readings: list[Reading]
    industry: str | None = None  # industry template id; defaults to "pump"


def _monitor(machine_id: str) -> MachineMonitor:
    if machine_id not in monitors:
        raise HTTPException(404, f"unknown machine {machine_id}")
    return monitors[machine_id]


@app.post("/ingest")
def ingest(batch: IngestBatch):
    industry = batch.industry or "pump"
    try:
        tpl = get_template(industry)
    except ValueError as e:
        raise HTTPException(400, str(e))
    sensors = list(tpl["sensors"])
    sensor_units = {name: spec["unit"] for name, spec in tpl["sensors"].items()}
    by_machine: dict[str, list[dict]] = {}
    for r in batch.readings:
        by_machine.setdefault(r.machine_id, []).append(r.model_dump())
    statuses = []
    for machine_id, readings in by_machine.items():
        mon = monitors.get(machine_id)
        if mon is None:
            mon = MachineMonitor(machine_id, sensors=sensors,
                                 sensor_units=sensor_units)
            monitors[machine_id] = mon
        try:
            statuses.append(mon.ingest(readings))
        except ValueError as e:
            raise HTTPException(400, str(e))
        # the durable store keeps the pump demo fleet; other industries are
        # ephemeral demos, regenerated on demand
        if industry == "pump":
            store.save(readings)
    return {"machines": statuses}


@app.get("/machines")
def list_machines():
    return [m.status() for m in monitors.values()]


@app.get("/machines/{machine_id}")
def machine_status(machine_id: str):
    return _monitor(machine_id).status()


@app.get("/machines/{machine_id}/telemetry")
def telemetry(machine_id: str, n: int = 1440):
    return _monitor(machine_id).telemetry(n=min(n, 20_000))


@app.get("/machines/{machine_id}/alerts")
def alerts(machine_id: str):
    return _monitor(machine_id).alerts


@app.get("/machines/{machine_id}/history")
def history(machine_id: str):
    """Per-window health/state history for charts."""
    return _monitor(machine_id).history()


@app.get("/machines/{machine_id}/score_breakdown")
def score_breakdown(machine_id: str):
    """How each score was calculated: formulas plus the actual values used."""
    try:
        return _monitor(machine_id).score_breakdown()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/demo/seed")
def seed():
    """Ensure the demo fleet is loaded (replaces the three demo machines,
    leaves any uploaded machines alone)."""
    _build_fleet()
    return [m.status() for m in monitors.values()]


class Scenario(BaseModel):
    fault: str = "bearing_wear"  # bearing_wear | overheating | sensor_drift | healthy
    industry: str = "pump"  # pump | mining_shovel


# (industry, fault) -> machine id. Pump ids are the originals, kept stable.
SCENARIO_IDS = {
    ("pump", "bearing_wear"): "SIM-BW",
    ("pump", "overheating"): "SIM-OH",
    ("pump", "sensor_drift"): "SIM-SD",
    ("pump", "healthy"): "SIM-OK",
    ("mining_shovel", "bearing_wear"): "SHOVEL-BW",
    ("mining_shovel", "overheating"): "SHOVEL-OH",
    ("mining_shovel", "sensor_drift"): "SHOVEL-SD",
    ("mining_shovel", "healthy"): "SHOVEL-OK",
}
# fixed seeds per scenario so the story is the same on every click
SCENARIO_SEEDS = {
    ("pump", "bearing_wear"): 11,
    ("pump", "overheating"): 12,
    ("pump", "sensor_drift"): 13,
    ("pump", "healthy"): 14,
    ("mining_shovel", "bearing_wear"): 21,
    ("mining_shovel", "overheating"): 22,
    ("mining_shovel", "sensor_drift"): 23,
    ("mining_shovel", "healthy"): 24,
}


@app.post("/demo/scenario")
def scenario(sc: Scenario):
    """One-click demo: simulate 30 days for a new machine with the chosen
    fault (or none) and ingest it. Re-running the same scenario replaces
    that machine. Lets a visitor watch the full detect-and-explain loop
    without preparing their own data. The industry selects the sensor
    schema and fault story (e.g. a mining shovel instead of a pump)."""
    fault = sc.fault if sc.fault != "healthy" else None
    key = (sc.industry, sc.fault)
    if key not in SCENARIO_IDS:
        raise HTTPException(400, f"unknown scenario {sc.fault!r} for industry {sc.industry!r}")
    try:
        tpl = get_template(sc.industry)  # validates the industry id
    except ValueError as e:
        raise HTTPException(400, str(e))
    machine_id = SCENARIO_IDS[key]
    store.delete_machine(machine_id)
    if machine_id in monitors:
        del monitors[machine_id]
    sim = simulator_for(sc.industry, machine_id, seed=SCENARIO_SEEDS[key])
    df = sim.run(days=30, fault=fault, fault_start_day=14)
    df["machine_id"] = machine_id
    df["timestamp"] = df["timestamp"].astype(str)
    mon = MachineMonitor(
        machine_id,
        sensors=list(tpl["sensors"]),
        sensor_units={n: sp["unit"] for n, sp in tpl["sensors"].items()},
    )
    monitors[machine_id] = mon
    for start in range(30):
        chunk = df.iloc[start * 1440 : (start + 1) * 1440]
        records = chunk.to_dict("records")
        mon.ingest(records)
        # the durable store keeps the pump demo fleet; other industries are
        # ephemeral demos, regenerated on demand
        if sc.industry == "pump":
            store.save(records)
    return mon.status()


@app.get("/health")
def health():
    return {"ok": True, "machines": len(monitors)}
