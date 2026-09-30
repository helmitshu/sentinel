
"""Sentinel API: ingest telemetry, read machine health, review alerts."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.monitor import MachineMonitor
from sentinel.simulator import SENSORS, PumpSimulator
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
    timestamp: str
    machine_id: str
    vibration: float
    bearing_temp: float
    discharge_pressure: float
    rpm: float
    motor_current: float


class IngestBatch(BaseModel):
    readings: list[Reading]


def _monitor(machine_id: str) -> MachineMonitor:
    if machine_id not in monitors:
        raise HTTPException(404, f"unknown machine {machine_id}")
    return monitors[machine_id]


@app.post("/ingest")
def ingest(batch: IngestBatch):
    by_machine: dict[str, list[dict]] = {}
    for r in batch.readings:
        by_machine.setdefault(r.machine_id, []).append(r.model_dump())
    statuses = []
    for machine_id, readings in by_machine.items():
        mon = monitors.get(machine_id) or MachineMonitor(machine_id)
        monitors[machine_id] = mon
        statuses.append(mon.ingest(readings))
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
    """How each score was calculated: formulas plus the actual values used\."""
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


SCENARIO_IDS = {
    "bearing_wear": "SIM-BW",
    "overheating": "SIM-OH",
    "sensor_drift": "SIM-SD",
    "healthy": "SIM-OK",
}
# fixed seeds per scenario so the story is the same on every click
SCENARIO_SEEDS = {"bearing_wear": 11, "overheating": 12, "sensor_drift": 13, "healthy": 14}


@app.post("/demo/scenario")
def scenario(sc: Scenario):
    """One-click demo: simulate 30 days for a new machine with the chosen
    fault (or none) and ingest it. Re-running the same scenario replaces
    that machine. Lets a visitor watch the full detect-and-explain loop
    without preparing their own data."""
    fault = sc.fault if sc.fault != "healthy" else None
    if sc.fault not in SCENARIO_IDS:
        raise HTTPException(400, f"unknown scenario {sc.fault!r}")
    machine_id = SCENARIO_IDS[sc.fault]
    store.delete_machine(machine_id)
    if machine_id in monitors:
        del monitors[machine_id]
    sim = PumpSimulator(seed=SCENARIO_SEEDS[sc.fault])
    df = sim.run(days=30, fault=fault, fault_start_day=14)
    df["machine_id"] = machine_id
    df["timestamp"] = df["timestamp"].astype(str)
    mon = MachineMonitor(machine_id)
    monitors[machine_id] = mon
    for start in range(30):
        chunk = df.iloc[start * 1440 : (start + 1) * 1440]
        records = chunk.to_dict("records")
        mon.ingest(records)
        store.save(records)
    return mon.status()


@app.get("/health")
def health():
    return {"ok": True, "machines": len(monitors)}
