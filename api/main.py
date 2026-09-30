"""Sentinel API: ingest telemetry, read machine health, review alerts."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.monitor import MachineMonitor
from sentinel.simulator import SENSORS, PumpSimulator

app = FastAPI(title="Sentinel", description="Predictive maintenance engine")

monitors: dict[str, MachineMonitor] = {}


class Reading(BaseModel):
    timestamp: str
    machine_id: str
    vibration: float
    bearing_temp: float
    motor_current: float
    pressure: float
    flow_rate: float


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


@app.post("/demo/seed")
def seed():
    """Build a demo fleet: 30 days of history for 3 pumps, one healthy,
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
        sim = PumpSimulator(seed=seed)
        df = sim.run(days=30, fault=fault, fault_start_day=day)
        df["machine_id"] = machine_id
        df["timestamp"] = df["timestamp"].astype(str)
        mon = MachineMonitor(machine_id)
        monitors[machine_id] = mon
        # ingest day by day, like a real stream, so alerts tell the story
        for start in range(0, 30):
            chunk = df.iloc[start * 1440 : (start + 1) * 1440]
            mon.ingest(chunk.to_dict("records"))
    return [m.status() for m in monitors.values()]


@app.get("/health")
def health():
    return {"ok": True, "machines": len(monitors)}
