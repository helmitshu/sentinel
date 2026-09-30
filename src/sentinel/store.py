"""SQLite persistence for ingested readings.

The monitors are in-memory; this store is the durable copy. On startup the
API replays every stored reading through a fresh monitor, which rebuilds
health, alerts, and explanations deterministically. If the store is empty
(first boot, or the database file was wiped), the API seeds the demo fleet
instead, so the dashboard is never blank.
"""
from __future__ import annotations

import os
import sqlite3

COLUMNS = (
    "machine_id",
    "timestamp",
    "vibration",
    "bearing_temp",
    "discharge_pressure",
    "rpm",
    "motor_current",
)


def default_path() -> str:
    return os.environ.get("SENTINEL_DB", "data/sentinel.db")


class ReadingStore:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS readings ("
            "machine_id TEXT, timestamp TEXT, vibration REAL, bearing_temp REAL, "
            "discharge_pressure REAL, rpm REAL, motor_current REAL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_readings_machine "
            "ON readings (machine_id, timestamp)"
        )
        self._conn.commit()

    def save(self, readings: list[dict]) -> int:
        """Append readings. Returns the number stored."""
        rows = [tuple(r[c] for c in COLUMNS) for r in readings]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO readings VALUES (?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def machine_ids(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT machine_id FROM readings ORDER BY machine_id"
        )
        return [r[0] for r in cur.fetchall()]

    def load(self, machine_id: str) -> list[dict]:
        """All readings for one machine, oldest first."""
        cur = self._conn.execute(
            "SELECT machine_id, timestamp, vibration, bearing_temp, "
            "discharge_pressure, rpm, motor_current FROM readings "
            "WHERE machine_id = ? ORDER BY timestamp",
            (machine_id,),
        )
        return [dict(zip(COLUMNS, row)) for row in cur.fetchall()]

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM readings")
        return int(cur.fetchone()[0])

    def delete_machine(self, machine_id: str) -> int:
        """Remove all stored readings for one machine. Returns rows removed."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM readings WHERE machine_id = ?", (machine_id,)
            )
        return cur.rowcount
