"""SQLite persistence layer for experiment jobs and execution queue."""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None


class JobStore:
    """Thread-safe SQLite storage for experiment jobs and queue order."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._get_connection()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment TEXT,
                    group_name TEXT,
                    experiment_id TEXT,
                    status TEXT,
                    pid INTEGER,
                    created REAL,
                    started REAL,
                    finished REAL,
                    returncode INTEGER,
                    error TEXT,
                    total_timesteps INTEGER,
                    effective_timesteps INTEGER,
                    agents_json TEXT,
                    request_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    position INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE
                )
            """)
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        agents = json.loads(d.pop("agents_json") or "[]")
        request = json.loads(d.pop("request_json") or "{}")
        group = d.pop("group_name")
        return {
            **d,
            "group": group,
            "agents": agents,
            "request": request,
        }

    def save_job(self, job: dict[str, Any], job_id: str | None = None) -> None:
        """Insert or replace a job record."""
        actual_job_id = job_id or job.get("job_id")
        if not actual_job_id:
            return
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_id, experiment, group_name, experiment_id, status, pid,
                    created, started, finished, returncode, error,
                    total_timesteps, effective_timesteps, agents_json, request_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    actual_job_id,
                    job.get("experiment"),
                    job.get("group"),
                    job.get("experiment_id"),
                    job.get("status", "pending"),
                    job.get("pid"),
                    job.get("created", time.time()),
                    job.get("started"),
                    job.get("finished"),
                    job.get("returncode"),
                    job.get("error"),
                    job.get("total_timesteps"),
                    job.get("effective_timesteps"),
                    json.dumps(job.get("agents", [])),
                    json.dumps(job.get("request", {})),
                ),
            )
            conn.commit()

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update specific fields of an existing job."""
        if not kwargs:
            return
        fields = []
        values = []
        for k, v in kwargs.items():
            col = "group_name" if k == "group" else k
            if col == "agents":
                fields.append("agents_json = ?")
                values.append(json.dumps(v))
            elif col == "request":
                fields.append("request_json = ?")
                values.append(json.dumps(v))
            else:
                fields.append(f"{col} = ?")
                values.append(v)
        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
        with self._lock, self._connection() as conn:
            conn.execute(sql, tuple(values))
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return self._row_to_dict(row) if row else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cur = conn.execute("SELECT * FROM jobs ORDER BY created DESC")
            return [self._row_to_dict(row) for row in cur.fetchall()]

    def get_queue(self) -> list[str]:
        with self._lock, self._connection() as conn:
            cur = conn.execute("SELECT job_id FROM queue ORDER BY position ASC")
            return [row["job_id"] for row in cur.fetchall()]

    def set_queue(self, queue_order: list[str]) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM queue")
            for job_id in queue_order:
                conn.execute("INSERT INTO queue (job_id) VALUES (?)", (job_id,))
            conn.commit()

    def push_queue(self, job_id: str) -> int:
        with self._lock, self._connection() as conn:
            conn.execute("INSERT OR IGNORE INTO queue (job_id) VALUES (?)", (job_id,))
            cur = conn.execute("SELECT count(*) as cnt FROM queue")
            conn.commit()
            return cur.fetchone()["cnt"] - 1

    def remove_from_queue(self, job_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM queue WHERE job_id = ?", (job_id,))
            conn.commit()

    def recover_active_jobs(self) -> list[str]:
        """Check jobs that were running/pending at shutdown. Reconnect or mark failed."""
        recovered = []
        with self._lock, self._connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE status IN ('pending', 'running')")
            active_jobs = [self._row_to_dict(row) for row in cur.fetchall()]

            for job in active_jobs:
                pid = job.get("pid")
                alive = False
                if pid:
                    if psutil is not None:
                        try:
                            proc = psutil.Process(pid)
                            alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            alive = False
                    else:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except OSError:
                            alive = False

                if alive:
                    logger.info("Recovered active running job %s (pid %s)", job["job_id"], pid)
                    recovered.append(job["job_id"])
                else:
                    logger.info("Marking dead job %s as failed (pid %s no longer active)", job["job_id"], pid)
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', returncode = -1, finished = ?, error = ?
                        WHERE job_id = ?
                    """,
                        (time.time(), "Process terminated while server was stopped", job["job_id"]),
                    )

            conn.commit()
        return recovered
