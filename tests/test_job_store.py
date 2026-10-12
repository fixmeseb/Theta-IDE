"""Tests for JobStore persistence and process recovery."""
import time
from pathlib import Path

import pytest

from src.app.services.job_store import JobStore


def test_job_store_crud_and_queue(tmp_path):
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)

    # Save job
    job = {
        "job_id": "test-job-1",
        "experiment": "cartpole/ppo",
        "group": "thetaide",
        "experiment_id": "cartpole_1",
        "status": "pending",
        "pid": 999999,
        "created": time.time(),
        "agents": ["ppo"],
        "request": {"experiment": "cartpole/ppo"},
    }
    store.save_job(job)

    # Retrieve job
    loaded = store.get_job("test-job-1")
    assert loaded is not None
    assert loaded["job_id"] == "test-job-1"
    assert loaded["group"] == "thetaide"
    assert loaded["agents"] == ["ppo"]
    assert loaded["status"] == "pending"

    # Update job
    store.update_job("test-job-1", status="running", started=time.time())
    updated = store.get_job("test-job-1")
    assert updated["status"] == "running"
    assert updated["started"] is not None

    # Queue management
    store.set_queue(["test-job-1", "test-job-2"])
    queue = store.get_queue()
    assert queue == ["test-job-1", "test-job-2"]

    store.remove_from_queue("test-job-1")
    assert store.get_queue() == ["test-job-2"]


def test_job_store_recovers_dead_process(tmp_path):
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)

    # Job with a non-existent PID
    store.save_job({
        "job_id": "dead-job",
        "experiment": "cartpole/ppo",
        "group": "thetaide",
        "experiment_id": "cartpole_dead",
        "status": "running",
        "pid": 99999999,  # does not exist
        "created": time.time(),
    })

    recovered = store.recover_active_jobs()
    # Dead job should be marked failed, not recovered as alive
    assert "dead-job" not in recovered
    dead_job = store.get_job("dead-job")
    assert dead_job["status"] == "failed"
    assert "stopped" in dead_job["error"]


def test_job_store_connection_closed_on_error(tmp_path):
    import sqlite3
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    saved_conn = None
    with pytest.raises(RuntimeError):
        with store._connection() as conn:
            saved_conn = conn
            raise RuntimeError("Forced failure")
    assert saved_conn is not None
    with pytest.raises(sqlite3.ProgrammingError):
        saved_conn.execute("SELECT 1")
