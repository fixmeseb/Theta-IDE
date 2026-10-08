"""Tests for JobStore persistence, process recovery, and incremental telemetry."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.app.api.app as api
from src.app.api.app import app, _read_metrics_incremental
from src.app.api.job_store import JobStore


@pytest.fixture
def client():
    return TestClient(app)


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



def test_incremental_metrics_reader(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    # Write initial header + 2 rows
    csv_file.write_text("step,eval/reward,loss\n0,10.0,0.5\n50,15.0,0.4\n", encoding="utf-8")

    rows, next_byte, reset = _read_metrics_incremental(csv_file, last_byte=0)
    assert len(rows) == 2
    assert rows[0] == {"step": 0.0, "eval/reward": 10.0, "loss": 0.5}
    assert rows[1] == {"step": 50.0, "eval/reward": 15.0, "loss": 0.4}
    assert next_byte > 0
    assert not reset

    # Append 1 more row
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("100,22.0,0.3\n")

    # Read incrementally using next_byte
    new_rows, next_byte_2, reset = _read_metrics_incremental(csv_file, last_byte=next_byte)
    assert len(new_rows) == 1
    assert new_rows[0] == {"step": 100.0, "eval/reward": 22.0, "loss": 0.3}
    assert next_byte_2 > next_byte
    assert not reset

    # Append partial line (still being flushed)
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("150,25.0")  # no newline yet

    partial_rows, next_byte_3, _ = _read_metrics_incremental(csv_file, last_byte=next_byte_2)
    assert len(partial_rows) == 0
    assert next_byte_3 == next_byte_2


def test_telemetry_endpoint(client, monkeypatch):
    job_id = "pytest-telemetry-job"
    api.jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "group": "thetaide",
        "experiment_id": "test_exp",
        "agents": ["ppo"],
        "_log": ["step 10", "step 20", "step 30"],
        "log_dropped": 0,
    }
    api._sync_job(job_id)

    try:
        resp = client.get(f"/api/experiments/{job_id}/telemetry?since_log=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["job"]["status"] == "running"
        assert data["job"]["log"] == ["step 20", "step 30"]
        assert "hardware" in data
        assert "agents" in data
    finally:
        api.jobs.pop(job_id, None)
