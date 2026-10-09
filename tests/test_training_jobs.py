"""Training job lifecycle behind the GUI's Launch / Stop buttons (M1)."""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.app.api.app as api
from frontend.model import BASE_EXPERIMENT, METRIC_COLUMNS, Config, metric_points
from src.app.api.app import LaunchRequest, _latest_metrics_csv, _read_metrics_rows, app, jobs


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def experiment_ids(project_root):
    """Unique experiment IDs whose results are removed after the test."""
    created = []

    def make(prefix="pytest_job"):
        experiment_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        created.append(experiment_id)
        return experiment_id

    yield make
    for experiment_id in created:
        for kind in ("logs", "checkpoints", "plots", "tensorboard"):
            shutil.rmtree(project_root / "results" / kind / "thetaide" / experiment_id, ignore_errors=True)


def overrides(experiment_id, total_timesteps):
    return Config(name=experiment_id, total_timesteps=total_timesteps).overrides() + ["no_plot=true"]


def wait_for(predicate, timeout, interval=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def test_launch_rejects_invalid_config(client):
    bad = Config().overrides() + ["++methods.ppo.agent=cql"]
    response = client.post("/api/experiments/launch", json={"experiment": BASE_EXPERIMENT, "overrides": bad})
    assert response.status_code == 422
    assert "forbidden" in response.json()["detail"]


def test_launch_refuses_to_overwrite_existing_results(client, project_root, experiment_ids):
    experiment_id = experiment_ids()
    log_dir = project_root / "results" / "logs" / "thetaide" / experiment_id
    log_dir.mkdir(parents=True)
    (log_dir / "config.yaml").write_text("previous run", encoding="utf-8")
    response = client.post("/api/experiments/launch",
                           json={"experiment": BASE_EXPERIMENT, "overrides": overrides(experiment_id, 1000)})
    assert response.status_code == 409
    assert (log_dir / "config.yaml").exists()


def test_job_cancelled_before_start_never_spawns(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Popen must not run for a cancelled job")

    monkeypatch.setattr(api.subprocess, "Popen", fail)
    job_id = "pytest-cancelled-before-start"
    jobs[job_id] = {"status": "cancelled", "experiment": BASE_EXPERIMENT}
    try:
        api.run_experiment_task(job_id, LaunchRequest(experiment=BASE_EXPERIMENT))
        assert jobs[job_id]["status"] == "cancelled"
    finally:
        jobs.pop(job_id, None)


def test_status_returns_log_lines_since_offset(client):
    job_id = "pytest-log-offset"
    jobs[job_id] = {"status": "running", "_log": ["c", "d"], "log_dropped": 2}
    try:
        body = client.get(f"/api/experiments/{job_id}/status?since=3").json()
        assert body["log"] == ["d"]
        assert body["log_total"] == 4
        assert "_log" not in body
    finally:
        jobs.pop(job_id, None)


def test_metrics_reader_skips_partial_line_and_picks_newest_version(tmp_path):
    for version in (2, 10, 9):
        (tmp_path / f"version_{version}").mkdir()
        (tmp_path / f"version_{version}" / "metrics.csv").write_text("step\n", encoding="utf-8")
    newest = _latest_metrics_csv(tmp_path)
    assert newest.parent.name == "version_10"
    newest.write_text("eval/reward,step,transitions\n20.5,0,0.0\n,1,512.0\n31.2,2,10", encoding="utf-8")
    rows = _read_metrics_rows(newest)
    assert rows == [{"eval/reward": 20.5, "step": 0.0, "transitions": 0.0}, {"step": 1.0, "transitions": 512.0}]


def test_metric_points_keep_evaluation_and_training_rows_separate():
    rows = [{"eval/reward": 20.0, "eval/reward_std": 8.5, "transitions": 0.0},
            {"losses/total_loss": 1.5, "losses/entropy": 0.69, "losses/approx_kl": 0.001, "transitions": 512.0},
            {"eval/reward": 20.0, "step": 0.0},  # epoch-level duplicate without transitions
            {"time/train": 0.3, "transitions": 600.0}]
    evaluation, training, timing = metric_points(rows)
    assert (evaluation["step"], evaluation["reward"], evaluation["reward_std"], evaluation["entropy"]) == (0, 20.0, 8.5, None)
    assert (training["step"], training["reward"], training["loss"], training["entropy"], training["approx_kl"]) == \
        (512, None, 1.5, 0.69, 0.001)
    assert set(evaluation) == set(training) == {"step", *METRIC_COLUMNS}
    # Columns outside METRIC_COLUMNS are kept under their own names, not dropped.
    assert (timing["step"], timing["time/train"], timing["reward"]) == (600, 0.3, None)


def test_metric_points_keep_metrics_from_any_engine():
    """SB3 (through ContractLogger) logs loss names the Lightning agents never use."""
    rows = [{"step": 1000.0, "transitions": 1000.0, "eval/reward": 15.0, "losses/train/value_loss": 3.5},
            {"transitions": 2000.0, "custom/success_rate": 0.4, "epoch": 3.0}]
    first, second = metric_points(rows)
    assert (first["reward"], first["losses/train/value_loss"]) == (15.0, 3.5)
    assert "step" in first and first["step"] == 1000  # the counter becomes the position, not a metric
    assert second["custom/success_rate"] == 0.4 and "epoch" not in second


@pytest.mark.slow
def test_launch_streams_logs_and_metrics_until_completed(client, experiment_ids):
    experiment_id = experiment_ids()
    job = client.post("/api/experiments/launch",
                      json={"experiment": BASE_EXPERIMENT, "overrides": overrides(experiment_id, 3000)}).json()
    assert (job["group"], job["experiment_id"], job["agents"]) == ("thetaide", experiment_id, ["ppo"])
    assert job["effective_timesteps"] == 3072  # 3,000 requested, rounded up to 6 PPO rollouts of 512

    status = wait_for(lambda: (s := client.get(f"/api/experiments/{job['job_id']}/status").json())["status"]
                      in api.TERMINAL_STATUSES and s, timeout=240)
    assert status["status"] == "completed", status["stdout"]
    assert status["returncode"] == 0
    assert any("Evaluation at" in line for line in status["log"])

    metrics = client.get(f"/api/experiments/{job['job_id']}/metrics").json()["agents"]["ppo"]
    points = metric_points(metrics["rows"])
    assert [p["step"] for p in points if p["reward"] is not None][-1] >= 3000
    assert any(p["loss"] is not None for p in points)
    assert max(p["step"] for p in points) == job["effective_timesteps"]  # the prediction matches real training
    later = client.get(f"/api/experiments/{job['job_id']}/metrics?since={metrics['total']}").json()
    assert later["agents"]["ppo"]["rows"] == []


@pytest.mark.slow
def test_cancel_stops_pipeline_and_training_subprocesses(client, experiment_ids):
    experiment_id = experiment_ids()
    job = client.post("/api/experiments/launch",
                      json={"experiment": BASE_EXPERIMENT, "overrides": overrides(experiment_id, 2000000)}).json()
    job_id = job["job_id"]
    wait_for(lambda: any("Evaluation at" in line
                         for line in client.get(f"/api/experiments/{job_id}/status").json()["log"]), timeout=240)

    assert client.post(f"/api/experiments/{job_id}/cancel").json()["status"] == "cancelled"
    status = wait_for(lambda: (s := client.get(f"/api/experiments/{job_id}/status").json())
                      .get("returncode") is not None and s, timeout=30)
    assert status["status"] == "cancelled"

    # train.py is a grandchild of the job; if it survived, its metrics file would keep growing.
    source = client.get(f"/api/experiments/{job_id}/metrics").json()["agents"]["ppo"]["source"]
    size = Path(source).stat().st_size
    time.sleep(4)
    assert Path(source).stat().st_size == size
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {status['pid']}"], capture_output=True, text=True)
        assert str(status["pid"]) not in listing.stdout
