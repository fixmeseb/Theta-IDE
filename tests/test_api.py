import pytest
from fastapi.testclient import TestClient

from src.app.api.app import app, jobs


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_environments(client):
    response = client.get("/api/environments")
    assert response.status_code == 200
    data = response.json()
    assert "environments" in data
    env_names = [e["name"] for e in data["environments"]]
    # Should include cartpole and should NOT include _base
    assert any("cartpole" in name for name in env_names)
    assert not any(name.startswith("_") for name in env_names)


def test_list_methods(client):
    response = client.get("/api/methods")
    assert response.status_code == 200
    data = response.json()
    assert "registered_agents" in data
    assert "method_styles" in data
    assert "ppo" in data["registered_agents"]


def test_list_experiments_filters_base(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    exp_names = [e["name"] for e in data["experiments"]]
    # Base templates must be excluded
    for name in exp_names:
        assert not name.endswith("_base")
        assert "/_" not in name


def test_list_runs(client):
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)


def test_get_metrics_not_found(client):
    response = client.get("/api/runs/nonexistent_group/nonexistent_exp/nonexistent_agent/metrics")
    assert response.status_code == 404
    assert response.json()["detail"] == "Metrics not found"


def test_cancel_nonexistent_job(client):
    response = client.post("/api/experiments/nonexistent-job-id/cancel")
    assert response.status_code == 404


def test_cancel_inactive_job(client):
    job_id = "test-completed-job"
    jobs[job_id] = {"status": "completed"}
    try:
        response = client.post(f"/api/experiments/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "already" in data["message"]
    finally:
        jobs.pop(job_id, None)


def test_check_status_nonexistent(client):
    response = client.get("/api/experiments/fake-job-id/status")
    assert response.status_code == 404


def _write_run(root, group, experiment_id, *, env="cartpole", seed=7, versions=None):
    """A results/logs/<group>/<experiment_id> folder like the pipeline writes."""
    import json

    run_dir = root / "results" / "logs" / group / experiment_id
    agent_dir = run_dir / "ppo"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"env:\n  name: {env}\n  env_id: CartPole-v1\nseed: {seed}\ntotal_timesteps: 10000\nparadigm: online_rl\n"
        "methods:\n  ppo:\n    agent: ppo\n    model: dnn\n    lr: 0.0003\n    batch_size: 64\n    gamma: 0.99\n",
        encoding="utf-8")
    for number, (rewards, finished) in (versions or {0: ([10.0, 30.0, 20.0], True)}).items():
        version = agent_dir / f"version_{number}"
        version.mkdir(parents=True)
        runtime = {"start_time_iso": "2026-10-01T10:00:00+00:00", "training_time_seconds": 75.0}
        if finished:
            runtime["end_time_iso"] = "2026-10-01T10:01:15+00:00"
        (version / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
        lines = ["epoch,eval/reward,transitions"]
        for i, reward in enumerate(rewards, start=1):
            lines.append(f",{reward},{i * 1000}.0")
            lines.append(f"{i},{reward + 1000},")  # Lightning's duplicate epoch row, without transitions
        (version / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_list_runs_summarizes_each_run(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "sweeps", "cli_run")
    (data,) = client.get("/api/runs").json()["runs"]
    assert (data["group"], data["experiment_id"], data["metadata"]) == ("sweeps", "cli_run", {})
    assert data["config"]["env"] == "cartpole" and data["config"]["seed"] == 7
    assert data["config"]["methods"] == {"ppo": {"agent": "ppo", "model": "dnn", "lr": 0.0003, "batch_size": 64,
                                                 "gamma": 0.99}}
    (agent,) = data["agents"]
    assert agent["name"] == "ppo" and agent["training_time_seconds"] == 75.0
    # Rows without a step count (Lightning's duplicate epoch rows) are ignored, as in the GUI's charts.
    assert (agent["best_reward"], agent["latest_reward"], agent["timesteps"]) == (30.0, 20.0, 3000)
    assert data["finished"] is True


def test_list_runs_unfinished_and_unreadable(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "g", "unfinished", versions={0: ([5.0], False)})
    empty = tmp_path / "results" / "logs" / "g" / "empty"
    empty.mkdir(parents=True)
    (empty / "config.yaml").write_text(": not yaml : [", encoding="utf-8")
    runs = {r["experiment_id"]: r for r in client.get("/api/runs").json()["runs"]}
    assert runs["unfinished"]["finished"] is False and runs["unfinished"]["agents"][0]["finished"] is None
    assert runs["empty"] == {"group": "g", "experiment_id": "empty", "metadata": {}, "config": {}, "agents": [],
                             "finished": False}


def test_newest_version_is_chosen_by_number(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "g", "multi", versions={9: ([1.0], True), 10: ([99.0], True)})
    (data,) = client.get("/api/runs").json()["runs"]
    assert data["agents"][0]["best_reward"] == 99.0  # version_10, not version_9
    metrics = client.get("/api/runs/g/multi/ppo/metrics").json()
    assert metrics["source"].replace("\\", "/").endswith("version_10/metrics.csv")
