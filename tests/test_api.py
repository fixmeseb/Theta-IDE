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


def test_metrics_returns_every_column_and_supports_versions_and_since(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, "g", "multi", versions={0: ([1.0, 2.0], True), 1: ([5.0, 6.0, 7.0], True)})
    newest = client.get("/api/runs/g/multi/ppo/metrics").json()
    assert newest["version"] == "version_1" and newest["versions"] == ["version_0", "version_1"]
    assert newest["columns"] == ["epoch", "eval/reward", "transitions"] and newest["total_rows"] == 6
    older = client.get("/api/runs/g/multi/ppo/metrics", params={"version": "0"}).json()
    assert older["version"] == "version_0" and older["total_rows"] == 4
    # `since` returns only rows after the ones the caller already has.
    tail = client.get("/api/runs/g/multi/ppo/metrics", params={"since": 4}).json()
    assert len(tail["metrics"]) == 2 and tail["metrics"][0]["eval/reward"] == "7.0"
    assert client.get("/api/runs/g/multi/ppo/metrics", params={"version": "9"}).status_code == 404


def _write_artifacts(root):
    run = _write_run(root, "g", "exp")
    plots = root / "results" / "plots" / "g" / "exp"
    (plots / "losses" / "ppo").mkdir(parents=True)
    (plots / "losses" / "ppo" / "losses_value_loss.png").write_bytes(b"\x89PNG fake")
    (plots / "time_report.csv").write_bytes(b"Method,Avg\nppo,1.0\n")
    (plots / "hyperparameters_report.md").write_text("# Report\n", encoding="utf-8")
    ckpt = root / "results" / "checkpoints" / "g" / "exp" / "ppo" / "0"
    ckpt.mkdir(parents=True)
    (ckpt / "best_model.ckpt").write_bytes(b"weights")
    tb = root / "results" / "tensorboard" / "g" / "exp" / "ppo" / "version_0"
    tb.mkdir(parents=True)
    (tb / "events.out.tfevents.1.host.0").write_bytes(b"tb")
    # Another run's file, which this run's file endpoint must refuse.
    other = root / "results" / "plots" / "g" / "other"
    other.mkdir(parents=True)
    (other / "secret.csv").write_text("x\n1\n", encoding="utf-8")
    return run


def test_manifest_lists_every_artifact_of_a_run(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_artifacts(tmp_path)
    data = client.get("/api/runs/g/exp/manifest").json()
    kinds = {a["path"]: a["kind"] for a in data["artifacts"]}
    assert kinds == {
        "logs/g/exp/config.yaml": "config",
        "logs/g/exp/ppo/version_0/metrics.csv": "metrics",
        "logs/g/exp/ppo/version_0/runtime.json": "metadata",
        "plots/g/exp/hyperparameters_report.md": "report",
        "plots/g/exp/losses/ppo/losses_value_loss.png": "figure",  # nested plots are found too
        "plots/g/exp/time_report.csv": "table",
        "checkpoints/g/exp/ppo/0/best_model.ckpt": "checkpoint",
        "tensorboard/g/exp/ppo/version_0/events.out.tfevents.1.host.0": "tensorboard",
    }
    assert data["agents"] == {"ppo": ["version_0"]} and data["truncated"] is False
    metrics = next(a for a in data["artifacts"] if a["kind"] == "metrics")
    assert (metrics["root"], metrics["agent"], metrics["version"]) == ("logs", "ppo", "version_0")
    assert metrics["size"] > 0 and metrics["modified"] > 0
    assert client.get("/api/runs/g/missing/manifest").status_code == 404


def test_run_files_are_served_only_from_that_run(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_artifacts(tmp_path)
    ok = client.get("/api/runs/g/exp/files/plots/g/exp/time_report.csv")
    assert ok.status_code == 200 and ok.content == b"Method,Avg\nppo,1.0\n"
    assert client.get("/api/runs/g/exp/files/plots/g/exp/nope.csv").status_code == 404
    # Another run's file, a path climbing out of results/, and bad segments are all refused.
    assert client.get("/api/runs/g/exp/files/plots/g/other/secret.csv").status_code == 403
    assert client.get("/api/runs/g/exp/files/plots/g/exp/..%2F..%2F..%2F..%2Fsecret.txt").status_code in (403, 404)
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    escaped = client.get("/api/runs/g/exp/files/logs/g/exp/../../../../secret.txt")
    assert escaped.status_code in (403, 404) and "top secret" not in escaped.text
    assert client.get("/api/runs/g/exp/files/" + str(tmp_path / "secret.txt")).status_code in (403, 404)
    assert client.get("/api/runs/g/..%5Cexp/manifest").status_code in (400, 404)
