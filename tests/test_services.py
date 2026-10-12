"""In-process services the desktop app calls directly: the run catalog, config composition and metrics readers.

These cover what the FastAPI endpoints used to (GET /api/runs, POST /api/config/compose, ...), now as plain functions.
"""

import json

import pytest

from frontend.model import BASE_EXPERIMENT, METRIC_COLUMNS, Config, metric_points
from src.app.pipeline.compose import comparable_config, compose_experiment
from src.app.services import catalog, config_service
from src.app.services.metrics import MetricsTail, latest_metrics_csv, read_metrics_rows

# ── catalog ──────────────────────────────────────────────────────────────────


def test_environments_exclude_base_templates():
    names = [e["name"] for e in catalog.list_environments()]
    assert "cartpole" in names
    assert not any(name.startswith("_") for name in names)


def test_methods_list_registered_agents():
    methods = catalog.list_methods()
    assert "ppo" in methods["registered_agents"]
    assert "method_styles" in methods


def test_experiments_exclude_base_templates():
    for name in (e["name"] for e in catalog.list_experiments()):
        assert not name.endswith("_base") and "/_" not in name


def write_run(root, group, experiment_id, *, env="cartpole", seed=7, versions=None):
    """A results/logs/<group>/<experiment_id> folder like the pipeline writes."""
    run_dir = root / group / experiment_id
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


@pytest.fixture
def logs(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setattr(catalog, "LOGS_DIR", root)
    return root


def test_list_runs_summarizes_each_run(logs):
    write_run(logs, "sweeps", "cli_run")
    (data,) = catalog.list_runs()
    assert (data["group"], data["experiment_id"], data["metadata"]) == ("sweeps", "cli_run", {})
    assert data["config"]["env"] == "cartpole" and data["config"]["seed"] == 7
    assert data["config"]["methods"] == {"ppo": {"agent": "ppo", "model": "dnn", "lr": 0.0003, "batch_size": 64,
                                                 "gamma": 0.99}}
    (agent,) = data["agents"]
    assert agent["name"] == "ppo" and agent["training_time_seconds"] == 75.0
    # Rows without a step count (Lightning's duplicate epoch rows) are ignored, as in the GUI's charts.
    assert (agent["best_reward"], agent["latest_reward"], agent["timesteps"]) == (30.0, 20.0, 3000)
    assert data["finished"] is True


def test_list_runs_unfinished_and_unreadable(logs):
    write_run(logs, "g", "unfinished", versions={0: ([5.0], False)})
    empty = logs / "g" / "empty"
    empty.mkdir(parents=True)
    (empty / "config.yaml").write_text(": not yaml : [", encoding="utf-8")
    runs = {r["experiment_id"]: r for r in catalog.list_runs()}
    assert runs["unfinished"]["finished"] is False and runs["unfinished"]["agents"][0]["finished"] is None
    assert runs["empty"] == {"group": "g", "experiment_id": "empty", "metadata": {}, "config": {}, "agents": [],
                             "finished": False}


def test_newest_version_is_chosen_by_number(logs):
    write_run(logs, "g", "multi", versions={9: ([1.0], True), 10: ([99.0], True)})
    (data,) = catalog.list_runs()
    assert data["agents"][0]["best_reward"] == 99.0  # version_10, not version_9
    metrics = catalog.run_metrics("g", "multi", "ppo")
    assert metrics["source"].replace("\\", "/").endswith("version_10/metrics.csv")


def test_run_metrics_missing_raises(logs):
    with pytest.raises(FileNotFoundError):
        catalog.run_metrics("nonexistent_group", "nonexistent_exp", "nonexistent_agent")


def test_no_logs_folder_means_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "missing")
    assert catalog.list_runs() == []


# ── config composition ───────────────────────────────────────────────────────


def gui_compose(config):
    return compose_experiment(BASE_EXPERIMENT, config.overrides())


def test_compose_report_matches_direct_composition():
    config = Config(name="seed_study-7", seed=7, total_timesteps=25000, lr=1e-6, batch_size=256, gamma=0.9)
    body = config_service.compose_report(BASE_EXPERIMENT, config.overrides())
    assert body["valid"], body["errors"]
    assert body["argv"] == ["python", "run_pipeline.py", BASE_EXPERIMENT, *config.overrides()]
    expected = comparable_config(gui_compose(config).cfg)
    body["config"].pop("experiment_name")
    assert body["config"] == expected
    train_overrides = body["methods"]["ppo"]["train_overrides"]
    assert f"++agent.lr={config.lr}" in train_overrides
    assert f"++agent.batch_size={config.batch_size}" in train_overrides


def test_compose_report_flags_an_invalid_override():
    body = config_service.compose_report(BASE_EXPERIMENT, ["seed=="])
    assert not body["valid"] and body["errors"][0]["stage"] == "compose"


def test_compose_report_flags_a_paradigm_violation():
    body = config_service.compose_report(BASE_EXPERIMENT, Config().overrides() + ["++methods.ppo.agent=cql"])
    assert not body["valid"] and body["errors"][0]["stage"] == "validation"


def test_compose_report_flags_an_unknown_experiment():
    assert not config_service.compose_report("thetaide/does_not_exist")["valid"]


def test_schema_defaults_come_from_the_pipeline_and_match_the_builder():
    schema = config_service.config_schema()
    assert schema["base_experiment"] == BASE_EXPERIMENT
    assert schema["environment"]["env_id"] == "CartPole-v1"
    defaults = {f["key"]: f["default"] for f in schema["fields"]}
    builder = Config()
    assert defaults["methods.ppo.lr"] == builder.lr
    assert defaults["methods.ppo.batch_size"] == builder.batch_size
    assert defaults["methods.ppo.gamma"] == builder.gamma
    assert defaults["tensorboard"] is builder.tensorboard
    assert set(schema["fixed_overrides"]) <= set(builder.overrides())


def test_compose_report_explains_a_rounded_budget():
    def notices(total):
        body = config_service.compose_report(BASE_EXPERIMENT, Config(total_timesteps=total).overrides())
        return [n for n in body["notices"] if "rollouts" in n]

    assert "trains 10,240 steps, not 10,000" in notices(10000)[0]
    assert notices(10240) == []


def test_prepare_launch_describes_the_job():
    plan = config_service.prepare_launch(BASE_EXPERIMENT, Config(name="pytest_plan_only", total_timesteps=3000).overrides())
    assert (plan["group"], plan["experiment_id"], plan["agents"]) == ("thetaide", "pytest_plan_only", ["ppo"])
    assert plan["effective_timesteps"] == 3072  # 3,000 requested, rounded up to 6 PPO rollouts of 512


def test_prepare_launch_refuses_a_clash_with_an_active_job():
    active = [{"job_id": "j1", "status": "running", "group": "thetaide", "experiment_id": "pytest_clash"}]
    with pytest.raises(config_service.LaunchError) as error:
        config_service.prepare_launch(BASE_EXPERIMENT, Config(name="pytest_clash").overrides(), active_jobs=active)
    assert error.value.conflict


# ── metrics readers ──────────────────────────────────────────────────────────


def test_metrics_reader_skips_partial_line_and_picks_newest_version(tmp_path):
    for version in (2, 10, 9):
        (tmp_path / f"version_{version}").mkdir()
        (tmp_path / f"version_{version}" / "metrics.csv").write_text("step\n", encoding="utf-8")
    newest = latest_metrics_csv(tmp_path)
    assert newest.parent.name == "version_10"
    newest.write_text("eval/reward,step,transitions\n20.5,0,0.0\n,1,512.0\n31.2,2,10", encoding="utf-8")
    rows = read_metrics_rows(newest)
    assert rows == [{"eval/reward": 20.5, "step": 0.0, "transitions": 0.0}, {"step": 1.0, "transitions": 512.0}]


def test_metrics_tail_reads_only_appended_complete_rows(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,eval/reward,loss\n0,10.0,0.5\n50,15.0,0.4\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    rows, reset = tail.read()
    assert rows == [{"step": 0.0, "eval/reward": 10.0, "loss": 0.5}, {"step": 50.0, "eval/reward": 15.0, "loss": 0.4}]
    assert not reset

    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("100,22.0,0.3\n")
    assert tail.read() == ([{"step": 100.0, "eval/reward": 22.0, "loss": 0.3}], False)

    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("150,25.0")  # still being flushed
    assert tail.read() == ([], False)
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write(",0.2\n")
    assert tail.read() == ([{"step": 150.0, "eval/reward": 25.0, "loss": 0.2}], False)


def test_metrics_tail_restarts_when_lightning_rewrites_the_file(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,loss\n0,0.5\n1,0.4\n2,0.3\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    tail.read()
    csv_file.write_text("step,loss,entropy\n0,0.5,0.6\n", encoding="utf-8")  # new column: rewritten, shorter
    rows, reset = tail.read()
    assert reset and rows == [{"step": 0.0, "loss": 0.5, "entropy": 0.6}]


def test_metric_points_keep_evaluation_and_training_rows_separate():
    rows = [{"eval/reward": 20.0, "eval/reward_std": 8.5, "transitions": 0.0},
            {"losses/total_loss": 1.5, "losses/entropy": 0.69, "losses/approx_kl": 0.001, "transitions": 512.0},
            {"eval/reward": 20.0, "step": 0.0},  # epoch-level duplicate without transitions
            {"time/train": 0.3, "transitions": 600.0}]
    evaluation, training = metric_points(rows)
    assert (evaluation["step"], evaluation["reward"], evaluation["reward_std"], evaluation["entropy"]) == (0, 20.0, 8.5, None)
    assert (training["step"], training["reward"], training["loss"], training["entropy"], training["approx_kl"]) == \
        (512, None, 1.5, 0.69, 0.001)
    assert set(evaluation) == set(training) == {"step", *METRIC_COLUMNS}
