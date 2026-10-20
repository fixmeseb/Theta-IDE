"""GUI-to-CLI configuration equivalence (M0).

The ThetaIDE experiment builder expresses a config as Hydra overrides on a base recipe
(frontend.model.Config). These tests prove those overrides compose to exactly the same
config as hand-written recipes, and that the builder's exported recipe round-trips.
"""

import os
import subprocess
import sys

import pytest

from frontend.model import BASE_EXPERIMENT, Config
from src.app.pipeline.compose import comparable_config, compose_experiment, method_plans, validate_composed

REFERENCE = "thetaide/cartpole_ppo_reference"

CASES = [
    Config(),
    Config(name="seed_study-7", seed=7, total_timesteps=25000, lr=1e-6, batch_size=256, gamma=0.9),
    Config(name="123", seed=0, total_timesteps=1000, lr=0.5, batch_size=32, gamma=1.0, tensorboard=False),
]


def gui_compose(config):
    return compose_experiment(BASE_EXPERIMENT, config.overrides())


@pytest.fixture
def export_recipe(project_root):
    written = []

    def write(config):
        path = project_root / "in" / "config" / "experiment" / "thetaide" / f"_pytest_export_{len(written)}.yaml"
        path.write_text(config.recipe_yaml(), encoding="utf-8")
        written.append(path)
        return f"thetaide/{path.stem}"

    yield write
    for path in written:
        path.unlink(missing_ok=True)


def test_builder_command_passes_pipeline_dry_run(project_root):
    command = ["run_pipeline.py", BASE_EXPERIMENT, *Config().overrides(), "dry_run=true"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "KMP_USE_SHM": "0"}
    result = subprocess.run(
        [sys.executable, *command], cwd=project_root, capture_output=True, text=True, timeout=300, env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation Success" in result.stdout


def test_default_builder_matches_hand_written_reference():
    gui = gui_compose(Config())
    reference = compose_experiment(REFERENCE)
    assert comparable_config(gui.cfg) == comparable_config(reference.cfg)
    validate_composed(gui)


@pytest.mark.parametrize("config", CASES, ids=lambda c: c.name)
def test_exported_recipe_composes_like_builder_overrides(config, export_recipe):
    gui = gui_compose(config)
    exported = compose_experiment(export_recipe(config))
    assert comparable_config(gui.cfg) == comparable_config(exported.cfg)


@pytest.mark.parametrize("config", CASES, ids=lambda c: c.name)
def test_builder_values_reach_the_method_config(config):
    cfg = gui_compose(config).cfg
    assert cfg.experiment_id == config.name
    assert (cfg.seed, cfg.total_timesteps) == (config.seed, config.total_timesteps)
    method = cfg.methods.ppo
    assert (method.agent, method.model) == ("ppo", "dnn")
    assert (method.lr, method.batch_size, method.gamma) == (config.lr, config.batch_size, config.gamma)
    assert isinstance(method.lr, float) and isinstance(method.gamma, float)
    assert cfg.tensorboard is config.tensorboard


@pytest.mark.parametrize("total, extra, expected", [
    (10000, [], {"size": 512, "rollouts": 20, "timesteps": 10240}),
    (10240, [], {"size": 512, "rollouts": 20, "timesteps": 10240}),
    (1000, [], {"size": 512, "rollouts": 2, "timesteps": 1024}),
    (10000, ["++methods.ppo.num_envs=2"], {"size": 256, "rollouts": 40, "timesteps": 10240}),
])
def test_ppo_budget_rounds_up_to_whole_rollouts(total, extra, expected):
    rollout = method_plans(compose_experiment(BASE_EXPERIMENT, Config(total_timesteps=total).overrides() + extra))
    assert {k: rollout["ppo"]["rollout"][k] for k in expected} == expected
