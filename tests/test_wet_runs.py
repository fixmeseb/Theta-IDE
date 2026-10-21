"""End-to-End Wet Tests for Theta-IDE Pipeline.

Proves that:
1. If pre-flight validation passes (i.e. GUI Play button is enabled / compose_report is valid),
   the experiment executes and trains for real without runtime plumbing errors.
2. If invalid parameters, missing components, missing environments, or missing datasets are configured,
   pre-flight validation rejects them BEFORE any run can be launched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize

from src.app.pipeline.validation import find_missing_components
from src.app.services.config_service import compose_report


@pytest.fixture(scope="session")
def python_bin() -> str:
    backend_py = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python3"
    if backend_py.exists():
        return str(backend_py)
    return sys.executable


@pytest.mark.skipif(
    os.environ.get("THETA_RUN_WET_TESTS") != "1",
    reason="wet training tests require separately installed component plugins; opt in with THETA_RUN_WET_TESTS=1",
)
def test_wet_run_online_rl_cartpole(python_bin):
    """Wet test: Online RL with SB3 PPO on CartPole simulator runs for real."""
    exp_name = "thetaide/cartpole_ppo_reference"
    
    # 1. Pre-flight check must be valid
    report = compose_report(exp_name)
    assert report["valid"] is True, f"Expected valid config, got errors: {report['errors']}"
    assert len(report["errors"]) == 0

    # 2. Execute micro-training wet run
    cmd = [
        python_bin,
        "run_pipeline.py",
        exp_name,
        "total_timesteps=100",
        "eval_episodes=1",
        "no_plot=true",
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"Online RL run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "[SB3 Engine Complete]" in res.stdout or "total_timesteps" in res.stdout


@pytest.mark.skipif(
    os.environ.get("THETA_RUN_WET_TESTS") != "1",
    reason="wet training tests require separately installed component plugins; opt in with THETA_RUN_WET_TESTS=1",
)
def test_wet_run_supervised_tabular_classification(python_bin):
    """Wet test: Supervised paradigm with Lightning + MLPClassifier on tabular dataset runs for real."""
    exp_name = "tabular_classification/quick_test"
    
    # 1. Pre-flight check must be valid
    report = compose_report(exp_name)
    assert report["valid"] is True, f"Expected valid config, got errors: {report['errors']}"

    # 2. Execute micro-training wet run (2 epochs)
    cmd = [
        python_bin,
        "run_pipeline.py",
        exp_name,
        "++epochs=2",
        "++model.epochs=2",
        "no_plot=true",
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"Supervised run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "[Training Complete]" in res.stdout or "Trainer.fit" in res.stdout


@pytest.mark.skipif(
    os.environ.get("THETA_RUN_WET_TESTS") != "1",
    reason="wet training tests require separately installed component plugins; opt in with THETA_RUN_WET_TESTS=1",
)
def test_wet_run_unsupervised_representation(python_bin):
    """Wet test: Unsupervised paradigm with Lightning + Autoencoder on tabular dataset runs for real."""
    exp_name = "representation/quick_test"
    
    # 1. Pre-flight check must be valid
    report = compose_report(exp_name)
    assert report["valid"] is True, f"Expected valid config, got errors: {report['errors']}"

    # 2. Execute micro-training wet run (2 epochs)
    cmd = [
        python_bin,
        "run_pipeline.py",
        exp_name,
        "++epochs=2",
        "++model.epochs=2",
        "no_plot=true",
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"Unsupervised run failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "[Training Complete]" in res.stdout or "Trainer.fit" in res.stdout


def test_preflight_blocks_online_rl_without_environment():
    """A component-free config with no live environment reports the missing requirement."""
    config = {
        "paradigm": "online_rl",
        "env": None,
        "methods": {"test": {"agent": "ppo", "model": "dnn"}},
    }
    missing = find_missing_components(config)
    assert any("environment: (none configured)" in item for item in missing)


def test_preflight_blocks_supervised_without_dataset():
    """A component-free supervised config reports a missing dataset path."""
    config = {
        "paradigm": "supervised",
        "env": {"dataset_name": "nonexistent_data.npz"},
        "methods": {"test": {"model": "mlp_classifier"}},
    }
    missing = find_missing_components(config)
    assert any("dataset" in item.lower() for item in missing)


def test_preflight_blocks_uninstalled_components():
    """A component-free config reports an unavailable agent by its declared name."""
    config = {
        "paradigm": "online_rl",
        "env": {"name": "cartpole"},
        "methods": {"test": {"agent": "nonexistent_agent_999", "model": "dnn"}},
    }
    missing = find_missing_components(config)
    assert any("agent: nonexistent_agent_999" in item for item in missing)
