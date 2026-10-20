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

from src.app.pipeline.validation import validate_experiment_config, find_missing_components
from src.app.pipeline.exceptions import ConfigurationError
from src.app.services.config_service import compose_report


@pytest.fixture(scope="session")
def python_bin() -> str:
    backend_py = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python3"
    if backend_py.exists():
        return str(backend_py)
    return sys.executable


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
    """Verify that online_rl experiment missing an environment is rejected before launch."""
    report = compose_report("thetaide/cartpole_ppo_reference", ["~env"])
    assert report["valid"] is False
    assert any("env" in str(e).lower() for e in report["errors"])


def test_preflight_blocks_supervised_without_dataset():
    """Verify that supervised experiment missing a dataset file is rejected before launch."""
    report = compose_report("tabular_classification/quick_test", ["++env.dataset_name=nonexistent_data.npz"])
    assert report["valid"] is False
    assert any("dataset" in str(e).lower() for e in report["errors"])


def test_preflight_blocks_uninstalled_components():
    """Verify that an experiment naming an uninstalled agent or model is rejected before launch."""
    report = compose_report("thetaide/cartpole_ppo_reference", ["methods.ppo.agent=nonexistent_agent_999"])
    assert report["valid"] is False
    assert any("agent" in str(e).lower() or "missing component" in str(e).lower() for e in report["errors"])
