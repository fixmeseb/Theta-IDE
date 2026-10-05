"""Shared helpers that replaced duplicated code (folder layout, dataset resolution, experiment_id
inference, checkpoint inference, competency tiers, torch helpers)."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.app.pipeline import runtime
from src.app.pipeline.config import infer_experiment_id
from src.app.pipeline.datasets import resolve_method_dataset

REPO = Path(__file__).resolve().parents[1]


# ── Folder layout (src/app/pipeline/runtime.py) ──────────────────────────────

def test_folder_layout_matches_the_previous_literal_paths():
    assert (runtime.CHECKPOINTS_DIR, runtime.LOGS_DIR, runtime.PLOTS_DIR) == (
        "results/checkpoints", "results/logs", "results/plots")
    assert (runtime.TENSORBOARD_DIR, runtime.JOBS_DIR, runtime.OPTUNA_DIR) == (
        "results/tensorboard", "results/jobs", "results/optuna")
    assert (runtime.DATASETS_DIR, runtime.CONFIG_DIR, runtime.RULES_DIR, runtime.ENVS_DIR) == (
        "in/datasets", "in/config", "in/rules", "in/envs")
    assert runtime.PROJECT_ROOT == REPO


def test_api_uses_the_shared_layout():
    from src.app.api import app as api

    assert api.PROJECT_ROOT == runtime.PROJECT_ROOT
    assert api.JOBS_DIR == Path("results/jobs") and api.TENSORBOARD_DIR == Path("results/tensorboard")


# ── Dataset resolution shared by the local and Slurm runners ─────────────────

def test_both_runners_use_the_same_dataset_resolver():
    from src.app.pipeline import local_runner, slurm_runner

    assert local_runner.resolve_method_dataset is resolve_method_dataset
    assert slurm_runner.resolve_method_dataset is resolve_method_dataset
    assert not hasattr(local_runner, "_resolve_dataset_for_method")
    assert not hasattr(slurm_runner, "_resolve_dataset_for_method")


def test_resolve_method_dataset_explicit_path(tmp_path):
    data = tmp_path / "data.npz"
    data.write_bytes(b"")
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e"})
    assert resolve_method_dataset("m", {"dataset_path": str(data)}, cfg) == data


def test_resolve_method_dataset_env_then_folder_then_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "env": {"name": "cartpole", "dataset_name": "ds.npz"}})
    with mock.patch("src.app.pipeline.datasets.resolve_dataset_path", return_value=Path("found.npz")) as resolve:
        assert resolve_method_dataset("m", {}, cfg) == Path("found.npz")
    assert resolve.call_args.kwargs["dataset_id"] == "ds" and resolve.call_args.kwargs["group"] == "cartpole"
    with mock.patch("src.app.pipeline.datasets.resolve_dataset_path", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match="Cannot resolve dataset for method 'm'"):
            resolve_method_dataset("m", {}, cfg)
        (tmp_path / "in" / "datasets" / "g" / "e").mkdir(parents=True)
        assert resolve_method_dataset("m", {}, cfg) == Path("in/datasets/g/e")


# ── experiment_id inference shared by train.py and the early-prediction scripts ──

def test_infer_experiment_id():
    explicit = OmegaConf.create({"experiment_id": "mine"})
    infer_experiment_id(explicit)
    assert explicit.experiment_id == "mine"

    outside_hydra = OmegaConf.create({"experiment_id": "default_exp"})
    infer_experiment_id(outside_hydra)
    assert outside_hydra.experiment_id == "default_exp"

    hydra_cfg = SimpleNamespace(overrides=SimpleNamespace(task=["seed=1", "+experiment=sweeps/cartpole_ppo.yaml"]))
    with mock.patch("hydra.core.hydra_config.HydraConfig.initialized", return_value=True), \
            mock.patch("hydra.core.hydra_config.HydraConfig.get", return_value=hydra_cfg):
        cfg = OmegaConf.create({"experiment_id": "default_exp"})
        infer_experiment_id(cfg)
    assert cfg.experiment_id == "cartpole_ppo"


def test_entry_points_share_the_helpers():
    from src.usr.eval.early_prediction import model, tune_optuna

    assert tune_optuna.Dict2Obj is model.Dict2Obj
    for path in ("src/app/train.py", "src/usr/eval/early_prediction/model.py",
                 "src/usr/eval/early_prediction/tune_optuna.py"):
        text = (REPO / path).read_text(encoding="utf-8")
        assert "infer_experiment_id(cfg)" in text and "overrides.task" not in text


# ── Checkpoint inference shared by PyreneesEvaluator and ClinicalAlignmentPlotter ──

def test_get_probs_and_actions_uses_agent_api():
    import torch

    from src.usr.eval.checkpoint_inference import get_probs_and_actions

    probs = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
    agent = SimpleNamespace(get_action_probs=lambda obs: probs)
    got_probs, actions = get_probs_and_actions(agent, torch.zeros(2, 3))
    assert torch.equal(got_probs, probs) and actions.tolist() == [1, 0]


def _fake_agent_modules(load):
    """Stand-ins for the CEW/CQL/IQL agent modules (CEW needs `kneed`, which test environments may lack)."""
    cls = type("FakeAgent", (), {"load_from_checkpoint": staticmethod(load)})
    return {f"src.usr.methods.{name}_agent": SimpleNamespace(**{f"{name.upper()}Agent": cls})
            for name in ("cew", "cql", "iql")}


def test_load_agent_reports_failure_with_prefix(capsys):
    from src.usr.eval import checkpoint_inference

    def fail(*args, **kwargs):
        raise RuntimeError("bad")

    with mock.patch.dict(sys.modules, _fake_agent_modules(fail)):
        assert checkpoint_inference.load_agent("x.ckpt", "cpu", "MyCaller") is None
    assert "[MyCaller] Checkpoint load error for x.ckpt" in capsys.readouterr().out


def test_load_agent_moves_to_device_and_evaluates():
    from src.usr.eval import checkpoint_inference

    agent = mock.Mock()
    with mock.patch.dict(sys.modules, _fake_agent_modules(lambda *a, **k: agent)):
        assert checkpoint_inference.load_agent("x.ckpt", "cpu") is agent
    agent.to.assert_called_with("cpu")
    agent.eval.assert_called()


def test_both_callers_delegate_to_the_shared_functions():
    from src.usr.eval.pyrenees_evaluator import PyreneesEvaluator

    with mock.patch("src.usr.eval.pyrenees_evaluator.load_agent", return_value="agent") as load:
        evaluator = PyreneesEvaluator(device="cpu")
        assert evaluator._load_agent("p.ckpt") == "agent"
    load.assert_called_once_with("p.ckpt", "cpu", "PyreneesEvaluator")
    text = (REPO / "plot/clinical_alignment.py").read_text(encoding="utf-8")
    assert "return load_agent(path, dev," in text and "load_from_checkpoint" not in text


# ── Competency tiers shared by the Pyrenees scripts ──────────────────────────

def test_fit_tiered_gaussians_tiers_and_weights():
    from src.usr.eval.competency_tiers import fit_tiered_gaussians, sample_rows

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    model = fit_tiered_gaussians(X, X[:, 0], p_low=35, p_high=92)
    assert np.bincount(model["labels"]).tolist() == [350, 570, 80]
    assert np.allclose(model["weights"], [0.35, 0.57, 0.08])
    assert model["means"][0, 0] < model["means"][1, 0] < model["means"][2, 0]
    assert sample_rows(X, [1, 2], sample_size=10).shape == (10, 2)


def test_fit_tiered_gaussians_empty_tier_is_safe():
    from src.usr.eval.competency_tiers import fit_tiered_gaussians

    X = np.ones((50, 3))
    model = fit_tiered_gaussians(X, np.zeros(50))  # every score equal: tiers 1 and 2 are empty
    assert np.allclose(model["weights"], [1.0, 1 / 3, 1 / 3])
    assert np.array_equal(model["covariances"][1], np.eye(3)) and np.isfinite(model["log_dets"]).all()


# ── softor: one implementation, each package keeps its default ───────────────

def test_softor_defaults_preserved():
    import torch

    import importlib.util

    sys.path[:0] = [str(REPO / "src/usr/models"), str(REPO / "src/usr/environments")]
    import nsfr.utils.torch as nsfr_utils

    def load(path, name):  # the module file only: the neumann package itself needs torch_geometric
        spec = importlib.util.spec_from_file_location(name, REPO / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    nudge_utils = load("src/usr/environments/nudge/torch_utils.py", "nudge_torch_utils_under_test")
    neumann_utils = load("src/usr/models/neumann/torch_utils.py", "neumann_torch_utils_under_test")

    x = torch.rand(4, 5)
    assert torch.equal(nsfr_utils.softor(x), nsfr_utils.softor(x, gamma=0.01))
    for module in (nudge_utils, neumann_utils):
        assert torch.equal(module.softor(x), nsfr_utils.softor(x, gamma=0.015))
        assert module.logsumexp is nsfr_utils.logsumexp and module.weight_sum is nsfr_utils.weight_sum


def test_unused_neural_agent_copy_removed():
    assert not (REPO / "src/usr/environments/nudge/agents/stable_neural_agent.py").exists()
