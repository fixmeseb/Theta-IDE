"""Tests for Early Prediction checkpoint loading (.pt and Lightning .ckpt)."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.usr.eval.early_prediction.eval_logic import (
    discover_ep_checkpoints,
    load_ep_model,
)
from src.usr.eval.early_prediction.model import SepsisLSTM
from src.usr.eval.reward_shaping import EPRewardShaper


def test_load_ep_model_pt_format(tmp_path):
    ckpt_file = tmp_path / "ep_model.pt"
    lstm = SepsisLSTM(input_dim=10, hidden_dim=16, num_layers=1)
    torch.save({
        "model_type": "lstm",
        "input_dim": 10,
        "hyperparams": {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0},
        "model_state_dict": lstm.state_dict(),
        "opt_thresh": 0.42,
    }, ckpt_file)

    model, m_type, in_dim, thresh = load_ep_model(ckpt_file, device=torch.device("cpu"))
    assert m_type == "lstm"
    assert in_dim == 10
    assert thresh == 0.42
    assert isinstance(model, SepsisLSTM)


def test_load_ep_model_lightning_ckpt_format(tmp_path):
    ckpt_file = tmp_path / "ep_lightning.ckpt"
    lstm = SepsisLSTM(input_dim=10, hidden_dim=16, num_layers=1)
    state_dict = {f"model.{k}": v for k, v in lstm.state_dict().items()}
    state_dict["loss_fn.pos_weight"] = torch.tensor([1.0])

    torch.save({
        "hyper_parameters": {
            "architecture_name": "lstm_no_v",
            "input_dim": 10,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "state_dict": state_dict,
    }, ckpt_file)

    model, m_type, in_dim, thresh = load_ep_model(ckpt_file, device=torch.device("cpu"))
    assert m_type == "lstm"
    assert in_dim == 10
    assert isinstance(model, SepsisLSTM)


def test_discover_ep_checkpoints_both_extensions(tmp_path):
    pt_file = tmp_path / "lstm_no_v_tau5_split0.pt"
    ckpt_file = tmp_path / "lstm_no_v_tau5_split1.ckpt"
    pt_file.touch()
    ckpt_file.touch()

    discovered = discover_ep_checkpoints(tmp_path)
    assert "lstm_no_v" in discovered
    assert 5 in discovered["lstm_no_v"]
    found_paths = [p.name for p in discovered["lstm_no_v"][5]]
    assert "lstm_no_v_tau5_split0.pt" in found_paths
    assert "lstm_no_v_tau5_split1.ckpt" in found_paths


def test_ep_reward_shaper_loads_both_ckpt_and_pt(tmp_path):
    lstm = SepsisLSTM(input_dim=10, hidden_dim=16, num_layers=1)

    pt_file = tmp_path / "lstm_split0.pt"
    torch.save({
        "model_type": "lstm",
        "input_dim": 10,
        "hyperparams": {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0},
        "model_state_dict": lstm.state_dict(),
    }, pt_file)

    ckpt_file = tmp_path / "lstm_split1.ckpt"
    torch.save({
        "hyper_parameters": {
            "architecture_name": "lstm_no_v",
            "input_dim": 10,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "state_dict": {f"model.{k}": v for k, v in lstm.state_dict().items()},
    }, ckpt_file)

    cfg = OmegaConf.create({
        "env": {
            "reward_shaping": {
                "ep_ckpt_dir": str(tmp_path),
                "ep_architecture": None,
                "ep_weight": 1.0,
                "gamma": 0.99,
            }
        }
    })
    shaper = EPRewardShaper(cfg, device="cpu")
    assert len(shaper._models) == 2
