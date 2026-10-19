import pytest
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

from src.app.core.lightning_builder import build_trainer


def test_default_offline_checkpointing(tmp_path):
    cfg = OmegaConf.create({
        "paradigm": "offline_rl",
        "output_dir": str(tmp_path),
        "env": {"name": "cartpole", "monitor_metric": "val/loss"},
        "epochs": 1,
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
        "checkpoint": None,  # omitted / default
    })
    # Remove explicit checkpoint key to simulate omission
    del cfg.checkpoint

    trainer, _, _ = build_trainer(cfg)
    checkpoints = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    assert len(checkpoints) >= 1
    best_ckpt = next(c for c in checkpoints if c.filename == "best_model")
    assert best_ckpt.monitor == "val/loss"
    assert best_ckpt.mode == "min"
    assert best_ckpt.save_top_k == 1


def test_declarative_checkpoint_custom_metric(tmp_path):
    cfg = OmegaConf.create({
        "paradigm": "supervised",
        "output_dir": str(tmp_path),
        "env": {"name": "tabular"},
        "epochs": 1,
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
        "checkpoint": {
            "monitor": "val/auprc",
            "mode": "max",
            "save_top_k": 3,
            "filename": "best_auprc",
        },
    })

    trainer, _, _ = build_trainer(cfg)
    checkpoints = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    best_ckpt = next(c for c in checkpoints if c.filename == "best_auprc")
    assert best_ckpt.monitor == "val/auprc"
    assert best_ckpt.mode == "max"
    assert best_ckpt.save_top_k == 3


def test_declarative_checkpoint_disabled(tmp_path):
    cfg = OmegaConf.create({
        "paradigm": "offline_rl",
        "output_dir": str(tmp_path),
        "env": {"name": "cartpole"},
        "epochs": 1,
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
        "checkpoint": False,
    })

    trainer, _, _ = build_trainer(cfg)
    best_ckpts = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint) and c.filename == "best_model"]
    assert len(best_ckpts) == 0
