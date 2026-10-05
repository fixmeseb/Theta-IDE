import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for p in [
    PROJECT_ROOT,
    SRC_DIR,
    os.path.join(SRC_DIR, "app"),
    os.path.join(SRC_DIR, "usr"),
    os.path.join(SRC_DIR, "usr", "models"),
    os.path.join(SRC_DIR, "usr", "environments"),
    os.path.join(SRC_DIR, "usr", "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
import time

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    torch.serialization.add_safe_globals(
        [
            omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig,
            omegaconf.base.Container,
            omegaconf.nodes.UntypedNode,
        ]
    )
except (AttributeError, TypeError) as e:
    logger.debug("PyTorch safe globals registration skipped: %s", e)
except Exception as e:
    logger.debug("Unexpected error registering safe globals: %s", e)

from src.app.core.lightning_builder import build_trainer, finalize_training
from src.app.data.rl_data_module import RLDataModule
from src.app.pipeline.config import infer_experiment_id
from src.usr.methods.registry import auto_discover, get_agent_class


@hydra.main(version_base=None, config_path="../../in/config", config_name="config")
def main(cfg: DictConfig):
    infer_experiment_id(cfg)

    print(OmegaConf.to_yaml(cfg))

    # === OS ENVIRON BRIDGE PATTERN ===
    # We copy certain Hydra config values into os.environ to pass them down to nested components
    # (like gym environments or legacy hooks) that cannot easily receive the `cfg` object directly.
    # While some components have been refactored to read from `cfg`, others still rely on this.
    if "env" in cfg and "reward_type" in cfg.env:
        os.environ["MIMIC_REWARD_TYPE"] = str(cfg.env.reward_type)

    if cfg.get("paradigm") == "supervised":
        from src.usr.eval.early_prediction.data_module import EPSepsisDataModule
        from src.usr.eval.early_prediction.lightning_module import EPSepsisLightningModule

        datamodule = EPSepsisDataModule(cfg)
        input_dim = getattr(datamodule, "input_dim", 64)

        model_cfg = cfg.get("model", {})
        arch_name = str(model_cfg.get("architecture", model_cfg.get("name", "lstm"))).lower()
        lr = float(model_cfg.get("lr", cfg.get("lr", 1e-3)))

        kwargs = {}
        for k in (
            "hidden_dim",
            "num_layers",
            "dropout",
            "use_dual_pooling",
            "use_tcn_conv",
            "bidirectional",
            "d_model",
            "nhead",
            "dim_feedforward",
            "pos_type",
            "max_len",
            "use_cls_token",
            "use_focal_loss",
            "pos_weight",
            "weight_decay",
        ):
            if hasattr(model_cfg, "get") and model_cfg.get(k) is not None:
                kwargs[k] = model_cfg.get(k)
            elif hasattr(cfg, "get") and cfg.get(k) is not None:
                kwargs[k] = cfg.get(k)

        print(f"Supervised Paradigm: constructing {arch_name.upper()} model (input_dim={input_dim}, lr={lr})")
        model = EPSepsisLightningModule(architecture_name=arch_name, input_dim=input_dim, lr=lr, **kwargs)
    else:
        auto_discover()
        agent_cfg = cfg.agent
        base_algo_name = agent_cfg.get("algorithm", agent_cfg.get("name", None))
        print(f"Extracted algorithm name: {base_algo_name}")

        if not base_algo_name:
            raise ValueError("Could not extract algorithm name from config.")

        datamodule = RLDataModule(cfg)
        AgentClass = get_agent_class(base_algo_name)
        print(f"Resolved agent class: {AgentClass.__name__}")
        model = AgentClass(cfg)

    trainer, ckpt_dir, ckpt_path = build_trainer(cfg, model)

    start_time = time.time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n[Training Complete] Total execution time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    metric = finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time)
    return metric


if __name__ == "__main__":
    main()
