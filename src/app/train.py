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
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    safe_types = [
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.Container,
    ]
    for node_name in ["AnyNode", "Node", "ValueNode", "UntypedNode"]:
        if hasattr(omegaconf.nodes, node_name):
            safe_types.append(getattr(omegaconf.nodes, node_name))
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_types)
except (AttributeError, TypeError) as e:
    logger.debug("PyTorch safe globals registration skipped: %s", e)
except Exception as e:
    logger.debug("Unexpected error registering safe globals: %s", e)

from src.app.core.lightning_builder import build_trainer, finalize_training
from src.app.data.rl_data_module import RLDataModule
from src.usr.methods.agent_registry import auto_discover, get_agent_class


@hydra.main(version_base=None, config_path="../../in/config", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("experiment_id", "default_exp") == "default_exp":
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                for override in HydraConfig.get().overrides.task:
                    if override.startswith("+experiment=") or override.startswith("experiment="):
                        exp_stem = Path(override.split("=")[-1]).stem
                        cfg.experiment_id = exp_stem
                        break
        except Exception as e:
            logger.debug("Could not infer experiment_id from Hydra task overrides: %s", e)

    print(OmegaConf.to_yaml(cfg))

    # === OS ENVIRON BRIDGE PATTERN ===
    # We copy certain Hydra config values into os.environ to pass them down to nested components
    # (like gym environments or legacy hooks) that cannot easily receive the `cfg` object directly.
    # While some components have been refactored to read from `cfg`, others still rely on this.
    if "env" in cfg and "reward_type" in cfg.env:
        os.environ["MIMIC_REWARD_TYPE"] = str(cfg.env.reward_type)

    paradigm_name = cfg.get("paradigm", "online_rl")
    from src.app.core.paradigm_loader import get_component, load_paradigm_definition

    paradigm_def = load_paradigm_definition(paradigm_name)

    # 1. Resolve Data Module from explicit config or paradigm definition
    dm_name = cfg.get("data_module") or (cfg.env.get("data_module") if hasattr(cfg, "env") else None)
    DataModuleCls = get_component(dm_name) if dm_name else paradigm_def.data_module_cls
    if DataModuleCls is None:
        DataModuleCls = RLDataModule

    datamodule = DataModuleCls(cfg)

    # 2. Build Model / Agent based on paradigm
    # Driven by whether the paradigm declares agents, not by its name, so an
    # agent-less paradigm such as unsupervised builds a model the same way.
    from src.app.pipeline.validation import paradigm_uses_agents

    if not paradigm_uses_agents(paradigm_name):
        input_dim = getattr(datamodule, "input_dim", 64)

        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else getattr(cfg, "model", {})
        arch_name = str(model_cfg.get("architecture", model_cfg.get("name", "lstm"))).lower()
        lr = float(model_cfg.get("lr", cfg.get("lr", 1e-3)))

        # Check if an explicit LightningModule component was registered for this architecture
        target_module_name = model_cfg.get("lightning_module") or model_cfg.get("module")
        SupervisedModelCls = None
        if target_module_name:
            try:
                SupervisedModelCls = get_component(target_module_name)
            except KeyError:
                SupervisedModelCls = None
        if SupervisedModelCls is None:
            from src.usr.eval.early_prediction.lightning_module import EPSepsisLightningModule

            SupervisedModelCls = EPSepsisLightningModule

        kwargs = {}
        if hasattr(model_cfg, "items"):
            reserved = {
                "architecture",
                "name",
                "lightning_module",
                "module",
                "lr",
                "type",
                "epochs_per_interval",
                "eval_interval_epochs",
            }
            for k, v in model_cfg.items():
                if k not in reserved and v is not None:
                    kwargs[k] = v

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
            if k not in kwargs and hasattr(cfg, "get") and cfg.get(k) is not None:
                kwargs[k] = cfg.get(k)

        print(
            f"Supervised Paradigm: constructing {SupervisedModelCls.__name__} ({arch_name.upper()}, input_dim={input_dim}, lr={lr})"
        )
        model = SupervisedModelCls(architecture_name=arch_name, input_dim=input_dim, lr=lr, **kwargs)
    else:
        auto_discover()
        agent_cfg = cfg.agent
        base_algo_name = agent_cfg.get("algorithm", agent_cfg.get("name", None))
        print(f"Extracted algorithm name: {base_algo_name}")

        if not base_algo_name:
            raise ValueError("Could not extract algorithm name from config.")

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
