import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_method_name(cfg) -> str:
    """Resolve method/agent/model identifier safely from config across all paradigms."""
    if hasattr(cfg, "agent") and cfg.agent is not None:
        name = cfg.agent.get("name", None) if hasattr(cfg.agent, "get") else getattr(cfg.agent, "name", None)
        if name:
            return str(name)
    if hasattr(cfg, "model") and cfg.model is not None:
        name = cfg.model.get("name", None) if hasattr(cfg.model, "get") else getattr(cfg.model, "name", None)
        if name:
            return str(name)
    return str(cfg.get("experiment_id", "default_method"))


class SaveInitialCheckpointCallback(Callback):
    """Callback to save an initial, untrained model checkpoint before training starts."""

    def __init__(self, ckpt_dir, cfg):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        init_ckpt = os.path.join(self.ckpt_dir, "best_model.ckpt")
        trainer.save_checkpoint(init_ckpt)
        from hydra.utils import get_original_cwd

        try:
            base_root = get_original_cwd()
        except (ValueError, AttributeError) as e:
            logger.debug("Hydra original cwd unavailable (%s), using os.getcwd()", e)
            base_root = os.getcwd()
        except Exception as e:
            logger.debug("Unexpected error getting Hydra original cwd (%s), using os.getcwd()", e)
            base_root = os.getcwd()
        parent_ckpt_root = os.path.join(base_root, "results/checkpoints", self.cfg.group, self.cfg.experiment_id)
        os.makedirs(parent_ckpt_root, exist_ok=True)
        method_name = get_method_name(self.cfg)
        named_ckpt = os.path.join(parent_ckpt_root, f"{method_name}.ckpt")
        if os.path.exists(init_ckpt):
            import shutil

            try:
                shutil.copy2(init_ckpt, named_ckpt)
            except Exception as e:
                print(f"[Init Checkpoint] Warning: could not copy to {named_ckpt}: {e}")

        init_epoch = os.path.join(self.ckpt_dir, "epoch_000.ckpt")
        trainer.save_checkpoint(init_epoch)
        print(f"[Init Checkpoint] Saved initial model checkpoint to: {init_ckpt}")


def print_hardware_diagnostics():
    """Print GPU hardware info and diagnostics before training."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_idx)
        total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        print("\n" + "=" * 50)
        print("      GPU HARDWARE DIAGNOSTICS")
        print("=" * 50)
        print(f"  Device:          {gpu_name} (ID: {device_idx})")
        print(f"  Total VRAM:      {total_vram_gb:.2f} GB")
        print(f"  CUDA Version:    {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print("=" * 50 + "\n")


def _agent_or_global(cfg, key, default):
    """A schedule setting from the agent config, else the experiment's own value.

    Reading only the agent silently discarded a value the experiment did set: an
    agent-less paradigm (supervised, unsupervised) still composes the default
    agent group, and that group does not declare these keys, so every run came
    out with max_epochs=1 no matter what the config asked for.
    """
    agent = getattr(cfg, "agent", None)
    if agent is None and hasattr(cfg, "get"):
        agent = cfg.get("agent", None)  # plain dicts hold it as a key, not an attribute
    if agent is not None and hasattr(agent, "get"):
        value = agent.get(key, None)
        if value is not None:
            return value
    return cfg.get(key, default) if hasattr(cfg, "get") else default


def infer_dynamic_timesteps(cfg):
    """Dynamically infer total timesteps based on offline dataset size."""
    if cfg.get("paradigm") == "offline_rl":
        ds_path = cfg.get("dataset_path", None)
        if ds_path and os.path.exists(ds_path):
            try:
                from src.app.dataset_utils import DatasetReader

                temp_reader = DatasetReader(ds_path)
                ds_len = len(temp_reader)
                if ds_len > 0:
                    current_tt = cfg.get("total_timesteps", None)
                    if current_tt is None or str(current_tt).lower() in ("auto", "-1", "none") or current_tt != ds_len:
                        print(f"\n[Dynamic Config] Detected offline dataset with {ds_len} transitions at '{ds_path}'.")
                        print(f"[Dynamic Config] Setting total_timesteps = {ds_len}.\n")
                        OmegaConf.set_struct(cfg, False)
                        cfg.total_timesteps = ds_len
                        OmegaConf.set_struct(cfg, True)
            except Exception as e:
                print(f"[Dynamic Config Warning] Failed to dynamically inspect dataset size: {e}")


def setup_loggers(cfg, base_root):
    method_name = get_method_name(cfg)
    group_name = str(cfg.get("group", "default_group") if hasattr(cfg, "get") else getattr(cfg, "group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp") if hasattr(cfg, "get") else getattr(cfg, "experiment_id", "default_exp"))
    log_dir = os.path.join(base_root, "results/logs", group_name, exp_id)
    loggers = [CSVLogger(log_dir, name=method_name)]

    use_tb = False
    if "tensorboard" in cfg:
        use_tb = bool(cfg.tensorboard)
    elif "agent" in cfg and hasattr(cfg.agent, "get") and cfg.agent.get("tensorboard", False):
        use_tb = bool(cfg.agent.tensorboard)

    if use_tb:
        tb_dir = os.path.join(base_root, "results/tensorboard", group_name, exp_id)
        tb_logger = TensorBoardLogger(tb_dir, name=method_name, default_hp_metric=False)
        _ = tb_logger.experiment
        loggers.append(tb_logger)
    telemetry_logger = _telemetry_logger(cfg, loggers[0], method_name)
    if telemetry_logger is not None:
        loggers.append(telemetry_logger)
    return loggers


def _telemetry_logger(cfg, csv_logger, method_name):
    """A ThetaLogger writing the run's other telemetry sinks (events.jsonl by default) into the CSVLogger's
    version folder. CSVLogger itself stays the metrics.csv writer, so the "csv" sink is left out here."""
    from src.app.telemetry import settings_from_config
    from src.app.telemetry.lightning import ThetaLogger

    sinks, system_interval = settings_from_config(cfg)
    sinks = [s for s in sinks if s != "csv"]
    if not sinks:
        return None
    try:
        config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # an interpolation that cannot resolve outside Hydra: keep it unresolved
        config = OmegaConf.to_container(cfg, resolve=False)
    meta = {
        "group": str(cfg.get("group", "")),
        "experiment_id": str(cfg.get("experiment_id", "")),
        "agent": method_name,
    }
    return ThetaLogger(lambda: csv_logger.log_dir, sinks, config, meta, system_interval)


def build_trainer(cfg, model=None):
    print_hardware_diagnostics()
    infer_dynamic_timesteps(cfg)

    default_seed = cfg.get("seed", 42) if hasattr(cfg, "get") else getattr(cfg, "seed", 42)
    agent_seed = (
        cfg.agent.get("seed", default_seed)
        if hasattr(cfg, "agent") and cfg.agent is not None and hasattr(cfg.agent, "get")
        else default_seed
    )
    if isinstance(agent_seed, DictConfig):
        agent_seed = agent_seed.get("seed", default_seed)
    L.seed_everything(agent_seed)

    from hydra.utils import get_original_cwd

    try:
        base_root = get_original_cwd()
    except (ValueError, AttributeError) as e:
        logger.debug("Hydra original cwd unavailable (%s), using os.getcwd()", e)
        base_root = os.getcwd()
    except Exception as e:
        logger.debug("Unexpected error getting Hydra original cwd (%s), using os.getcwd()", e)
        base_root = os.getcwd()

    loggers = setup_loggers(cfg, base_root)

    from hydra.core.hydra_config import HydraConfig

    trial_id = "0"
    if HydraConfig.initialized():
        try:
            trial_id = str(HydraConfig.get().job.num)
        except Exception as e:
            logger.debug("Could not inspect Hydra job num: %s", e)
    method_name = get_method_name(cfg)
    group_name = str(cfg.get("group", "default_group") if hasattr(cfg, "get") else getattr(cfg, "group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp") if hasattr(cfg, "get") else getattr(cfg, "experiment_id", "default_exp"))
    ckpt_dir = os.path.join(base_root, "results/checkpoints", group_name, exp_id, method_name, trial_id)

    def graceful_shutdown(signum, frame):
        print(f"\n[SIGTERM] Received termination signal {signum}. Attempting graceful shutdown...")
        # Since trainer is constructed here, we need a global reference or just sys.exit(0)
        # We will handle it by just capturing SIGTERM for basic shutdown
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, graceful_shutdown)
        signal.signal(signal.SIGUSR1, graceful_shutdown)
    except (ValueError, AttributeError, OSError) as e:
        logger.debug("Could not register shutdown signal handlers: %s", e)
    except Exception as e:
        logger.debug("Unexpected error registering shutdown signal handlers: %s", e)

    is_online = cfg.get("paradigm") == "online_rl"
    callbacks = [SaveInitialCheckpointCallback(ckpt_dir, cfg)]

    has_explicit_ckpt = "checkpoint" in cfg and cfg.get("checkpoint") is not None
    ckpt_cfg = cfg.get("checkpoint") if has_explicit_ckpt else None
    enable_checkpoint = True
    if has_explicit_ckpt and (ckpt_cfg is False or str(ckpt_cfg).lower() in ("none", "false", "disabled")):
        enable_checkpoint = False

    if enable_checkpoint:
        default_monitor = "eval/reward" if is_online else cfg.env.get("monitor_metric", "val/loss")
        default_mode = "max" if is_online else ("max" if any(k in str(default_monitor).lower() for k in ["f1", "reward", "auc", "acc", "prc"]) else "min")
        default_save_top_k = 1
        default_save_last = not is_online
        default_filename = "best_model"

        if hasattr(ckpt_cfg, "get"):
            monitor_metric = ckpt_cfg.get("monitor", default_monitor)
            monitor_mode = ckpt_cfg.get("mode")
            if not monitor_mode:
                monitor_mode = "max" if any(k in str(monitor_metric).lower() for k in ["f1", "reward", "auc", "acc", "prc"]) else "min"
            save_top_k = int(ckpt_cfg.get("save_top_k", default_save_top_k))
            save_last = bool(ckpt_cfg.get("save_last", default_save_last))
            filename = str(ckpt_cfg.get("filename", default_filename))
        else:
            monitor_metric = default_monitor
            monitor_mode = default_mode
            save_top_k = default_save_top_k
            save_last = default_save_last
            filename = default_filename

        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=filename,
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=save_top_k,
                save_last=save_last,
                enable_version_counter=False,
            )
        )

        if not is_online:
            eval_every_n_epochs = (
                ckpt_cfg.get("every_n_epochs") if hasattr(ckpt_cfg, "get") and "every_n_epochs" in ckpt_cfg
                else _agent_or_global(cfg, "eval_every_n_epochs", 1)
            )
            if eval_every_n_epochs:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=ckpt_dir,
                        filename="epoch_{epoch:03d}",
                        every_n_epochs=eval_every_n_epochs,
                        save_top_k=-1,
                        save_on_train_epoch_end=False,
                    )
                )

    if is_online:
        from src.app.core.callbacks import EnvironmentEvaluatorCallback, PeriodicCheckpointCallback
        from src.app.core.checkpoints import checkpoint_interval

        callbacks.append(EnvironmentEvaluatorCallback(cfg))

        interval = checkpoint_interval(cfg)
        if interval:
            callbacks.append(PeriodicCheckpointCallback(ckpt_dir, interval, int(cfg.get("total_timesteps") or 0)))

    if cfg.get("paradigm") == "online_rl":
        num_envs = getattr(model, "num_envs", cfg.env.num_envs) if model else cfg.env.num_envs
        num_steps = getattr(model, "num_steps", cfg.env.num_steps) if model else getattr(cfg.env, "num_steps", 256)
        batch_size = num_envs * num_steps

        import math

        max_epochs = math.ceil(cfg.total_timesteps / batch_size) if batch_size > 0 else 1
        if max_epochs == 0:
            max_epochs = 1

        limit_val_batches = 0
        check_val_every_n_epoch = 1000000
    else:
        max_epochs = _agent_or_global(cfg, "epochs", 50)
        eval_every_n_epochs = _agent_or_global(cfg, "eval_every_n_epochs", 1)
        limit_val_batches = 1.0
        check_val_every_n_epoch = eval_every_n_epochs if eval_every_n_epochs else 1

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": limit_val_batches,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False if not sys.stdout.isatty() else True,
    }
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        precision_setting = cfg.get("precision", "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed")
        trainer_kwargs["precision"] = precision_setting

    if "trainer" in cfg and cfg.trainer is not None:
        trainer_overrides = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer_kwargs.update(trainer_overrides)

    from src.app.telemetry.lightning import TelemetryCallback, ThetaLogger

    callbacks.extend(TelemetryCallback(lg) for lg in loggers if isinstance(lg, ThetaLogger))
    trainer = L.Trainer(**trainer_kwargs, logger=loggers, callbacks=callbacks)

    ckpt_path = None
    if cfg.get("recover", False):
        potential_ckpt = os.path.join(ckpt_dir, "best_model.ckpt")
        if os.path.exists(potential_ckpt):
            print(f"Recovering from checkpoint: {potential_ckpt}")
            ckpt_path = potential_ckpt

    return trainer, ckpt_dir, ckpt_path


def finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time):
    gpu_stats = {}
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        peak_alloc_gb = torch.cuda.max_memory_allocated(device_idx) / (1024**3)
        peak_res_gb = torch.cuda.max_memory_reserved(device_idx) / (1024**3)
        mem_eff_pct = (peak_alloc_gb / total_vram_gb) * 100.0 if total_vram_gb > 0 else 0.0

        gpu_stats = {
            "gpu_device": torch.cuda.get_device_name(device_idx),
            "gpu_total_vram_gb": round(total_vram_gb, 2),
            "gpu_peak_alloc_gb": round(peak_alloc_gb, 3),
            "gpu_peak_reserved_gb": round(peak_res_gb, 3),
            "gpu_mem_efficiency_pct": round(mem_eff_pct, 2),
        }

        print("\n" + "=" * 50)
        print("      GPU MEMORY & RESOURCE FOOTPRINT")
        print("=" * 50)
        print(f"  GPU Device:        {gpu_stats['gpu_device']}")
        print(f"  Total VRAM:        {gpu_stats['gpu_total_vram_gb']:.2f} GB")
        print(f"  Peak Allocated:    {gpu_stats['gpu_peak_alloc_gb']:.3f} GB")
        print(f"  Peak Reserved:     {gpu_stats['gpu_peak_reserved_gb']:.3f} GB")
        print(f"  VRAM Footprint %:  {gpu_stats['gpu_mem_efficiency_pct']:.2f}%")
        print("=" * 50 + "\n")

    active_loggers = (
        trainer.loggers
        if (hasattr(trainer, "loggers") and trainer.loggers)
        else ([trainer.logger] if trainer.logger else [])
    )
    for lg in active_loggers:
        try:
            if hasattr(lg, "log_metrics"):
                metrics_to_log = {"training_time_seconds": training_time}
                if gpu_stats:
                    metrics_to_log.update(
                        {
                            "gpu/peak_alloc_gb": gpu_stats["gpu_peak_alloc_gb"],
                            "gpu/peak_reserved_gb": gpu_stats["gpu_peak_reserved_gb"],
                        }
                    )
                lg.log_metrics(metrics_to_log, step=trainer.global_step)
        except Exception as e:
            logger.warning("Failed to log metrics to %s: %s", lg, e)

    os.makedirs(ckpt_dir, exist_ok=True)

    import datetime

    from hydra.core.hydra_config import HydraConfig

    from src.app.core.metadata import collect_run_metadata, save_git_diff

    hydra_output_dir = None
    if HydraConfig.initialized():
        try:
            hydra_output_dir = HydraConfig.get().runtime.output_dir
        except Exception as e:
            logger.debug("Could not inspect Hydra runtime output_dir: %s", e)

    start_time_iso = (
        datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat() if start_time else None
    )
    end_time_iso = datetime.datetime.fromtimestamp(end_time, datetime.timezone.utc).isoformat() if end_time else None

    meta = collect_run_metadata(cfg)
    save_git_diff(ckpt_dir)

    method_name = get_method_name(cfg)
    runtime_info = {
        "agent": method_name,
        "experiment_id": str(cfg.experiment_id),
        "group": str(cfg.group),
        "training_time_seconds": round(training_time, 2),
        "start_time": start_time,
        "end_time": end_time,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
        "hydra_output_dir": hydra_output_dir,
        **meta,
        **gpu_stats,
    }

    def atomic_json_dump(obj, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def atomic_yaml_save(cfg_obj, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            OmegaConf.save(config=cfg_obj, f=f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    atomic_json_dump(runtime_info, os.path.join(ckpt_dir, "runtime.json"))

    try:
        atomic_yaml_save(cfg, os.path.join(ckpt_dir, "config.yaml"))
        exp_ckpt_root = os.path.join("results/checkpoints", cfg.group, cfg.experiment_id)
        os.makedirs(exp_ckpt_root, exist_ok=True)
        atomic_yaml_save(cfg, os.path.join(exp_ckpt_root, "config.yaml"))

        if hydra_output_dir:
            import shutil

            overrides_path = os.path.join(hydra_output_dir, ".hydra", "overrides.yaml")
            if os.path.exists(overrides_path):
                shutil.copy2(overrides_path, os.path.join(ckpt_dir, "overrides.yaml"))
    except Exception as e:
        print(f"Notice: Could not save checkpoint config.yaml or overrides.yaml: {e}")

    for lg in active_loggers:
        if hasattr(lg, "log_dir") and lg.log_dir:
            os.makedirs(lg.log_dir, exist_ok=True)
            atomic_json_dump(runtime_info, os.path.join(lg.log_dir, "runtime.json"))
            try:
                atomic_yaml_save(cfg, os.path.join(lg.log_dir, "config.yaml"))
                if hydra_output_dir:
                    overrides_path = os.path.join(hydra_output_dir, ".hydra", "overrides.yaml")
                    if os.path.exists(overrides_path):
                        import shutil

                        shutil.copy2(overrides_path, os.path.join(lg.log_dir, "overrides.yaml"))
            except (OSError, Exception) as e:
                logger.warning("Could not save config or overrides to log_dir %s: %s", lg.log_dir, e)
        if hasattr(lg, "save"):
            try:
                lg.save()
            except Exception as e:
                logger.warning("Failed to save/flush logger %s: %s", lg, e)

    exp_log_root = os.path.join("results/logs", cfg.group, cfg.experiment_id)
    os.makedirs(exp_log_root, exist_ok=True)
    try:
        atomic_yaml_save(cfg, os.path.join(exp_log_root, "config.yaml"))
    except Exception as e:
        print(f"Notice: Could not save logger config.yaml: {e}")

    final_ckpt_target = os.path.join(ckpt_dir, "best_model.ckpt")
    if not os.path.exists(final_ckpt_target):
        print(f"Saving final trained model checkpoint to: {final_ckpt_target}")
        trainer.save_checkpoint(final_ckpt_target)

    parent_ckpt_dir = str(Path(ckpt_dir).parent) if Path(ckpt_dir).name.isdigit() else str(Path(ckpt_dir))
    parent_ckpt_target = os.path.join(parent_ckpt_dir, "best_model.ckpt")
    if os.path.abspath(final_ckpt_target) != os.path.abspath(parent_ckpt_target) and os.path.exists(final_ckpt_target):
        try:
            import shutil

            os.makedirs(parent_ckpt_dir, exist_ok=True)
            shutil.copy2(final_ckpt_target, parent_ckpt_target)
            named_ckpt = os.path.join(parent_ckpt_dir, f"{method_name}.ckpt")
            shutil.copy2(final_ckpt_target, named_ckpt)
        except Exception as e:
            logger.debug("Could not copy single-trial checkpoint to agent root: %s", e)

    metric_name = cfg.env.get("monitor_metric", "eval/reward")
    if metric_name in trainer.callback_metrics:
        return trainer.callback_metrics[metric_name].item()
    return 0.0
