"""Early prediction task execution module.

Handles early prediction tasks: deep learning sweeps, checkpoint evaluation,
and Optuna hyperparameter tuning using modular components:
- Domain data provider: EPSepsisDataModule (src.usr.eval.early_prediction.data_module)
- Generic training loop: SupervisedRunner (src.app.core.paradigm_impls.base.supervised)
- Generic evaluation: ClassificationEvalProtocol (src.app.core.paradigm_impls.base.supervised)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.app.core.paradigm_impls.base.supervised import ClassificationEvalProtocol, SupervisedRunner
from src.app.pipeline.datasets import fast_purge_dir, run_plotting
from src.app.pipeline.runtime import get_python_executable, get_shell_env_block, get_shell_python_cmd
from src.app.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.app.pipeline.task_registry import register_task
from src.usr.eval.early_prediction.data_module import EPSepsisDataModule
from src.usr.eval.early_prediction.lightning_module import EPSepsisLightningModule
from src.usr.eval.early_prediction.model import load_target_params
from src.app.pipeline import runtime

log = logging.getLogger(__name__)


@register_task("early_prediction")
def run_early_prediction_task_wrapper(cfg: Any, context: dict) -> None:
    run_early_prediction_task(
        cfg,
        context.get("is_interactive", context.get("local_val", True)),
        context.get("sanitized_extra_args", []),
        context.get("storage_url"),
        context.get("is_sweep", False),
        paradigm_def=context.get("paradigm_def"),
    )


def run_early_prediction_task(
    cfg: Any,
    local_val: bool,
    sanitized_extra_args: list[str],
    storage_url: str | None,
    is_sweep: bool,
    *,
    paradigm_def: Any = None,
) -> bool:
    """Handle standalone early_prediction tasks (sweeps, evals, tuning)."""
    task_name = cfg.get("task", "")
    if not task_name.startswith("early_prediction"):
        return False

    site_cfg = cfg.get("site", None)

    if task_name == "early_prediction_sweep":
        if not cfg.get("recover", False):
            clean_exp = Path(cfg.experiment_id).stem
            plot_dir = Path(runtime.PLOTS_DIR) / cfg.group / clean_exp
            fast_purge_dir(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)

        if local_val:
            print(f"Running Modular Supervised Sweep for {cfg.experiment_id}...")
            run_modular_early_prediction_sweep(cfg, paradigm_def=paradigm_def)
            if not cfg.get("no_plot", False):
                run_plotting(
                    cfg.experiment_id,
                    style=cfg.get("plot_style", None),
                    base_experiment=cfg.get("experiment_name", ""),
                    site_cfg=site_cfg,
                )
            sys.exit(0)
        else:
            slurm_dir = Path(f"{runtime.LOGS_DIR}/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)

            target_models = ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"]
            print(f"Submitting 4 parallel SLURM model jobs ({target_models}) for {cfg.experiment_id}...")

            job_ids = []
            for tm in target_models:
                slurm_script_path = slurm_dir / f"early_pred_sweep_{tm}.slurm"
                log_dir = Path(f"{runtime.LOGS_DIR}/slurm") / cfg.group / cfg.experiment_id

                from omegaconf import OmegaConf

                local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in local_cfg:
                    local_cfg.resources = {}
                local_cfg.resources.cores = 16

                header = generate_sbatch_header(
                    job_name=f"ep_{tm}_{cfg.experiment_id}", log_dir=log_dir, cfg=local_cfg
                )
                env_block = get_shell_env_block(site_cfg)
                python_cmd = get_shell_python_cmd(site_cfg)

                extra_args_str = (
                    " ".join([f'"{a}"' if " " in a else a for a in sanitized_extra_args])
                    if sanitized_extra_args
                    else ""
                )
                experiment_arg = cfg.get("experiment_name", f"{cfg.group}/{cfg.experiment_id}")

                script_content = f"""{header}
echo "=== Sepsis Early Prediction Sweep Execution Start ({tm}) ==="
echo "Node: $(hostname)"
date

{env_block}
mkdir -p results/plots/early_prediction
mkdir -p results/logs

{python_cmd} run_pipeline.py {experiment_arg} \
    ++early_prediction.target_model={tm} \
    {extra_args_str}

echo "=== Sepsis Early Prediction Sweep Execution End ({tm}) ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Model SLURM Job ({tm}): {slurm_script_path}")

                job_id = submit_sbatch(script_content)
                if job_id:
                    job_ids.append(job_id)

            # Dependent plotting SLURM job
            if not cfg.get("no_plot", False) and job_ids:
                plot_slurm_script = slurm_dir / "early_pred_sweep_plot.slurm"
                dep_str = ":".join(job_ids)
                plot_cmd_str = f"{get_shell_python_cmd(site_cfg)} plot/manager.py {cfg.group}/{cfg.experiment_id}"
                if cfg.get("plot_style", None):
                    plot_cmd_str += f" --style {cfg.get('plot_style', None)}"

                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                from omegaconf import OmegaConf

                plot_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in plot_cfg:
                    plot_cfg.resources = {}
                plot_cfg.resources.gpus = 0
                plot_cfg.resources.cores = 4

                header = generate_sbatch_header(
                    job_name=f"ep_plot_{cfg.experiment_id}",
                    log_dir=log_dir,
                    cfg=plot_cfg,
                    dependency=dep_str,
                    dependency_type="afterok",
                )

                env_block = get_shell_env_block(site_cfg)
                plot_script_content = f"""{header}
echo "=== Early Prediction Sweep Plotting Start ==="
echo "Node: $(hostname)"
date

{env_block}

echo "Running plotting script..."
{plot_cmd_str}

echo "=== Early Prediction Sweep Plotting End ==="
date
"""
                with open(plot_slurm_script, "w") as f:
                    f.write(plot_script_content)

                print(f"Submitting Plotting SLURM Job (Dependent on {dep_str}): {plot_slurm_script}")
                submit_sbatch(plot_script_content)

            sys.exit(0)

    elif task_name == "early_prediction_eval":
        print(f"\n=== Running Early Prediction Checkpoint Evaluation ({cfg.experiment_id}) ===")
        if local_val:
            cmd = [get_python_executable(site_cfg), "-u", "plot/manager.py", f"{cfg.group}/{cfg.experiment_id}"]
            res = subprocess.run(cmd)
            sys.exit(res.returncode)
        else:
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            slurm_script_path = slurm_dir / "early_pred_eval.slurm"

            from omegaconf import OmegaConf

            local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            if "resources" not in local_cfg:
                local_cfg.resources = {}
            local_cfg.resources.cores = 16

            header = generate_sbatch_header(job_name=f"eval_pred_{cfg.experiment_id}", log_dir=slurm_dir, cfg=local_cfg)
            env_block = get_shell_env_block(site_cfg)
            python_cmd = get_shell_python_cmd(site_cfg)

            script_content = f"""{header}
{env_block}

{python_cmd} -u plot/manager.py {cfg.group}/{cfg.experiment_id}
"""
            with open(slurm_script_path, "w") as f:
                f.write(script_content)
            submit_sbatch(script_content)
            sys.exit(0)

    elif task_name in ("early_prediction_tune", "early_prediction_optuna"):
        print(f"\n=== Running Early Prediction Optuna Hyperparameter Search ({cfg.experiment_id}) ===")
        ep_cfg = cfg.get("early_prediction", {})
        target_models = ep_cfg.get(
            "target_models", ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"]
        )
        if isinstance(target_models, str):
            target_models = [m.strip() for m in target_models.split(",")]

        slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
        slurm_dir.mkdir(parents=True, exist_ok=True)

        for m_target in target_models:
            print(f"\n--> Setting up Optuna Study for architecture target: [{m_target}]")
            out_dir = ep_cfg.get("output_dir", f"results/plots/early_prediction/{cfg.experiment_id}")
            stray_study_dir = Path(out_dir) / "optuna_study"
            if stray_study_dir.exists() and stray_study_dir.is_dir():
                shutil.rmtree(stray_study_dir)

            if local_val:
                cmd = [
                    get_python_executable(site_cfg),
                    "-u",
                    "src/usr/eval/early_prediction/tune_optuna.py",
                    f"+experiment={cfg.get('experiment_name', f'{cfg.group}/{cfg.experiment_id}')}",
                    f"++early_prediction.model_target={m_target}",
                ]
                if sanitized_extra_args:
                    cmd.extend(sanitized_extra_args)
                subprocess.run(cmd, check=True)
            else:
                slurm_script_path = slurm_dir / f"tune_pred_{m_target}.slurm"
                from omegaconf import OmegaConf

                local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in local_cfg:
                    local_cfg.resources = {}
                local_cfg.resources.cores = 16

                header = generate_sbatch_header(
                    job_name=f"tune_{m_target}_{cfg.experiment_id}", log_dir=slurm_dir, cfg=local_cfg
                )
                env_block = get_shell_env_block(site_cfg)
                python_cmd = get_shell_python_cmd(site_cfg)
                extra_args_str = (
                    " ".join([f'"{a}"' if " " in a else a for a in sanitized_extra_args])
                    if sanitized_extra_args
                    else ""
                )
                experiment_arg = cfg.get("experiment_name", f"{cfg.group}/{cfg.experiment_id}")

                script_content = f"""{header}
echo "=== Sepsis Early Prediction Optuna Search [{m_target}] Start ==="
echo "Node: $(hostname)"
date

{env_block}

mkdir -p {out_dir}
mkdir -p results/logs

{python_cmd} -u src/usr/eval/early_prediction/tune_optuna.py \
    +experiment={experiment_arg} \
    ++early_prediction.model_target={m_target} \
    {extra_args_str}

echo "=== Sepsis Early Prediction Optuna Search [{m_target}] End ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                submit_sbatch(script_content)
        sys.exit(0)

    return True


# ---------------------------------------------------------------------------
# Modular Early Prediction Sweep Implementation
# ---------------------------------------------------------------------------

def run_modular_early_prediction_sweep(cfg: Any, paradigm_def: Any = None) -> None:
    """Execute Sepsis early prediction CV training and multi-tau evaluation
    using domain data provider and reusable paradigm runner components.
    """
    ep_cfg = cfg.get("early_prediction", {}) or {}
    if hasattr(ep_cfg, "__iter__") and not isinstance(ep_cfg, dict):
        from omegaconf import OmegaConf

        ep_cfg = OmegaConf.to_container(ep_cfg, resolve=True)

    # 1. Instantiate reusable paradigm components
    runner: SupervisedRunner = (
        paradigm_def.runner_cls()
        if (paradigm_def and paradigm_def.runner_cls)
        else SupervisedRunner()
    )
    eval_protocol: ClassificationEvalProtocol = (
        paradigm_def.eval_protocol_cls()
        if (paradigm_def and paradigm_def.eval_protocol_cls)
        else ClassificationEvalProtocol()
    )

    # 2. Instantiate domain data module
    data_module = EPSepsisDataModule(cfg)

    # 3. Configure horizons and architectures
    tau_min = ep_cfg.get("tau_min", 1)
    tau_max = ep_cfg.get("tau_max", 33)
    tau_step = ep_cfg.get("tau_step", 4)
    tau_train = ep_cfg.get("tau_train", 12)
    n_splits = ep_cfg.get("n_splits", 20)
    epochs = ep_cfg.get("epochs", 20)
    batch_size = ep_cfg.get("batch_size", 64)
    lr = ep_cfg.get("lr", 1e-3)
    save_checkpoints = ep_cfg.get("save_checkpoints", True)
    use_tuned_params = ep_cfg.get("use_tuned_params", True)
    tune_dir = ep_cfg.get("tune_dir", "results/plots/early_prediction/tune_early_pred")
    output_dir = ep_cfg.get("output_dir", "results/plots/early_prediction")
    exp_id = cfg.get("experiment_id", "default_exp")
    ep_ckpt_root = ep_cfg.get("ep_ckpt_root", "results/checkpoints/early_prediction")

    target_model = ep_cfg.get("target_model", "all")
    if isinstance(target_model, str):
        target_model = target_model.lower()

    tau_list = list(range(tau_min, tau_max + 1, tau_step))
    log.info("Sweeping tau lead times: %s", tau_list)

    all_configs = [
        ("LSTM (no V)", "lstm", False, "lstm_no_v"),
        ("LSTM (with V)", "lstm", True, "lstm_with_v"),
        ("Transformer (no V)", "transformer", False, "transformer_no_v"),
        ("Transformer (with V)", "transformer", True, "transformer_with_v"),
    ]

    if target_model and target_model != "all":
        model_configs = [c for c in all_configs if c[3] == target_model]
        if not model_configs:
            model_configs = list(all_configs)
    else:
        model_configs = list(all_configs)

    results = {
        name: {
            "tau": [],
            "auc": [],
            "auc_sem": [],
            "auprc": [],
            "auprc_sem": [],
            "f1_opt": [],
            "f1_opt_sem": [],
            "f1_max": [],
            "f1_max_sem": [],
            "f1_05": [],
            "f1_05_sem": [],
        }
        for name, _, _, _ in model_configs
    }

    # 4. Train and evaluate each architecture across splits
    for m_cfg_name, m_type, use_v, m_key in model_configs:
        log.info("Executing Architecture: %s across %d CV splits", m_cfg_name, n_splits)

        tau_metrics: dict[int, dict[str, list[float]]] = {
            tau: {"auc": [], "auprc": [], "f1_opt": [], "f1_max": [], "f1_05": []}
            for tau in tau_list
        }

        for split_idx in range(n_splits):
            log.info("Split %d / %d for %s", split_idx + 1, n_splits, m_cfg_name)
            train_loader, x_train, y_train, input_dim = data_module.get_train_dataloader(
                split_idx=split_idx, use_v=use_v, batch_size=batch_size
            )

            params = load_target_params(tune_dir, m_cfg_name) if use_tuned_params else {}
            split_lr = params.get("lr", lr)
            split_epochs = params.get("epochs", epochs)

            # Build LightningModule
            module = EPSepsisLightningModule(
                architecture_name=m_type,
                input_dim=input_dim,
                lr=split_lr,
                hidden_dim=params.get("hidden_dim", 64),
                d_model=params.get("d_model", 64),
                nhead=params.get("nhead", 4),
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.1),
                weight_decay=params.get("weight_decay", 1e-4),
                use_focal_loss=params.get("use_focal_loss", False),
                use_tcn_conv=params.get("use_tcn_conv", False),
                bidirectional=params.get("bidirectional", False),
            )

            # Reusing standard generic SupervisedRunner!
            runner.train_model(
                model=module,
                train_loader=train_loader,
                val_loader=None,
                epochs=split_epochs,
            )

            # Determine optimal decision threshold on training set
            train_eval = eval_protocol.evaluate(module, train_loader)
            opt_thresh = train_eval.get("opt_thresh", 0.5)

            # Save split checkpoint if enabled
            if save_checkpoints:
                ckpt_dir = Path(ep_ckpt_root) / (exp_id or "default")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                clean_name = m_cfg_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                ckpt_path = ckpt_dir / f"{clean_name}_split{split_idx}.pt"
                torch.save(
                    {
                        "model_state_dict": module.model.state_dict(),
                        "model_type": m_type,
                        "model_name": m_cfg_name,
                        "tau_train": tau_train,
                        "split_idx": split_idx,
                        "input_dim": input_dim,
                        "opt_thresh": float(opt_thresh),
                        "hyperparams": params,
                    },
                    ckpt_path,
                )

            # Evaluate model across each evaluation horizon tau
            for tau in tau_list:
                eval_loader = data_module.get_eval_dataloader(
                    split_idx=split_idx,
                    tau=tau,
                    x_train=x_train,
                    use_v=use_v,
                    input_dim=input_dim,
                )
                if eval_loader is None:
                    continue

                metrics = eval_protocol.evaluate(module, eval_loader, opt_thresh=opt_thresh)
                for k in ["auc", "auprc", "f1_opt", "f1_max", "f1_05"]:
                    tau_metrics[tau][k].append(metrics[k])

        # Aggregate metrics across CV splits
        for tau in tau_list:
            m_dict = tau_metrics[tau]
            if not m_dict["auc"]:
                continue
            results[m_cfg_name]["tau"].append(tau)
            for k in ["auc", "auprc", "f1_opt", "f1_max", "f1_05"]:
                vals = m_dict[k]
                results[m_cfg_name][k].append(float(np.mean(vals)))
                results[m_cfg_name][f"{k}_sem"].append(
                    float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )

    # 5. Output metrics JSON for downstream BasePlotter subclasses
    out_dir = Path(output_dir)
    if exp_id:
        out_dir = out_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for m_cfg_name, _, _, _ in model_configs:
        if results[m_cfg_name].get("tau"):
            clean_key = m_cfg_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            json_path = out_dir / f"metrics_{clean_key}.json"
            with open(json_path, "w") as f:
                json.dump(results[m_cfg_name], f, indent=2)
            log.info("Saved early prediction metrics to %s", json_path)
