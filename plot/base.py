import os
import sys

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
for p in [
    src_path,
    os.path.join(src_path, "app"),
    os.path.join(src_path, "usr"),
    os.path.join(src_path, "usr", "models"),
    os.path.join(src_path, "usr", "environments"),
    os.path.join(src_path, "usr", "eval"),
    os.path.join(src_path, "usr", "models", "fyd_repo", "src"),
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Import styling and alias resolution from the unified method registry
from src.usr.methods.method_registry import clean_label, get_canonical_method_name, get_method_aliases, get_style_info
from src.app.pipeline import runtime


def moving_average(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) == 0:
        return np.array([])
    n = min(len(a), max(1, n))
    a_padded = np.pad(a, (n - 1, 0), mode="edge")
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def deep_update(base: dict, update: dict) -> dict:
    """Recursively updates a nested dictionary."""
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


class BasePlotter:
    name: str = "base"

    def __init__(self, name: str):
        self.name = name
        self.plot_dir = Path(__file__).parent
        self.default_config_path = self.plot_dir / f"_{name}_config.yaml"
        self.default_cfg = self._load_yaml(self.default_config_path)

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_config_defaults(self, raw_cfg: dict, visited: set | None = None) -> dict:
        """Recursively resolves Hydra-style defaults entries."""
        if visited is None:
            visited = set()

        resolved = dict(raw_cfg)
        defaults = resolved.pop("defaults", None)
        if not defaults or not isinstance(defaults, list):
            return resolved

        base_acc = {}
        for item in defaults:
            if isinstance(item, str):
                item_str = item.strip()
                if item_str == "_self_":
                    continue
                # Handle override patterns
                if item_str.startswith("override /env:"):
                    env_name = item_str.split(":", 1)[1].strip()
                    env_path = Path(f"{runtime.CONFIG_DIR}/env/{env_name}.yaml")
                    if env_path.exists():
                        base_acc["env"] = self._load_yaml(env_path)
                    continue
                if item_str.startswith("override /agent:"):
                    agent_name = item_str.split(":", 1)[1].strip()
                    agent_path = Path(f"{runtime.CONFIG_DIR}/agent/{agent_name}.yaml")
                    if agent_path.exists():
                        base_acc["agent"] = self._load_yaml(agent_path)
                    continue
                # Experiment base config (e.g. mimic/_base or cartpole/_base)
                base_cands = [
                    Path(f"{runtime.CONFIG_DIR}/experiment/{item_str}.yaml"),
                    Path(f"{runtime.CONFIG_DIR}/experiment/{Path(item_str).stem}.yaml"),
                ]
                for bcand in base_cands:
                    if bcand.exists() and str(bcand) not in visited:
                        visited.add(str(bcand))
                        parent_raw = self._load_yaml(bcand)
                        parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                        base_acc = deep_update(base_acc, parent_resolved)
                        break
            elif isinstance(item, dict):
                for k, v in item.items():
                    if k.startswith("override /env"):
                        env_path = Path(f"{runtime.CONFIG_DIR}/env/{v}.yaml")
                        if env_path.exists():
                            base_acc["env"] = self._load_yaml(env_path)
                    elif k.startswith("override /agent"):
                        agent_path = Path(f"{runtime.CONFIG_DIR}/agent/{v}.yaml")
                        if agent_path.exists():
                            base_acc["agent"] = self._load_yaml(agent_path)
                    elif isinstance(v, str):
                        base_cand = Path(f"{runtime.CONFIG_DIR}/experiment/{v}.yaml")
                        if base_cand.exists() and str(base_cand) not in visited:
                            visited.add(str(base_cand))
                            parent_raw = self._load_yaml(base_cand)
                            parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                            base_acc = deep_update(base_acc, parent_resolved)

        return deep_update(base_acc, resolved)

    def get_experiment_config(self, exp_id: str, exp_config_name: str | None = None) -> dict:
        """Finds, resolves, and loads the experiment configuration YAML or saved run config."""
        clean_exp = Path(exp_id).stem

        group_hint = None
        if "/" in exp_id:
            group_hint = exp_id.split("/")[0]
        elif exp_config_name and "/" in exp_config_name:
            group_hint = exp_config_name.split("/")[0]

        # 1. Check saved config.yaml from the experiment's log or checkpoint directory first
        # to ensure we capture the true group and execution hyperparameters
        saved_cfg = {}
        for base_dir in [Path(runtime.LOGS_DIR), Path(runtime.CHECKPOINTS_DIR)]:
            if base_dir.exists():
                if group_hint:
                    target_direct = base_dir / group_hint / clean_exp / "config.yaml"
                    if target_direct.exists():
                        saved_cfg = self._load_yaml(target_direct)
                        break
                    target_nested = list(base_dir.glob(f"{group_hint}/{clean_exp}/*/config.yaml"))
                    if target_nested:
                        saved_cfg = self._load_yaml(target_nested[0])
                        break
                matches = list(base_dir.glob(f"*/{clean_exp}/config.yaml"))
                if matches:
                    saved_cfg = self._load_yaml(matches[0])
                    break
                matches_nested = list(base_dir.glob(f"*/{clean_exp}/*/config.yaml"))
                if matches_nested:
                    saved_cfg = self._load_yaml(matches_nested[0])
                    break

        if not group_hint:
            if saved_cfg and saved_cfg.get("group"):
                group_hint = str(saved_cfg.get("group"))
            else:
                for base_dir in [Path(runtime.LOGS_DIR), Path(runtime.CHECKPOINTS_DIR), Path(runtime.PLOTS_DIR)]:
                    if base_dir.exists():
                        for g_dir in base_dir.iterdir():
                            if g_dir.is_dir() and (g_dir / clean_exp).exists():
                                group_hint = g_dir.name
                                break
                        if group_hint:
                            break

        # 2. Find live experiment YAML candidates
        candidates = []
        if exp_config_name:
            clean_base = Path(exp_config_name).stem
            candidates.extend(
                [
                    Path(f"{runtime.CONFIG_DIR}/experiment/{exp_config_name}.yaml"),
                    Path(f"{runtime.CONFIG_DIR}/experiment/{clean_base}.yaml"),
                ]
            )
            if group_hint:
                candidates.append(Path(f"{runtime.CONFIG_DIR}/experiment/{group_hint}/{clean_base}.yaml"))
            candidates.extend(list(Path(f"{runtime.CONFIG_DIR}/experiment").glob(f"**/{clean_base}.yaml")))

        if group_hint:
            candidates.append(Path(f"{runtime.CONFIG_DIR}/experiment/{group_hint}/{clean_exp}.yaml"))
        candidates.extend(
            [
                Path(f"{runtime.CONFIG_DIR}/experiment/{exp_id}.yaml"),
                Path(f"{runtime.CONFIG_DIR}/experiment/{clean_exp}.yaml"),
            ]
        )
        if not group_hint:
            candidates.extend(list(Path(f"{runtime.CONFIG_DIR}/experiment").glob(f"**/{clean_exp}.yaml")))

        live_cfg = {}
        for cand in candidates:
            if cand.exists():
                raw = self._load_yaml(cand)
                live_cfg = self._resolve_config_defaults(raw)
                break

        if live_cfg and saved_cfg:
            # Saved config represents the ground-truth hyperparameters and methods executed.
            # Live config allows overriding visualization-only keys (plots, default_plots, etc.).
            merged = dict(saved_cfg)
            for k, v in live_cfg.items():
                if k in ("plots", "default_plots", "style", "plot_style"):
                    merged[k] = v
                elif k not in merged:
                    merged[k] = v
            return merged
        elif live_cfg:
            return live_cfg
        elif saved_cfg:
            return saved_cfg

        return {}

    def get_group(self, exp_id: str, exp_config: dict) -> str:
        """Resolves group for the given experiment ID."""
        # If exp_id has a group prefix like mimic/mimic_test, prioritize it
        if "/" in exp_id:
            parts = exp_id.split("/")
            return parts[0]

        if "group" in exp_config and exp_config["group"]:
            return exp_config["group"]

        clean_exp = Path(exp_id).stem
        # Scan results/logs/*/clean_exp and results/checkpoints/*/clean_exp
        for base_dir in [Path(runtime.LOGS_DIR), Path(runtime.CHECKPOINTS_DIR), Path(runtime.PLOTS_DIR)]:
            if base_dir.exists():
                for g_dir in base_dir.iterdir():
                    if g_dir.is_dir() and (g_dir / clean_exp).exists():
                        return g_dir.name
        return "ungrouped"

    def get_effective_config(
        self, exp_id: str, cli_overrides: dict | None = None, exp_config_name: str | None = None
    ) -> tuple[dict, str, Path]:
        """
        Merges default module config < default_cfg
               < experiment config plots.<module_name>
               < CLI overrides
        Returns (merged_config, group, output_dir).
        """
        exp_cfg = self.get_experiment_config(exp_id, exp_config_name=exp_config_name)
        group = self.get_group(exp_id, exp_cfg)
        clean_exp = Path(exp_id).stem

        # Extract per-plotter options from experiment YAML
        exp_plot_opts = {}
        plots_sec = exp_cfg.get("plots", {})
        if isinstance(plots_sec, dict) and self.name in plots_sec:
            if isinstance(plots_sec[self.name], dict):
                exp_plot_opts = plots_sec[self.name]

        merged = deep_update(self.default_cfg, exp_plot_opts)
        if cli_overrides:
            merged = deep_update(merged, cli_overrides)

        # Allow output_dir to be explicitly overridden in plotter config, experiment config, or CLI
        if "output_dir" in merged and merged["output_dir"]:
            custom_dir = str(merged["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
            output_dir = Path(custom_dir)
        elif "output_dir" in exp_cfg and exp_cfg["output_dir"]:
            custom_dir = str(exp_cfg["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
            output_dir = Path(custom_dir)
        else:
            output_dir = Path(runtime.PLOTS_DIR) / group / clean_exp
        output_dir.mkdir(parents=True, exist_ok=True)
        return merged, group, output_dir

    def load_metrics(self, group: str, exp_id: str) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Loads metrics.csv files for all runs matching results/logs/[group]/[exp_id]/[method]/*.
        Filters by active online_methods and offline_methods from experiment config if defined.
        Returns dict: { method_name: { version_str: df } }
        """
        clean_exp = Path(exp_id).stem
        exp_dir = Path(runtime.LOGS_DIR) / group / clean_exp
        if not exp_dir.exists():
            print(f"Warning: Log directory {exp_dir} not found.")
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases = set()
        has_active_filter = False
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, [])
            if val:
                has_active_filter = True
                if isinstance(val, (list, tuple)):
                    methods = list(val)
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    active_aliases.update(get_method_aliases(m))

        results = {}
        for method_dir in sorted(exp_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            raw_method_name = method_dir.name
            if has_active_filter:
                parts = raw_method_name.rsplit("_", 1)
                is_active = (raw_method_name in active_aliases) or (
                    len(parts) == 2 and parts[0] in active_aliases and parts[1].isdigit()
                )
                if not is_active:
                    continue

            canon_name = raw_method_name
            if canon_name not in results:
                results[canon_name] = {}

            # Check version_X subdirectories
            version_dirs = sorted([d for d in method_dir.glob("version_*") if d.is_dir()])
            if not version_dirs:
                version_dirs = [method_dir]

            for v_dir in version_dirs:
                csv_path = v_dir / "metrics.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            v_key = (
                                f"{raw_method_name}_{v_dir.name}"
                                if canon_name in results and v_dir.name in results[canon_name]
                                else v_dir.name
                            )
                            results[canon_name][v_key] = df
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        # Filter out empty method entries
        return {k: v for k, v in results.items() if v}

    def plot_metric_series(self, exp_id: str, group: str, output_dir: Path, metrics: list, cfg: dict):
        """Standard multi-method metric plotting with multi-version mean±SEM.

        Shared implementation used by ConvergencePlotter, LossesPlotter, and
        any future plotter that plots time-series metrics from metrics.csv.
        """
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"No log data found for experiment '{exp_id}' in group '{group}'.")
            return

        window = cfg.get("smoothing_window", 10)
        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [8, 5]))
        x_axis_col = cfg.get("x_axis", "transitions")
        subdir = cfg.get("output_subdir", self.name)
        filename_prefix = cfg.get("filename_prefix", "")

        out_dir = (output_dir / subdir) if subdir else output_dir
        any_saved = False

        print(f"=== Generating {self.name.title()} Plots for '{exp_id}' ===")

        for metric in metrics:
            plt.figure(figsize=figsize)
            has_data = False
            used_xlabel = None
            for method_name, versions in sorted(runs_data.items()):
                all_x = []
                all_y = []

                for v_name, df in versions.items():
                    if metric in df.columns:
                        valid_df = df.dropna(subset=[metric])
                        if not valid_df.empty:
                            x_vals = None
                            if x_axis_col in df.columns and df[x_axis_col].notna().any():
                                full_x = df[x_axis_col].interpolate(method="linear").ffill().bfill()
                                s_x = full_x.loc[valid_df.index]
                                if not s_x.empty and s_x.nunique() > 1 and not s_x.isna().any():
                                    x_vals = s_x.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", x_axis_col.replace("_", " ").title())
                            if x_vals is None:
                                if "step" in valid_df.columns and valid_df["step"].nunique() > 1:
                                    x_vals = valid_df["step"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Training Steps")
                                elif "epoch" in valid_df.columns and valid_df["epoch"].nunique() > 1:
                                    x_vals = valid_df["epoch"].values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Epoch")
                                else:
                                    x_vals = valid_df.index.values
                                    if used_xlabel is None:
                                        used_xlabel = cfg.get("xlabel", "Index")
                            y_vals = valid_df[metric].values
                            all_x.append(x_vals)
                            all_y.append(y_vals)

                if all_y:
                    # Resolve style from experiment config or group methods YAML
                    method_style = None
                    if isinstance(cfg, dict) and "methods" in cfg and isinstance(cfg["methods"], dict):
                        m_entry = cfg["methods"].get(method_name, {})
                        if isinstance(m_entry, dict) and "style" in m_entry:
                            method_style = m_entry["style"]
                    if method_style is None:
                        from src.app.pipeline.config import find_group_method_config
                        gm_cfg = find_group_method_config(method_name, group=group)
                        if gm_cfg and "style" in gm_cfg:
                            method_style = gm_cfg["style"]

                    display_name = clean_label(method_name, style_override=method_style)
                    color, ls, marker = get_style_info(method_name, style_override=method_style)

                    if len(all_y) > 1:
                        # Multi-version: compute mean ± SEM across versions
                        min_len = min(len(y) for y in all_y)
                        trimmed = np.array([y[:min_len] for y in all_y])
                        y_mean = np.mean(trimmed, axis=0)
                        y_sem = np.std(trimmed, axis=0) / np.sqrt(len(all_y))
                        y_smoothed = moving_average(y_mean, window)
                        sem_smoothed = moving_average(y_sem, window)
                        x_plot = all_x[0][: len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color, linestyle=ls, linewidth=2.0)
                        plt.fill_between(
                            x_plot, y_smoothed - sem_smoothed, y_smoothed + sem_smoothed, color=color, alpha=0.15
                        )
                    else:
                        # Single version: simple moving average
                        y_smoothed = moving_average(all_y[0], window)
                        x_plot = all_x[0][: len(y_smoothed)]
                        plt.plot(x_plot, y_smoothed, label=display_name, color=color, linestyle=ls, linewidth=2.0)

            if has_data:
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Resolve xlabel: explicit config > detected used_xlabel > fallback
                xlabel = cfg.get("xlabel") or used_xlabel or "Training Steps"
                plt.xlabel(xlabel)

                # Resolve ylabel (dictionary per metric or string or fallback)
                if isinstance(cfg.get("ylabel"), dict):
                    ylabel = cfg["ylabel"].get(metric, metric.replace("_", " ").title())
                elif cfg.get("ylabel"):
                    ylabel = str(cfg["ylabel"])
                else:
                    ylabel = metric.replace("_", " ").title()
                plt.ylabel(ylabel)

                # Resolve title (dictionary per metric, string template, or default)
                clean_metric = metric.split("/")[-1].replace("_", " ").title()
                if isinstance(cfg.get("title"), dict):
                    title = cfg["title"].get(metric, f"{exp_id.upper()}: {metric}")
                elif cfg.get("title"):
                    try:
                        title = str(cfg["title"]).format(
                            exp_id=exp_id.upper(),
                            metric=metric,
                            clean_metric=clean_metric,
                        )
                    except Exception:
                        title = str(cfg["title"])
                else:
                    title = f"{exp_id.upper()}: {metric}"
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()

                safe_metric_name = metric.replace("/", "_")
                out_path = out_dir / f"{filename_prefix}{safe_metric_name}.png"
                plt.savefig(out_path, dpi=dpi)
                plt.close()
                print(f"  Saved: {out_path}")
                any_saved = True
            else:
                plt.close()

        if out_dir.exists() and not any_saved and not any(out_dir.iterdir()):
            try:
                out_dir.rmdir()
            except Exception:
                pass

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        raise NotImplementedError("Subclasses must implement run()")
