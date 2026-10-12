"""What exists on disk: environments, experiments, finished and running runs, their metrics and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from . import LOGS_DIR, PLOTS_DIR, PROJECT_ROOT
from .metrics import latest_version_file, read_metrics_table

try:
    from src.usr.methods.agent_registry import list_registered_agents
except ImportError:  # pragma: no cover - registry is optional

    def list_registered_agents():
        return []


CONFIG_DIR = PROJECT_ROOT / "in" / "config"


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_environments() -> list[dict[str, Any]]:
    envs = []
    for f in sorted((CONFIG_DIR / "env").glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        try:
            envs.append({"name": f.stem, "config": yaml.safe_load(f.read_text(encoding="utf-8"))})
        except (OSError, yaml.YAMLError):
            pass
    return envs


def list_methods() -> dict[str, Any]:
    return {"registered_agents": list_registered_agents(), "method_styles": {}}


def list_experiments() -> list[dict[str, Any]]:
    """Experiment configs, skipping internal _base / template files."""
    exp_dir = CONFIG_DIR / "experiment"
    exps = []
    for f in sorted(exp_dir.glob("**/*.yaml")):
        if f.name.startswith("_"):
            continue
        rel_path = f.relative_to(exp_dir)
        try:
            exps.append({"path": rel_path.as_posix(), "name": rel_path.with_suffix("").as_posix(),
                         "config": yaml.safe_load(f.read_text(encoding="utf-8"))})
        except (OSError, yaml.YAMLError):
            pass
    return exps


def agent_summary(agent_dir: Path) -> dict[str, Any] | None:
    """Timing and reward summary for one agent of a run, or None if it never logged anything."""
    runtime_file = latest_version_file(agent_dir, "runtime.json")
    metrics_file = latest_version_file(agent_dir, "metrics.csv")
    if runtime_file is None and metrics_file is None:
        return None
    runtime: dict[str, Any] = {}
    if runtime_file is not None:
        try:
            runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            runtime = {}
    rewards, timesteps = [], None
    if metrics_file is not None:
        try:
            with open(metrics_file, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # Only rows with a step count, as the GUI's charts (metric_points) use: Lightning also
                    # repeats each evaluation in an epoch row without one.
                    transitions = _float(row.get("transitions"))
                    if transitions is None:
                        continue
                    timesteps = max(timesteps or 0, round(transitions))
                    reward = _float(row.get("eval/reward"))
                    if reward is not None:
                        rewards.append(reward)
        except (OSError, csv.Error):
            pass
    return {
        "name": agent_dir.name,
        "started": runtime.get("start_time_iso"),
        "finished": runtime.get("end_time_iso"),
        "training_time_seconds": _float(runtime.get("training_time_seconds")),
        "timesteps": timesteps,
        "best_reward": max(rewards) if rewards else None,
        "latest_reward": rewards[-1] if rewards else None,
    }


def run_config_summary(run_dir: Path) -> dict[str, Any]:
    """The fields of a run's saved Hydra config (config.yaml) that the Results browser shows."""
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    methods = cfg.get("methods") if isinstance(cfg.get("methods"), dict) else {}
    return {
        "env": env.get("name"),
        "env_id": env.get("env_id"),
        "seed": cfg.get("seed"),
        "total_timesteps": cfg.get("total_timesteps"),
        "paradigm": cfg.get("paradigm"),
        "methods": {
            name: {key: settings.get(key) for key in ("agent", "model", "lr", "batch_size", "gamma")}
            for name, settings in methods.items()
            if isinstance(settings, dict)
        },
    }


def list_runs() -> list[dict[str, Any]]:
    """Every experiment folder under results/logs, however it was launched (GUI, CLI or SLURM).

    `agents` lists each agent that logged something, with its timing and reward summary;
    `finished` is true once every agent has recorded an end time.
    """
    runs = []
    if not LOGS_DIR.exists():
        return runs
    for run_dir in LOGS_DIR.glob("*/*"):
        if not run_dir.is_dir():
            continue
        metadata = {}
        runtime_file = run_dir / "runtime.json"
        if runtime_file.exists():
            try:
                metadata = json.loads(runtime_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        agents = [
            summary
            for agent_dir in sorted(p for p in run_dir.iterdir() if p.is_dir())
            if (summary := agent_summary(agent_dir)) is not None
        ]
        runs.append({
            "group": run_dir.parent.name,
            "experiment_id": run_dir.name,
            "metadata": metadata,
            "config": run_config_summary(run_dir),
            "agents": agents,
            "finished": bool(agents) and all(agent["finished"] for agent in agents),
        })
    return runs


def run_metrics(group: str, experiment_id: str, agent: str) -> dict[str, Any]:
    """All rows of an agent's newest metrics.csv. Raises FileNotFoundError if it has none."""
    metrics_file = latest_version_file(LOGS_DIR / group / experiment_id / agent, "metrics.csv")
    if metrics_file is None or not metrics_file.exists():
        raise FileNotFoundError(f"No metrics for {group}/{experiment_id}/{agent}")
    return {"metrics": read_metrics_table(metrics_file), "source": str(metrics_file)}


def list_plots(group: str, experiment_id: str) -> list[str]:
    plot_dir = PLOTS_DIR / group / experiment_id
    return sorted(f.name for f in plot_dir.glob("*") if f.is_file()) if plot_dir.exists() else []
