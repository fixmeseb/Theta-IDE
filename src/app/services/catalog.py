"""What exists on disk: environments, experiments, finished and running runs, their metrics, plots and artifacts.

Paths come from the caller (the Results panel), so every group, experiment and agent name is checked to be a
plain folder name, and run files are only read from inside that run's own results folders.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from . import LOGS_DIR, PLOTS_DIR, PROJECT_ROOT
from .metrics import latest_version_file, read_metrics_table

try:
    from src.usr.algorithms.agent_registry import list_registered_agents
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
        if f.name.startswith("_") or f.parent.name == "methods" or "/methods/" in f.as_posix():
            continue
        rel_path = f.relative_to(exp_dir)
        try:
            exps.append(
                {
                    "path": rel_path.as_posix(),
                    "name": rel_path.with_suffix("").as_posix(),
                    "config": yaml.safe_load(f.read_text(encoding="utf-8")),
                }
            )
        except (OSError, yaml.YAMLError):
            pass
    return exps


# agent_summary results by agent folder, valid while the files they were read from are unchanged.
_SUMMARY_CACHE: dict[Path, tuple[tuple, dict[str, Any] | None]] = {}


def _file_stamp(path: Path | None) -> tuple:
    if path is None:
        return (None,)
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def agent_summary(agent_dir: Path) -> dict[str, Any] | None:
    """Timing and reward summary for one agent of a run, or None if it never logged anything.

    Cached until the newest runtime.json or metrics.csv changes, so listing many runs stays cheap.
    """
    runtime_file = latest_version_file(agent_dir, "runtime.json")
    metrics_file = latest_version_file(agent_dir, "metrics.csv")
    key = agent_dir.resolve()
    stamp = (_file_stamp(runtime_file and runtime_file.resolve()), _file_stamp(metrics_file and metrics_file.resolve()))
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    summary = _read_agent_summary(agent_dir, runtime_file, metrics_file)
    _SUMMARY_CACHE[key] = (stamp, summary)
    return summary


def _read_agent_summary(agent_dir: Path, runtime_file: Path | None, metrics_file: Path | None) -> dict[str, Any] | None:
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
        runs.append(
            {
                "group": run_dir.parent.name,
                "experiment_id": run_dir.name,
                "metadata": metadata,
                "config": run_config_summary(run_dir),
                "agents": agents,
                "finished": bool(agents) and all(agent["finished"] for agent in agents),
            }
        )
    return runs


def check_path_segment(part: str) -> None:
    """Reject names that could step outside results/ (e.g. '..', a slash or a drive letter)."""
    if not part or part in (".", "..") or any(c in part for c in "/\\\x00") or ":" in part:
        raise ValueError(f"Invalid path segment: {part!r}")


def _version_number(name: str) -> int:
    suffix = name.removeprefix("version_")
    return int(suffix) if suffix.isdigit() else -1


def run_metrics(
    group: str, experiment_id: str, agent: str, version: str | None = None, since: int = 0
) -> dict[str, Any]:
    """Every row of an agent's metrics.csv (as text), with every column the engine logged.

    `version` picks a version_N folder (default: the newest). `since` skips rows the caller already has,
    so a client following a run that is still training reads only new rows; `total_rows` says where to
    resume. Raises FileNotFoundError if there are no such metrics, ValueError for a bad name.
    """
    for part in (group, experiment_id, agent):
        check_path_segment(part)
    agent_dir = LOGS_DIR / group / experiment_id / agent
    versions = sorted(
        (p.name for p in agent_dir.glob("version_*") if (p / "metrics.csv").is_file()), key=_version_number
    )
    if version is None:
        metrics_file = latest_version_file(agent_dir, "metrics.csv")
    else:
        name = version if version.startswith("version_") else f"version_{version}"
        check_path_segment(name)
        metrics_file = agent_dir / name / "metrics.csv"
    if metrics_file is None or not metrics_file.exists():
        raise FileNotFoundError(f"No metrics for {group}/{experiment_id}/{agent}")
    with open(metrics_file, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return {
        "metrics": rows[max(int(since), 0) :],
        "columns": columns,
        "total_rows": len(rows),
        "version": metrics_file.parent.name,
        "versions": versions,
        "source": str(metrics_file),
    }


# ── Run manifest: every artifact a run produced ───────────────────────────────

# Top-level folders of results/ that hold per-run output as <root>/<group>/<experiment_id>/...
RESULT_ROOTS = ("logs", "plots", "checkpoints", "tensorboard", "datasets")
MANIFEST_LIMIT = 5000

_KIND_BY_SUFFIX = {
    ".csv": "table",
    ".tsv": "table",
    ".png": "figure",
    ".jpg": "figure",
    ".jpeg": "figure",
    ".svg": "figure",
    ".pdf": "figure",
    ".md": "report",
    ".txt": "report",
    ".html": "report",
    ".yaml": "config",
    ".yml": "config",
    ".json": "metadata",
    ".ckpt": "checkpoint",
    ".pt": "checkpoint",
    ".pth": "checkpoint",
    ".zip": "checkpoint",
    ".npz": "cache",
    ".npy": "cache",
    ".pkl": "cache",
    ".parquet": "cache",
    ".log": "log",
}


def results_dir() -> Path:
    """results/, the parent of results/logs (follows LOGS_DIR, so tests can point it elsewhere)."""
    return LOGS_DIR.parent


def _artifact_kind(root: str, path: Path) -> str:
    if root == "logs" and path.name == "metrics.csv":
        return "metrics"
    if root == "tensorboard" and path.name.startswith("events.out.tfevents"):
        return "tensorboard"
    return _KIND_BY_SUFFIX.get(path.suffix.lower(), "other")


def _run_dirs(group: str, experiment_id: str) -> dict[str, Path]:
    """The existing results/<root>/<group>/<experiment_id> folders of one run, by root."""
    check_path_segment(group)
    check_path_segment(experiment_id)
    base = results_dir()
    dirs = {root: base / root / group / experiment_id for root in RESULT_ROOTS}
    return {root: path for root, path in dirs.items() if path.is_dir()}


def run_manifest(group: str, experiment_id: str) -> dict[str, Any]:
    """Every file a run produced across results/{logs,plots,checkpoints,tensorboard,datasets}.

    Each artifact has a `path` relative to results/ (read it with read_run_file), a `kind` (metrics, table,
    figure, report, config, metadata, checkpoint, cache, tensorboard, log, other), and the `agent` and
    `version` folders it sits under, when it has them. `agents` maps each agent with logged metrics to its
    version folders, oldest first. Raises FileNotFoundError if the run has no folders.
    """
    dirs = _run_dirs(group, experiment_id)
    if not dirs:
        raise FileNotFoundError(f"Run not found: {group}/{experiment_id}")
    base = results_dir()
    artifacts: list[dict[str, Any]] = []
    truncated = False
    for root, run_dir in dirs.items():
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            if len(artifacts) >= MANIFEST_LIMIT:
                truncated = True
                break
            parts = path.relative_to(run_dir).parts
            stat = path.stat()
            artifacts.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "root": root,
                    "kind": _artifact_kind(root, path),
                    "agent": parts[0] if len(parts) > 1 else None,
                    "version": next((p for p in parts[1:-1] if p.startswith("version_")), None),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    agents: dict[str, list[str]] = {}
    for item in artifacts:
        if item["kind"] == "metrics" and item["agent"] and item["version"]:
            agents.setdefault(item["agent"], []).append(item["version"])
    return {
        "group": group,
        "experiment_id": experiment_id,
        "agents": {name: sorted(versions, key=_version_number) for name, versions in sorted(agents.items())},
        "artifacts": artifacts,
        "truncated": truncated,
    }


def read_run_file(group: str, experiment_id: str, path: str) -> bytes:
    """One artifact of a run, by the `path` its manifest lists (relative to results/).

    Only files inside this run's own results/<root>/<group>/<experiment_id> folders are read: anything
    else raises PermissionError. A missing file raises FileNotFoundError.
    """
    dirs = _run_dirs(group, experiment_id)
    target = (results_dir().resolve() / path).resolve()
    if not any(target.is_relative_to(d.resolve()) for d in dirs.values()):
        raise PermissionError("Path is outside this run's results")
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_bytes()


def list_plots(group: str, experiment_id: str) -> list[str]:
    plot_dir = PLOTS_DIR / group / experiment_id
    return sorted(f.name for f in plot_dir.glob("*") if f.is_file()) if plot_dir.exists() else []
