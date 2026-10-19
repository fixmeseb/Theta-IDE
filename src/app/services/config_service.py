"""Experiment builder contract: the form's fields, composing a config exactly as run_pipeline.py
would, and checking a launch before any process starts."""

from __future__ import annotations

import copy
import functools
import sys
import threading
from collections import OrderedDict
from typing import Any

import yaml
from omegaconf import OmegaConf

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.app.pipeline.compose import (
    _HYDRA_LOCK,
    CONFIG_DIR,
    compose_experiment,
    effective_config,
    method_plans,
    validate_composed,
)
from src.app.pipeline.config import normalize_agent_name

from . import LOGS_DIR

# A Hydra compose costs ~130-200 ms, and the GUI asks for the same one again and again: at startup,
# on every Settings visit, on reopening an experiment, after an undo. The result depends only on its
# arguments, the YAML under in/config, and the agent registry (auto-discovered from src/usr), so
# results are reused while all of those are byte-for-byte unchanged (~3 ms to check).
_USER_CODE_DIR = CONFIG_DIR.parents[1] / "src" / "usr"
_COMPOSE_CACHE: OrderedDict[tuple, tuple[bool, Any]] = OrderedDict()
_COMPOSE_CACHE_SIZE = 64
_COMPOSE_CACHE_LOCK = threading.Lock()


def _inputs_fingerprint() -> int:
    """Hash of everything a compose can depend on: the files (paths and exact contents) and the
    agents registered in memory (hot reload and tests register agents without touching files)."""
    registry_module = sys.modules.get("src.usr.algorithms.agent_registry")  # don't import it just to look
    registry = getattr(registry_module, "AGENT_REGISTRY", None) or {}
    parts: list[Any] = [tuple(sorted((name, id(cls)) for name, cls in registry.items()))]
    for root, pattern in ((CONFIG_DIR, "*.yaml"), (_USER_CODE_DIR, "*.py")):
        for path in sorted(root.rglob(pattern)):
            try:
                parts.append((str(path), path.read_bytes()))
            except OSError:
                continue
    return hash(tuple(parts))


def _cached_on_inputs(fn):
    """Memoize `fn` on its arguments plus _inputs_fingerprint(). Errors are cached too: with the
    same inputs a failed compose fails again. Callers get copies, so they can't alter the cache."""

    @functools.wraps(fn)
    def wrapper(*args):
        key = (fn.__name__, repr(args), _inputs_fingerprint())  # args are strings and lists of strings
        with _COMPOSE_CACHE_LOCK:
            hit = _COMPOSE_CACHE.get(key)
            if hit is not None:
                _COMPOSE_CACHE.move_to_end(key)
        if hit is None:
            try:
                hit = (True, fn(*args))
            except Exception as exc:  # noqa: BLE001 - re-raised below, for this and later calls
                hit = (False, exc)
            with _COMPOSE_CACHE_LOCK:
                _COMPOSE_CACHE[key] = hit
                while len(_COMPOSE_CACHE) > _COMPOSE_CACHE_SIZE:
                    _COMPOSE_CACHE.popitem(last=False)
        ok, value = hit
        if not ok:
            raise value.with_traceback(None)
        return copy.deepcopy(value)

    wrapper.uncached = fn
    return wrapper


def clear_compose_cache() -> None:
    with _COMPOSE_CACHE_LOCK:
        _COMPOSE_CACHE.clear()


# The GUI composes GUI_BASE_EXPERIMENT plus one override per field; defaults are read from the
# composed config rather than duplicated in the GUI.
GUI_BASE_EXPERIMENT = "thetaide/_base"
GUI_METHOD = {"name": "ppo", "agent": "ppo", "model": "dnn"}
GUI_FIELDS = [
    {
        "key": "experiment_id",
        "label": "Experiment name",
        "type": "str",
        "default": "cartpole_ppo_experiment",
        "pattern": r"^[A-Za-z0-9_\-]+$",
        "help": "Results directory name. Letters, digits, '_' and '-' only.",
    },
    {
        "key": "seed",
        "label": "Random seed",
        "type": "int",
        "default_from": "seed",
        "min": 0,
        "max": 2147483647,
        "help": "Seeds environments and network initialization.",
    },
    {
        "key": "total_timesteps",
        "label": "Total timesteps",
        "type": "int",
        "default_from": "total_timesteps",
        "min": 1000,
        "max": 10000000,
        "step": 1000,
        "help": "Environment steps across all training intervals.",
    },
    {
        "key": "methods.ppo.lr",
        "label": "Learning rate",
        "type": "float",
        "default": 0.0003,
        "default_from": "agent.lr",
        "min": 1e-6,
        "max": 1.0,
        "step": 1e-4,
        "help": "Adam learning rate for the PPO policy.",
    },
    {
        "key": "methods.ppo.batch_size",
        "label": "Batch size",
        "type": "int",
        "default": 64,
        "default_from": "agent.batch_size",
        "choices": [32, 64, 128, 256],
        "help": "Minibatch size for each PPO update.",
    },
    {
        "key": "methods.ppo.gamma",
        "label": "Discount factor · γ",
        "type": "float",
        "default": 0.99,
        "default_from": "env.gamma",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "help": "Weight of future rewards.",
    },
    {
        "key": "tensorboard",
        "label": "Log to TensorBoard",
        "type": "bool",
        "default": False,
        "default_from": "tensorboard",
        "help": "Also write TensorBoard logs to results/tensorboard/ for the TensorBoard tab.",
    },
]


class LaunchError(Exception):
    """The experiment can't be launched; `conflict` marks a clash with existing results or jobs."""

    def __init__(self, message: str, conflict: bool = False):
        super().__init__(message)
        self.conflict = conflict


def rollout_notices(plans: dict[str, dict], total_timesteps) -> list[str]:
    notices = []
    for name, plan in plans.items():
        rollout = plan.get("rollout")
        if rollout and rollout["timesteps"] != total_timesteps:
            notices.append(
                f"Method '{name}' trains {rollout['timesteps']:,} steps, not {total_timesteps:,}: PPO collects whole "
                f"rollouts of {rollout['size']} steps ({rollout['num_envs']} envs × {rollout['num_steps']} steps), "
                f"so it runs {rollout['rollouts']} rollouts."
            )
    return notices


def effective_timesteps(plans: dict[str, dict], total_timesteps):
    """Environment steps training will actually run: the largest method's rounded-up PPO budget."""
    rollouts = [plan["rollout"]["timesteps"] for plan in plans.values() if plan.get("rollout")]
    return max(rollouts, default=total_timesteps)


@_cached_on_inputs
def config_schema() -> dict[str, Any]:
    """The builder's fields with defaults read from the composed base experiment."""
    composed = None
    try:
        composed = compose_experiment(GUI_BASE_EXPERIMENT)
    except Exception:
        exp_dir = CONFIG_DIR / "experiment"
        if exp_dir.is_dir():
            candidates = sorted(
                p for p in exp_dir.rglob("*.yaml")
                if not p.name.startswith("_") and "/methods/" not in p.as_posix() and p.is_file()
            )
            if candidates:
                try:
                    composed = compose_experiment(str(candidates[0].relative_to(exp_dir).with_suffix("")))
                except Exception:
                    pass

    if composed is not None:
        fields = []
        for field in GUI_FIELDS:
            field = dict(field)
            source = field.pop("default_from", None)
            if source:
                val = OmegaConf.select(composed.cfg, source)
                if val is None and source == "agent.lr":
                    val = OmegaConf.select(composed.cfg, "agent.learning_rate")
                if val is not None:
                    field["default"] = val
            fields.append(field)
        method_prefix = f"methods.{GUI_METHOD['name']}"
        return {
            "base_experiment": composed.experiment,
            "environment": {
                "name": getattr(composed.cfg.env, "name", None)
                if hasattr(composed.cfg, "env") and composed.cfg.env
                else None,
                "env_id": composed.cfg.env.get("env_id")
                if hasattr(composed.cfg, "env") and composed.cfg.env and hasattr(composed.cfg.env, "get")
                else None,
            },
            "fixed_overrides": [
                f"++{method_prefix}.agent={GUI_METHOD['agent']}",
                f"++{method_prefix}.model={GUI_METHOD['model']}",
            ],
            "fields": fields,
        }

    # Clean setup fallback: verify Hydra composition without requiring any experiment files
    with _HYDRA_LOCK:
        GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR))
        cfg = compose(config_name="config", return_hydra_config=True)

    return {
        "base_experiment": None,
        "environment": {"name": None, "env_id": None},
        "fixed_overrides": [],
        "fields": [dict(f) for f in GUI_FIELDS],
    }


@_cached_on_inputs
def compose_report(experiment: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose and validate an experiment exactly as `run_pipeline.py` would, without running it."""
    overrides = list(overrides or [])
    argv = ["python", "run_pipeline.py", experiment, *overrides]
    try:
        composed = compose_experiment(experiment, overrides)
    except Exception as e:
        return {
            "valid": False,
            "experiment": experiment,
            "argv": argv,
            "config": None,
            "config_yaml": "",
            "notices": [],
            "errors": [{"stage": "compose", "message": str(e)}],
        }

    errors, notices, methods = [], [], {}
    try:
        notices = validate_composed(composed)
    except Exception as e:
        errors.append({"stage": "validation", "message": str(e).strip()})
    try:
        methods = method_plans(composed)
        notices.extend(rollout_notices(methods, composed.cfg.get("total_timesteps")))
    except Exception as e:
        errors.append({"stage": "methods", "message": str(e).strip()})
    config = effective_config(composed.cfg)
    return {
        "valid": not errors,
        "experiment": composed.experiment,
        "argv": composed.argv,
        "config": config,
        "config_yaml": yaml.safe_dump(config, sort_keys=False),
        "methods": methods,
        "methods_yaml": yaml.safe_dump({name: plan["settings"] for name, plan in methods.items()}, sort_keys=False),
        "notices": notices,
        "errors": errors,
    }


def prepare_launch(
    experiment: str, overrides: list[str] | None = None, overwrite: bool = False, active_jobs: list[dict] | None = None
) -> dict[str, Any]:
    """Validate a launch exactly as the pipeline will and describe the job to create.

    Raises LaunchError for an invalid config, for existing results that launching would delete
    (unless `overwrite`), or for a queued/running job already using the same results folder.
    """
    overrides = list(overrides or [])
    try:
        composed = compose_experiment(experiment, overrides)
        validate_composed(composed)
        plans = method_plans(composed)
    except Exception as e:
        raise LaunchError(str(e).strip()) from e

    cfg = composed.cfg
    log_dir = LOGS_DIR / cfg.group / cfg.experiment_id
    if log_dir.exists() and any(log_dir.iterdir()) and not overwrite and not cfg.get("recover", False):
        raise LaunchError(
            f"Results already exist for {cfg.group}/{cfg.experiment_id}; launching would delete them. "
            "Choose another experiment_id.",
            conflict=True,
        )
    clash = next(
        (j for j in (active_jobs or []) if (j.get("group"), j.get("experiment_id")) == (cfg.group, cfg.experiment_id)),
        None,
    )
    if clash:
        raise LaunchError(
            f"Job {clash['job_id']} ({clash['status']}) already uses {cfg.group}/{cfg.experiment_id}; "
            "running both would overwrite one run's results with the other's.",
            conflict=True,
        )
    agents = [plan["settings"].get("name", normalize_agent_name(name)) for name, plan in plans.items()]
    return {
        "experiment": composed.experiment,
        "overrides": overrides,
        "group": cfg.group,
        "experiment_id": cfg.experiment_id,
        "agents": agents,
        "total_timesteps": cfg.get("total_timesteps"),
        "effective_timesteps": effective_timesteps(plans, cfg.get("total_timesteps")),
    }
