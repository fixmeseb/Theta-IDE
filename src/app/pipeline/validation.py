"""Pre-flight configuration validation module.

Validates experiment configs against their declared paradigm before any
training begins. Raises ConfigurationError on fatal mismatches so the
pipeline aborts cleanly at startup.

Paradigm definitions live in in/config/paradigms/. Adding a new paradigm
requires only a new YAML file — no changes to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.app.pipeline.config import normalize_agent_name, parse_methods_dict, resolve_experiment_config_name
from src.app.pipeline.datasets import resolve_dataset_path
from src.app.pipeline.exceptions import ConfigurationError
from src.usr.methods.method_style_registry import METHOD_STYLE

# ---------------------------------------------------------------------------
# Paradigm registry loading
# ---------------------------------------------------------------------------

_PARADIGMS_DIR = Path("in/config/paradigms")


def _load_all_paradigms() -> dict[str, dict]:
    """Load all paradigm YAML files from in/config/paradigms/.

    Returns a dict mapping paradigm name → paradigm definition dict.
    """
    paradigms: dict[str, dict] = {}
    for yaml_path in sorted(_PARADIGMS_DIR.glob("**/*.yaml")):
        name = yaml_path.stem  # filename without extension = paradigm name
        with open(yaml_path) as f:
            defn = yaml.safe_load(f) or {}
        paradigms[name] = defn
    return paradigms


def list_paradigms() -> list[str]:
    """Return sorted list of registered paradigm names."""
    return sorted(_load_all_paradigms().keys())


def load_paradigm(paradigm_name: str) -> dict:
    """Load and return the definition dict for a named paradigm.

    Raises ConfigurationError if the paradigm is not found.
    """
    paradigms = _load_all_paradigms()
    if paradigm_name not in paradigms:
        raise ConfigurationError(
            f"[ConfigurationError] Unknown paradigm '{paradigm_name}'. "
            f"Available paradigms: {sorted(paradigms.keys())}. "
            f"To add a new paradigm create in/config/paradigms/<name>.yaml."
        )
    return paradigms[paradigm_name]


# ---------------------------------------------------------------------------
# Constraint checking helpers
# ---------------------------------------------------------------------------


def _deep_get(cfg: Any, dotted_key: str, default=None):
    """Resolve a dotted key path against a Hydra cfg or plain dict.

    E.g. _deep_get(cfg, 'env.offline_only') returns cfg.env.offline_only.
    """
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            # OmegaConf / MagicMock — use getattr then .get
            try:
                obj = getattr(obj, part)
            except AttributeError:
                try:
                    obj = obj.get(part)
                except Exception:
                    return default
    return obj if obj is not None else default


_CONSTRAINT_CHECKERS = {
    "non_empty": lambda v: bool(v),
    "non_zero": lambda v: bool(v) and v != 0,
    "true": lambda v: v is True,
    "false": lambda v: v is False,
    True: lambda v: v is True,
    False: lambda v: v is False,
}


def _check_paradigm_constraints(cfg: Any, paradigm_name: str, constraints: dict, env_name: str) -> None:
    """Apply requires/forbids rules from the paradigm definition.

    Raises ConfigurationError on any violation.
    """
    for key, rule in (constraints.get("requires") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue  # unknown rule type, skip silently
        val = _deep_get(cfg, key)
        if not _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' requires '{key}' "
                f"to satisfy rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )

    for key, rule in (constraints.get("forbids") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue
        val = _deep_get(cfg, key)
        if _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' forbids '{key}' "
                f"from satisfying rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )


# ---------------------------------------------------------------------------
# Raw experiment YAML loader (for explicit override checks)
# ---------------------------------------------------------------------------


def _load_raw_experiment_yaml(experiment_name: str) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    rel_path = resolve_experiment_config_name(experiment_name)
    exp_path = Path("in/config/experiment") / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    raw_exp = _load_raw_experiment_yaml(experiment_name)

    # --- Paradigm lookup ---
    paradigm_name = cfg.get("paradigm", None)
    if not paradigm_name:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' has no 'paradigm' declared. "
            f"Add 'paradigm: <name>' to its group _base.yaml. "
            f"Available paradigms: {list_paradigms()}."
        )

    paradigm_def = load_paradigm(paradigm_name)  # raises ConfigurationError if unknown
    constraints = paradigm_def.get("constraints", {})

    # --- intervals_count: check raw YAML for explicit override ---
    # Only enforce if the paradigm does not allow intervals > 1.
    # Currently only paradigms with offline data disallow intervals.
    # Paradigms with allows_intervals: true skip this.
    allows_intervals = paradigm_def.get("allows_intervals", False)
    if not allows_intervals:
        explicit_intervals = raw_exp.get("intervals_count", None)
        resolved_intervals = cfg.get("intervals_count", 1)
        intervals_to_check = explicit_intervals if explicit_intervals is not None else resolved_intervals
        if intervals_to_check and intervals_to_check > 1:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' (env '{env_name}') does not "
                f"support intervals_count > 1 (got {intervals_to_check}). "
                f"Progressive dataset slicing requires a paradigm with allows_intervals: true."
            )

    # --- eval_episodes: check raw YAML for explicit override ---
    allows_eval_episodes = paradigm_def.get("allows_eval_episodes", False)
    if not allows_eval_episodes:
        explicit_eval_ep = raw_exp.get("eval_episodes", None)
        resolved_eval_ep = cfg.get("eval_episodes", 0)
        eval_ep_to_check = explicit_eval_ep if explicit_eval_ep is not None else resolved_eval_ep
        if eval_ep_to_check and eval_ep_to_check > 0:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' (env '{env_name}') disables "
                f"simulated gym rollouts — policy actions cannot alter historical trajectories. "
                f"Remove 'eval_episodes' from the experiment config."
            )

    # --- Apply paradigm constraints ---
    _check_paradigm_constraints(cfg, paradigm_name, constraints, env_name)

    # --- Paradigm-Method/Agent Validation ---
    _validate_methods_for_paradigm(cfg, paradigm_name, paradigm_def)

    # --- Optuna sweep direction notice ---
    if is_sweep:
        try:
            sweeper = cfg.get("hydra", {}).get("sweeper", {})
            direction = sweeper.get("direction", None)
            monitor = paradigm_def.get("monitor_metric", None)
            if direction == "maximize" and monitor and "loss" in monitor:
                notices.append(
                    f"Notice: Optuna direction is 'maximize' on paradigm '{paradigm_name}' "
                    f"which monitors '{monitor}'. Consider using 'direction: minimize'."
                )
        except Exception:
            pass

    # --- Method registry checks (non-fatal) ---
    _validate_method_registrations(cfg, notices)

    # --- Offline dataset path checks (non-fatal) ---
    _validate_offline_dataset_paths(cfg, notices)

    return notices


def _validate_methods_for_paradigm(cfg: Any, paradigm_name: str, paradigm_def: dict) -> None:
    """Validate that methods declared in config match the paradigm's agent and model requirements.

    Raises:
        ConfigurationError: If any method is incompatible with the paradigm (e.g. CQL in supervised).
    """
    methods = parse_methods_dict(cfg)
    if not methods:
        return

    allowed_agents = paradigm_def.get("allowed_agents", None)
    forbidden_agents = paradigm_def.get("forbidden_agents", [])

    for method_name, method_cfg in methods.items():
        ag = method_cfg.get("agent")
        if isinstance(ag, dict):
            agent_algo = ag.get("name") or ag.get("algorithm") or ag.get("type") or ag.get("algo")
        elif isinstance(ag, str):
            agent_algo = ag
        else:
            agent_algo = None

        mo = method_cfg.get("model")
        if isinstance(mo, dict):
            model_arch = mo.get("name") or mo.get("architecture") or mo.get("type") or mo.get("base")
        elif isinstance(mo, str):
            model_arch = mo
        else:
            model_arch = None

        # Every method must have a model
        if not model_arch:
            raise ConfigurationError(
                f"[ConfigurationError] Method '{method_name}' is missing required key 'model'. "
                f"Every method must specify a model architecture (e.g. 'model: dnn', 'model: blendrl')."
            )

        # Supervised paradigm validation
        if paradigm_name == "supervised":
            if agent_algo:
                raise ConfigurationError(
                    f"[ConfigurationError] Paradigm 'supervised' does not use RL agent harnesses, "
                    f"but method '{method_name}' declared 'agent: {agent_algo}'. "
                    f"Remove 'agent:' and specify 'model: <architecture>' instead."
                )

        # Paradigms forbidding agents entirely (e.g. allowed_agents: [])
        if allowed_agents is not None and len(allowed_agents) == 0:
            if agent_algo:
                raise ConfigurationError(
                    f"[ConfigurationError] Paradigm '{paradigm_name}' does not allow RL agent harnesses, "
                    f"but method '{method_name}' declared 'agent: {agent_algo}'."
                )

        # Forbidden agents check
        if forbidden_agents and agent_algo:
            if any(agent_algo == f or agent_algo.startswith(f + "_") for f in forbidden_agents):
                raise ConfigurationError(
                    f"[ConfigurationError] Method '{method_name}' uses agent '{agent_algo}', "
                    f"which is forbidden in paradigm '{paradigm_name}'. Forbidden agents: {forbidden_agents}."
                )

        # Paradigms requiring agents (e.g. online_rl, offline_rl)
        if allowed_agents:
            if not agent_algo:
                raise ConfigurationError(
                    f"[ConfigurationError] Paradigm '{paradigm_name}' requires an agent harness (allowed: {allowed_agents}), "
                    f"but method '{method_name}' has no 'agent' declared."
                )
            if not any(agent_algo == a or agent_algo.startswith(a + "_") for a in allowed_agents):
                raise ConfigurationError(
                    f"[ConfigurationError] Method '{method_name}' uses agent '{agent_algo}', "
                    f"which is not allowed in paradigm '{paradigm_name}'. Allowed agents: {allowed_agents}."
                )


def _validate_method_registrations(cfg: Any, notices: list[str]) -> None:
    """Append non-fatal notices for methods not found in METHOD_STYLE registry."""
    registered = set(METHOD_STYLE.keys())

    methods = parse_methods_dict(cfg)

    for method_name, method_cfg in methods.items():
        if "style" in method_cfg and isinstance(method_cfg["style"], dict) and method_cfg["style"].get("label"):
            continue
        ag = method_cfg.get("agent")
        base_algo = ag.get("name") if isinstance(ag, dict) else (ag or "")
        if base_algo and base_algo not in registered and method_name not in registered:
            notices.append(
                f"Notice: Method '{method_name}' uses agent '{base_algo}' which might not match a registered agent style."
            )


def _validate_offline_dataset_paths(cfg: Any, notices: list[str]) -> None:
    pass
