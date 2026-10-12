"""Single source of truth for training command construction.

Builds Hydra override lists from structured method config dicts.
"""

from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.validation import paradigm_uses_agents, paradigm_uses_static_dataset


def _format_hydra_val(x):
    """Format Python values into Hydra override syntax."""
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_format_hydra_val(item) for item in x) + "]"
    elif isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_format_hydra_val(val)}" for k, val in x.items()) + "}"
    elif isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def build_method_overrides(
    method_name: str,
    method_cfg: dict,
    dataset_path=None,
    extra_args: list | None = None,
    cfg=None,
    study_name: str | None = None,
    is_sweep: bool = False,
) -> list[str]:
    """Build Hydra override list from a structured methods: dict entry.

    Args:
        method_name:  Key in the experiment's methods: dict (e.g. 'cql_dnn').
        method_cfg:   The dict value for that entry. Must contain 'agent' and 'model'.
        dataset_path: Resolved filesystem path to the offline dataset (offline only).
        extra_args:   Additional Hydra overrides forwarded from the CLI.
        cfg:          Full Hydra config (used for experiment_name, paradigm).
        study_name:   Optuna study name (optional, for sweep runs).
        is_sweep:     Whether this specific method should run as an Optuna sweep.

    Returns:
        list[str]: Hydra override arguments ready to pass to train.py.
    """
    paradigm = cfg.get("paradigm", "offline_rl") if cfg is not None else "offline_rl"
    # Asked once: a paradigm with no allowed agents drives its methods by model.
    uses_agents = paradigm_uses_agents(paradigm)
    agent_val = method_cfg.get("agent")
    if isinstance(agent_val, dict):
        agent_algo = (
            agent_val.get("name") or agent_val.get("algorithm") or agent_val.get("type") or agent_val.get("algo")
        )
        agent_subparams = {k: v for k, v in agent_val.items() if k not in ("name", "algorithm", "type", "algo")}
    else:
        agent_algo = agent_val
        agent_subparams = {}

    model_val = method_cfg.get("model")
    if isinstance(model_val, dict):
        model_arch = (
            model_val.get("name") or model_val.get("architecture") or model_val.get("type") or model_val.get("base")
        )
        model_subparams = {k: v for k, v in model_val.items() if k not in ("name", "architecture", "type", "base")}
    else:
        model_arch = model_val
        model_subparams = {}

    agent_name = method_cfg.get("name", normalize_agent_name(method_name))
    experiment_name = cfg.get("experiment_name", "") if cfg is not None else ""

    if not model_arch:
        raise ValueError(f"Method '{method_name}' is missing required key 'model' (or 'model.name').")

    overrides = [
        f"+experiment={experiment_name}",
        f"paradigm={paradigm}",
        f"model={model_arch}",
    ]

    def _flatten_overrides(prefix, val):
        if isinstance(val, dict):
            for k, v in val.items():
                _flatten_overrides(f"{prefix}.{k}", v)
        else:
            formatted_v = _format_hydra_val(val)
            overrides.append(f"++{prefix}={formatted_v}")

    # Combine explicit subparams with inherited agent_params/model_params
    merged_agent_params = dict(method_cfg.get("agent_params", {}))
    merged_agent_params.update(agent_subparams)

    if model_arch == "blendrl":
        merged_model_params = dict(method_cfg.get("model_params", {}))
    else:
        merged_model_params = dict(method_cfg.get("model_params", {}))
        merged_model_params.update(model_subparams)

    # For BlendRL, if modules was auto-synthesized from neural/symbolic, omit the redundant modules CLI string
    if model_arch == "blendrl" and not merged_model_params.get("explicit_modules", False):
        merged_model_params.pop("modules", None)
    merged_model_params.pop("explicit_modules", None)

    _flatten_overrides("model", merged_model_params)
    _flatten_overrides("agent", merged_agent_params)

    if not uses_agents:
        overrides.append(f"++model.name={agent_name}")
        overrides.append(f"++agent.name={agent_name}")
    else:
        if not agent_algo:
            raise ValueError(f"Method '{method_name}' is missing required key 'agent' for paradigm '{paradigm}'.")
        overrides.extend(
            [
                f"agent={agent_algo}",
                f"++agent.name={agent_name}",
            ]
        )

    if paradigm_uses_static_dataset(paradigm) and dataset_path is not None:
        safe_ds_path = str(dataset_path)
        if any(c in safe_ds_path for c in "(), "):
            safe_ds_path = f'"{safe_ds_path}"'
        overrides.append(f"++dataset_path={safe_ds_path}")

    # Process remaining method-level or global hyperparameter overrides
    _INTERNAL_KEYS = {
        "agent",
        "model",
        "name",
        "agent_params",
        "model_params",
        "explicit_modules",
        "style",
        "label",
        "color",
        "marker",
        "linestyle",
        "display_name",
        "base",
        "template",
        "from_method",
        "tune",
        "search_space",
        "from_study",
    }
    if (
        model_arch == "blendrl"
        and not method_cfg.get("explicit_modules", False)
        and not merged_model_params.get("explicit_modules", False)
    ):
        _INTERNAL_KEYS.add("modules")
    _MODEL_KEYS = {
        "architecture",
        "modules",
        "rules",
        "ecm_dthr",
        "fyd",
        "fyd_top_k",
        "actor_mode",
        "blender_mode",
        "blend_function",
        "blender",
        "blender_actor",
        "neural_actor",
        "symbolic_actor",
        "neural",
        "symbolic",
        "hidden_sizes",
        "activation",
        "blend_q_values",
    }
    for k, v in method_cfg.items():
        if k in _INTERNAL_KEYS:
            continue
        if k in merged_agent_params or k in merged_model_params:
            continue

        if not uses_agents:
            _flatten_overrides(f"model.{k}", v)
            if k in ("lr", "batch_size", "epochs", "epochs_per_interval", "eval_interval_epochs", "weight_decay"):
                overrides.append(f"++{k}={_format_hydra_val(v)}")
        else:
            if k in _MODEL_KEYS:
                _flatten_overrides(f"model.{k}", v)
            else:
                _flatten_overrides(f"agent.{k}", v)
                if k in (
                    "epochs_per_interval",
                    "eval_interval_epochs",
                    "gamma",
                    "reward_scale",
                    "pos_action_weight",
                    "bellman_loss",
                    "weight_decay",
                ):
                    overrides.append(f"++{k}={_format_hydra_val(v)}")

    # Optuna Sweeper overrides (active ONLY if is_sweep is True or study_name is provided)
    if is_sweep or study_name:
        has_sweeper_override = extra_args and any("hydra/sweeper" in a or "hydra.sweeper" in a for a in extra_args)
        if not has_sweeper_override:
            sweeper_group = "hydra/sweeper=optuna_online" if paradigm == "online_rl" else "hydra/sweeper=optuna_offline"
            overrides.append(sweeper_group)

        if study_name:
            overrides.append(f"++hydra.sweeper.study_name={study_name}")

        tuning_cfg = cfg.get("tuning", {}) if cfg is not None and hasattr(cfg, "get") else {}
        if tuning_cfg:
            if tuning_cfg.get("n_trials"):
                overrides.append(f"++hydra.sweeper.n_trials={tuning_cfg['n_trials']}")
            if tuning_cfg.get("n_jobs"):
                overrides.append(f"++hydra.sweeper.n_jobs={tuning_cfg['n_jobs']}")
            if tuning_cfg.get("storage"):
                overrides.append(f"++hydra.sweeper.storage={tuning_cfg['storage']}")
            direction = tuning_cfg.get("direction") or get_sweep_direction(cfg, paradigm)
            overrides.append(f"++hydra.sweeper.direction={direction}")
            if tuning_cfg.get("monitor_metric"):
                overrides.append(f"++env.monitor_metric={tuning_cfg['monitor_metric']}")

        # Method-specific search space parameters
        tune_dict = method_cfg.get("tune") or method_cfg.get("search_space") or {}
        for k, v in tune_dict.items():
            if "." in k:
                full_k = k
            elif not uses_agents:
                full_k = f"model.{k}"
            elif k in _MODEL_KEYS:
                full_k = f"model.{k}"
            else:
                full_k = f"agent.{k}"
            overrides.append(f"{full_k}={str(v).strip()}")

        if "--multirun" not in overrides and "-m" not in overrides:
            overrides.insert(0, "--multirun")

    if extra_args:
        if not is_sweep and not study_name:
            # Filter out multirun flags if this method is running as a single normal model
            clean_extra = [a for a in extra_args if a not in ("--multirun", "-m")]
            overrides.extend(clean_extra)
        else:
            overrides.extend(extra_args)

    return overrides


def get_sweep_direction(cfg, paradigm: str) -> str:
    """Single source of truth for Optuna sweep direction.

    Checks cfg.hydra.sweeper.direction first (explicit override).
    Defaults: online_rl → 'maximize' (reward), offline_rl → 'minimize' (loss).
    """
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper"):
        explicit = cfg.hydra.sweeper.get("direction", None)
        if explicit:
            return str(explicit)
    return "maximize" if paradigm == "online_rl" else "minimize"
