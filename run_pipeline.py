"""Unified experiment pipeline orchestrator for NeSyRL.

Coordinates online and offline RL methods, Optuna sweeps, and Slurm cluster submissions.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
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

from omegaconf import OmegaConf

from src.app.pipeline.compose import compose_experiment, validate_composed
from src.app.pipeline.config import normalize_agent_name, parse_methods_dict, resolve_experiment_config_name
from src.app.pipeline.datasets import run_plotting
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.optuna_utils import launch_optuna_dashboard
from src.app.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.app.pipeline import runtime


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run_pipeline.py <group>/<experiment_id> [Hydra Overrides]")
        print("Standard Orchestration Overrides (use Hydra syntax, e.g. key=value):")
        print(
            "  site=local            Interactive CLI execution (default, for local machine or cluster interactive node)"
        )
        print("  site=ncshare          Slurm cluster execution on NCShare (GPU consolidated, default)")
        print("  site=ncshare_gpu      Slurm cluster on NCShare GPU partition (consolidated)")
        print("  site=ncshare_common   Slurm cluster on NCShare CPU common partition (one job per method)")
        print("  site=arc              Slurm cluster execution on ARC")
        print("  plot_only=true        Run plotting phase only (no training, no Slurm submission)")
        print("  no_plot=true          Skip automatic plotting")
        print("  no_online=true        Skip online training phase")
        print("  no_offline=true       Skip offline training phase")
        print("  dry_run=true          Validate config and exit")
        print("  sweep=true            Run Optuna hyperparameter sweep")
        print("  dash=true             Launch Optuna dashboard during run")
        print("  dash_only=true        Launch Optuna dashboard and exit")
        print("  remake=true           Force recalculation of summaries")
        print("  consolidate=true      Consolidate Slurm jobs into 1 single GPU job")
        print("  experiment_id=...     Override the experiment directory name")
        sys.exit(0)

    raw_experiment = sys.argv[1]
    extra_args = sys.argv[2:]

    # Resolve experiment config path
    experiment_arg = resolve_experiment_config_name(raw_experiment)

    # Load configuration (shared with the HTTP API so GUI and CLI configs compose identically)
    try:
        print(f"Loading Hydra configuration for '{experiment_arg}'...", flush=True)
        composed = compose_experiment(experiment_arg, extra_args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    cfg = composed.cfg
    sanitized_extra_args = composed.launch_args
    is_sweep = composed.is_sweep

    # Pre-flight validation
    try:
        notices = validate_composed(composed)
        for n in notices:
            print(f"[Config Notice] {n}")
    except ConfigurationError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    if cfg.get("dry_run", False):
        print(f"\n[Validation Success] Experiment config '{experiment_arg}' is valid and ready to run.")
        sys.exit(0)

    if cfg.get("plot_only", False):
        print(f"\n=== Running Plotting Phase Only for '{experiment_arg}' ===")
        run_plotting(
            experiment=experiment_arg,
            style=cfg.get("plot_style", None),
            wipe=cfg.get("wipe", False),
            use_cache=cfg.get("use_cache", False),
        )
        sys.exit(0)

    # Load paradigm definition for component assembly
    paradigm_def = None
    try:
        from src.app.core.paradigm_loader import load_paradigm_definition
        paradigm_name = cfg.get("paradigm", None)
        if paradigm_name:
            paradigm_def = load_paradigm_definition(paradigm_name)
    except Exception as e:
        # Paradigm loading is best-effort during transition; log but don't abort
        print(f"[Notice] Could not load paradigm definition: {e}")

    # Execution mode is determined solely by the site profile: site=local -> interactive CLI, any other site -> Slurm cluster
    site_name = None
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime") and hasattr(cfg.hydra.runtime, "choices"):
        site_name = cfg.hydra.runtime.choices.get("site")
    if not site_name and hasattr(cfg, "site"):
        site_name = getattr(cfg.site, "name", None)
    if not site_name:
        site_name = "local"

    if hasattr(cfg, "site") and OmegaConf.is_config(cfg.site) and "name" not in cfg.site:
        import omegaconf
        with omegaconf.open_dict(cfg.site):
            cfg.site.name = site_name

    is_interactive = site_name == "local"
    print(f"Execution Mode: {'Interactive (Local CLI)' if is_interactive else f'Slurm Cluster ({site_name})'}")

    storage_url = None
    if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
        storage_url = cfg.hydra.sweeper.storage
    if not storage_url and hasattr(cfg, "tuning") and cfg.tuning and hasattr(cfg.tuning, "get") and cfg.tuning.get("storage"):
        storage_url = cfg.tuning.storage
    if not storage_url:
        from src.app.pipeline.optuna_utils import DEFAULT_OPTUNA_DB_URL
        storage_url = DEFAULT_OPTUNA_DB_URL
    if storage_url:
        storage_url = str(storage_url).replace("${experiment_id}", cfg.experiment_id)
    import os
    os.makedirs(runtime.OPTUNA_DIR, exist_ok=True)

    if is_interactive and storage_url and (cfg.get("dash") or cfg.get("dash_only")):
        launch_optuna_dashboard(storage_url)
        if cfg.get("dash_only"):
            print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
            import time

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)

    if cfg.get("dash_only"):
        print("Error: Could not find Optuna storage URL in configuration.")
        sys.exit(1)

    # Parse structured methods: dict with shared params support
    methods_dict = parse_methods_dict(cfg)
    if not methods_dict:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_arg}' has no runnable methods declared under 'methods:'. "
            f"Declare methods as a dict with agent + model per entry."
        )

    # Reject legacy keys
    for legacy_key in ("online_methods", "offline_methods", "offline_datasets"):
        if cfg.get(legacy_key, None):
            raise ConfigurationError(
                f"[ConfigurationError] Legacy key '{legacy_key}' found in config. "
                f"Use the 'methods:' dict instead."
            )

    print("Declared Methods:")
    has_any_tune = is_sweep
    for name, mcfg in methods_dict.items():
        agent_str = f"agent={mcfg.get('agent')}, " if mcfg.get('agent') else ""
        tune_str = " [Optuna Sweep]" if (is_sweep or mcfg.get("tune")) else ""
        if mcfg.get("tune"):
            has_any_tune = True
        print(f"  {name}: {agent_str}model={mcfg.get('model')}{tune_str}")

    # Build context for tasks
    context = {
        "is_interactive": is_interactive,
        "site_name": site_name,
        "sanitized_extra_args": sanitized_extra_args,
        "storage_url": storage_url,
        "is_sweep": is_sweep or has_any_tune,
        "methods": methods_dict,
        "paradigm_def": paradigm_def,
    }

    # Extract task name
    task_name = cfg.get("task", "rl")
    if not task_name:
        task_name = "rl"

    from src.app.pipeline.task_registry import get_task

    task_fn = get_task(task_name)

    # Introspect task_fn to see if it accepts args (backwards compatibility for custom tasks)
    import inspect

    sig = inspect.signature(task_fn)
    if "args" in sig.parameters:
        task_fn(cfg, None, context)
    else:
        task_fn(cfg, context)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        print(f"Initializing BlendRL pipeline for: {sys.argv[1]} ...", flush=True)
    main()
