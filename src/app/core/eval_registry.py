"""Evaluator Registry — Single source of truth for pluggable model evaluators & verification protocols.

Evaluators compute verification metrics, rollouts, out-of-distribution tests, or domain scores.
They self-register using the @register_evaluator decorator:
    @register_evaluator("early_pred_auprc")
    class EarlyPredAUPRCEvaluator(BaseEvalProtocol):
        ...

The pipeline resolves evaluators via get_evaluator_class():
    EvalClass = get_evaluator_class("early_pred_auprc")
"""

import importlib
import os
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type

EVALUATOR_REGISTRY: Dict[str, Type[Any]] = {}


def register_evaluator(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register an Evaluator class under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = name.strip()
            EVALUATOR_REGISTRY[clean_name] = cls
            EVALUATOR_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component
                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_evaluator_class(name: str) -> Type[Any] | None:
    """Resolve an evaluator name to its registered class.

    Args:
        name: Identifier of the evaluator (e.g. 'ClassificationEvalProtocol', 'early_pred_auprc').

    Returns:
        The registered class or None if not found.
    """
    if not name:
        return None

    if not EVALUATOR_REGISTRY:
        auto_discover_evaluators()

    clean_name = str(name).strip()
    if clean_name in EVALUATOR_REGISTRY:
        return EVALUATOR_REGISTRY[clean_name]
    if clean_name.lower() in EVALUATOR_REGISTRY:
        return EVALUATOR_REGISTRY[clean_name.lower()]

    # Fallback to paradigm_loader's component registry
    try:
        from src.app.core.paradigm_loader import get_component
        cls = get_component(clean_name)
        if cls is not None:
            return cls
    except Exception:
        pass

    return None


def auto_discover_evaluators() -> None:
    """Auto-discover evaluator modules in src/usr/eval/ and plugin entrypoints."""
    import sys

    core_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(core_dir)
    src_dir = os.path.dirname(app_dir)
    usr_dir = os.path.join(src_dir, "usr")
    eval_dir = os.path.join(usr_dir, "eval")

    if os.path.isdir(eval_dir) and eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    # 1. Discover in src/usr/eval/
    if os.path.isdir(eval_dir):
        for module_info in pkgutil.iter_modules([eval_dir]):
            if module_info.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"src.usr.eval.{module_info.name}")
            except Exception as e:
                print(f"[Warning] Failed to import evaluator module '{module_info.name}': {e}")

    # 2. Register core built-in evaluators
    _register_builtins()


def _register_builtins() -> None:
    """Ensure core built-in eval protocols are registered."""
    try:
        from src.app.core.paradigm_impls.base.online_rl import EpisodicRewardEvalProtocol
        register_evaluator("EpisodicRewardEvalProtocol")(EpisodicRewardEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.offline_rl import OfflineRLEvalProtocol
        register_evaluator("OfflineRLEvalProtocol")(OfflineRLEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.supervised import ClassificationEvalProtocol
        register_evaluator("ClassificationEvalProtocol")(ClassificationEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.unsupervised import ReconstructionEvalProtocol
        register_evaluator("ReconstructionEvalProtocol")(ReconstructionEvalProtocol)
    except Exception:
        pass


def list_registered_evaluators() -> List[str]:
    """Return sorted list of registered evaluator names."""
    auto_discover_evaluators()
    return sorted(set(EVALUATOR_REGISTRY.keys()))
