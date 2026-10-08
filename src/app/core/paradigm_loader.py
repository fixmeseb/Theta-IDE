"""Paradigm loader: resolves paradigm YAML definitions to concrete component classes.

Components register themselves with @register_component("ClassName").
Paradigm YAML files declare runner/data_module/eval_protocol by class name string.
The loader resolves those strings to actual classes at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: dict[str, Type[Any]] = {}
_discovered: bool = False


def register_component(*names: str):
    """Class decorator to register a component class under one or more names.

    Usage::

        @register_component("OfflineRLRunner")
        class OfflineRLRunner(BaseParadigmRunner):
            ...
    """

    def decorator(cls):
        for name in names:
            _COMPONENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_component(name: str | None) -> Type[Any] | None:
    """Return a registered component class by name, or None if name is None/empty.

    Triggers auto-discovery of all paradigm_impls modules exactly once,
    regardless of whether individual components were imported manually first.
    Raises KeyError if name is provided but not found.
    """
    global _discovered
    if not name:
        return None
    if not _discovered:
        _auto_discover_components()
        _discovered = True
    if name not in _COMPONENT_REGISTRY:
        raise KeyError(
            f"Component '{name}' is not registered. Registered components: {sorted(_COMPONENT_REGISTRY.keys())}"
        )
    return _COMPONENT_REGISTRY[name]


def _auto_discover_components():
    """Import all modules in src/core/paradigm_impls/ to trigger @register_component decorators."""
    import importlib
    import pkgutil
    from pathlib import Path

    impls_dir = Path(__file__).parent / "paradigm_impls"
    if not impls_dir.exists():
        return
    for sub in ["base", "meta"]:
        sub_dir = impls_dir / sub
        if not sub_dir.exists():
            continue
        for module_info in pkgutil.iter_modules([str(sub_dir)]):
            if module_info.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"src.app.core.paradigm_impls.{sub}.{module_info.name}")
            except Exception:
                pass

    try:
        importlib.import_module("src.usr.eval.early_prediction.data_module")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ParadigmDefinition dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParadigmDefinition:
    """Fully resolved paradigm definition with concrete class references."""

    name: str
    type: str  # 'base' or 'meta'
    description: str = ""
    runner_cls: Type[Any] | None = None
    data_module_cls: Type[Any] | None = None
    eval_protocol_cls: Type[Any] | None = None
    default_callback_names: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)  # full raw YAML for access to extra keys


def load_paradigm_definition(paradigm_name: str) -> ParadigmDefinition:
    """Load a paradigm YAML and resolve class names to concrete implementations.

    Args:
        paradigm_name: Name of the paradigm (e.g. 'supervised', 'offline_rl').

    Returns:
        A ParadigmDefinition with resolved class references.

    Raises:
        ConfigurationError if the paradigm is not found.
        KeyError if a declared component class is not registered.
    """
    from src.app.pipeline.validation import load_paradigm

    raw = load_paradigm(paradigm_name)

    return ParadigmDefinition(
        name=paradigm_name,
        type=raw.get("type", "base"),
        description=raw.get("description", ""),
        runner_cls=get_component(raw.get("runner")),
        data_module_cls=get_component(raw.get("data_module")),
        eval_protocol_cls=get_component(raw.get("eval_protocol")),
        default_callback_names=raw.get("default_callbacks") or [],
        raw=raw,
    )
