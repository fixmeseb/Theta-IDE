"""Plotter Registry — Single source of truth for pluggable experiment plotters & visualization components.

Plotters compute convergence graphs, loss curves, domain verification plots, or executive reports.
They self-register using the @register_plotter decorator:
    @register_plotter("convergence")
    class ConvergencePlotter(BasePlotter):
        ...

The pipeline resolves plotters via get_plotter_class():
    PlotterClass = get_plotter_class("convergence")
"""

import importlib
import logging
import os
from pathlib import Path
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type

log = logging.getLogger(__name__)

PLOTTER_REGISTRY: Dict[str, Type[Any]] = {}


def register_plotter(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a Plotter class under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = str(name).strip()
            PLOTTER_REGISTRY[clean_name] = cls
            PLOTTER_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component
                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_plotter_class(name: str) -> Optional[Type[Any]]:
    """Resolve a plotter name to its registered class.

    Args:
        name: Identifier of the plotter (e.g. 'convergence', 'losses', 'reports').

    Returns:
        The registered class or None if not found.
    """
    if not name:
        return None

    if not PLOTTER_REGISTRY:
        auto_discover_plotters()

    clean_name = str(name).strip()
    if clean_name in PLOTTER_REGISTRY:
        return PLOTTER_REGISTRY[clean_name]
    if clean_name.lower() in PLOTTER_REGISTRY:
        return PLOTTER_REGISTRY[clean_name.lower()]

    # Fallback to paradigm_loader's component registry
    try:
        from src.app.core.paradigm_loader import get_component
        cls = get_component(clean_name)
        if cls is not None:
            return cls
    except Exception:
        pass

    return None


def list_registered_plotters() -> List[str]:
    """Return sorted list of unique registered plotter names."""
    if not PLOTTER_REGISTRY:
        auto_discover_plotters()
    return sorted(list(PLOTTER_REGISTRY.keys()))


def auto_discover_plotters() -> None:
    """Auto-discover plotter modules in plot/ and third-party component directories."""
    import sys

    core_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(core_dir)
    src_dir = os.path.dirname(app_dir)
    root_dir = os.path.dirname(src_dir)
    plot_dir = os.path.join(root_dir, "plot")

    if os.path.isdir(plot_dir) and plot_dir not in sys.path:
        sys.path.insert(0, plot_dir)

    try:
        from plot.base import BasePlotter
    except Exception as e:
        log.debug("Could not import BasePlotter during discovery: %s", e)
        return

    # 1. Discover standalone modules in plot/*.py
    if os.path.isdir(plot_dir):
        for module_info in pkgutil.iter_modules([plot_dir]):
            if module_info.name.startswith("_") or module_info.name in ("base", "manager"):
                continue
            try:
                mod = importlib.import_module(f"plot.{module_info.name}")
                # Auto-register any BasePlotter subclasses if not already registered
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name)
                    if (
                        isinstance(cls, type)
                        and issubclass(cls, BasePlotter)
                        and cls is not BasePlotter
                    ):
                        name = getattr(cls, "name", None)
                        if not name or name == "base":
                            try:
                                inst = cls()
                                name = getattr(inst, "name", None)
                            except Exception:
                                name = module_info.name
                        if name and name != "base":
                            register_plotter(name, module_info.name)(cls)
            except Exception as e:
                log.debug("Failed to inspect module 'plot.%s': %s", module_info.name, e)

        # 2. Discover package-based plotters in plot/<component_dir>/
        for entry in os.scandir(plot_dir):
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                # Scan all modules in the package subdirectory
                for sub_info in pkgutil.iter_modules([entry.path]):
                    if sub_info.name.startswith("_"):
                        continue
                    full_mod_path = f"plot.{entry.name}.{sub_info.name}"
                    try:
                        mod = importlib.import_module(full_mod_path)
                        for attr_name in dir(mod):
                            cls = getattr(mod, attr_name)
                            if (
                                isinstance(cls, type)
                                and issubclass(cls, BasePlotter)
                                and cls is not BasePlotter
                            ):
                                name = getattr(cls, "name", None)
                                if not name or name == "base":
                                    try:
                                        inst = cls()
                                        name = getattr(inst, "name", None)
                                    except Exception:
                                        name = sub_info.name
                                if name and name != "base":
                                    register_plotter(name, sub_info.name)(cls)
                    except Exception as e:
                        log.debug("Failed to inspect package plotter '%s': %s", full_mod_path, e)
