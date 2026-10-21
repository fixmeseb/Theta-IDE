"""DataModule Registry — Single source of truth for pluggable DataModules.

Data modules manage data ingestion, preprocessing, augmentation, and dataloader emission.
They self-register using the @register_datamodule decorator:
    @register_datamodule("my_data_module")
    class MyDataModule(L.LightningDataModule):
        ...

The pipeline resolves data modules via get_datamodule_class():
    DataModuleClass = get_datamodule_class("my_data_module")
"""

import importlib
import logging
import os
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type

DATAMODULE_REGISTRY: Dict[str, Type[Any]] = {}
_DISCOVERY_COMPLETE = False
log = logging.getLogger(__name__)


def register_datamodule(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a DataModule under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = name.strip()
            DATAMODULE_REGISTRY[clean_name] = cls
            DATAMODULE_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component

                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_datamodule_class(dm_name: str) -> Type[Any] | None:
    """Resolve a data module name to its registered class.

    Args:
        dm_name: Identifier of the data module (e.g., 'SimulatorDataModule').

    Returns:
        The registered class or None if not found.
    """
    if not dm_name:
        return None

    if not DATAMODULE_REGISTRY:
        auto_discover_datamodules()

    clean_name = str(dm_name).strip()
    if clean_name in DATAMODULE_REGISTRY:
        return DATAMODULE_REGISTRY[clean_name]
    if clean_name.lower() in DATAMODULE_REGISTRY:
        return DATAMODULE_REGISTRY[clean_name.lower()]

    # Fallback to paradigm_loader's component registry
    try:
        from src.app.core.paradigm_loader import get_component

        cls = get_component(clean_name)
        if cls is not None:
            return cls
    except Exception:
        pass

    return None


def auto_discover_datamodules() -> None:
    """Auto-discover data modules in src/usr/data/, src/app/data/, and plugin entrypoints."""
    import sys

    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE:
        return
    # Discovery is invoked whenever the UI asks for choices.  Mark it complete
    # up front so a missing optional training dependency cannot repeatedly
    # emit the same import warning while the user edits a config.
    _DISCOVERY_COMPLETE = True

    core_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(core_dir)
    src_dir = os.path.dirname(app_dir)
    usr_dir = os.path.join(src_dir, "usr")
    data_dir = os.path.join(usr_dir, "data")
    app_data_dir = os.path.join(app_dir, "data")

    for p in (data_dir, app_data_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # 1. Discover in src/usr/data/
    if os.path.isdir(data_dir):
        for module_info in pkgutil.iter_modules([data_dir]):
            if module_info.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"src.usr.data.{module_info.name}")
            except ModuleNotFoundError as exc:
                if exc.name in {"lightning", "pytorch_lightning"}:
                    log.debug(
                        "Skipping data module '%s': Lightning is not installed.",
                        module_info.name,
                    )
                    continue
                log.debug("Skipping data module '%s': %s", module_info.name, exc, exc_info=True)
            except Exception as e:
                # User-contributed data modules are optional.  Preserve the
                # traceback for diagnostics without turning a picker refresh
                # into a visible application warning.
                log.debug("Skipping data module '%s': %s", module_info.name, e, exc_info=True)

    # 2. Register core built-in data modules
    _register_builtins()


def _register_builtins() -> None:
    """Ensure core built-in data modules are registered."""
    try:
        from src.app.data.rl_data_module import RLDataModule

        register_datamodule("RLDataModule")(RLDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.online_rl import SimulatorDataModule

        register_datamodule("SimulatorDataModule")(SimulatorDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.offline_rl import RLReplayBufferDataModule

        register_datamodule("RLReplayBufferDataModule")(RLReplayBufferDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.supervised import SupervisedDataModule

        register_datamodule("SupervisedDataModule", "CrossValidationDataModule")(SupervisedDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.tabular import SupervisedNpzDataModule, TabularNpzDataModule

        register_datamodule("TabularNpzDataModule")(TabularNpzDataModule)
        register_datamodule("SupervisedNpzDataModule")(SupervisedNpzDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.unsupervised import UnsupervisedDataModule

        register_datamodule("UnsupervisedDataModule")(UnsupervisedDataModule)
    except Exception:
        pass

def list_registered_datamodules() -> List[str]:
    """Return sorted list of registered data module names."""
    auto_discover_datamodules()
    return sorted(set(DATAMODULE_REGISTRY.keys()))
