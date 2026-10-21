"""Backend component hot-reloading service for Theta-IDE.

Allows researchers to modify model layers, loss functions, environment step logic,
or data modules on disk and reload them live into the running process without
restarting Theta-IDE.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import List, Optional

log = logging.getLogger(__name__)


def reload_component_modules(target_prefix: str | None = None) -> List[str]:
    """Reload active research modules in sys.modules and refresh component registries.

    Args:
        target_prefix: Optional module name prefix to reload (e.g. 'src.usr.models.custom_model').
            If None, reloads all active user research modules ('src.usr.' and 'in.envs.').

    Returns:
        List of successfully reloaded module names.
    """
    prefixes = (target_prefix,) if target_prefix else ("src.usr.", "in.envs.")
    candidates = [
        name for name, mod in list(sys.modules.items()) if mod is not None and any(name.startswith(p) for p in prefixes)
    ]

    reloaded: List[str] = []
    # Sort modules so parent packages reload before or with children
    candidates.sort(key=lambda s: s.count("."))

    for mod_name in candidates:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
            reloaded.append(mod_name)
            log.debug("Hot-reloaded module: %s", mod_name)
        except Exception as exc:
            log.warning("Could not hot-reload module '%s': %s", mod_name, exc)

    # Refresh entry points and dynamic registries
    try:
        from src.app.core.plugin_discovery import load_all_plugins

        load_all_plugins(force_reload=True)
    except Exception as exc:
        log.warning("Error refreshing entry points during hot-reload: %s", exc)

    return reloaded
