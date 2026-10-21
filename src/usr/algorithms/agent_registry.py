"""
Algorithm / Agent Registry — Single source of truth for algorithm class lookup.

Algorithms self-register using the @register_algorithm (or @register_agent) decorator:
    @register_algorithm("custom_algo")
    class CustomAlgorithm(OfflineAgentBase):
        ...

The pipeline resolves algorithms via get_algorithm_class() / get_agent_class():
    AgentClass = get_algorithm_class("custom_algo_tuned")
    # → Returns CustomAlgorithm (longest prefix match)
"""

import importlib
import os
import pkgutil
from typing import Any

AGENT_REGISTRY: dict[str, Any] = {}


def register_agent(*prefixes):
    """Decorator to register an algorithm class under one or more prefixes.

    Usage:
        @register_algorithm("custom_algo")
        class CustomAlgorithm(OfflineAgentBase):
            ...
    """

    def decorator(cls):
        for prefix in prefixes:
            AGENT_REGISTRY[prefix] = cls
        return cls

    return decorator


def get_agent_class(algo_name: str):
    """Resolve an algorithm name to its registered agent class using longest-prefix matching.

    Args:
        algo_name: Algorithm identifier (e.g., "iql_cp_tuned", "offline_custom_variant").

    Returns:
        The registered agent class.

    Raises:
        ValueError: If no registered prefix matches the algorithm name.
    """
    if not AGENT_REGISTRY:
        auto_discover()

    # Try exact match first
    if algo_name in AGENT_REGISTRY:
        return AGENT_REGISTRY[algo_name]

    # Longest-prefix match: sort by key length descending, check startswith
    matches = [
        (prefix, cls)
        for prefix, cls in AGENT_REGISTRY.items()
        if algo_name == prefix or algo_name.startswith(prefix + "_")
    ]

    if not matches:
        registered = sorted(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent algorithm: '{algo_name}'. Registered prefixes: {registered}")

    # Return the class with the longest matching prefix
    return max(matches, key=lambda x: len(x[0]))[1]


def auto_discover():
    """Import modules in src/usr/algorithms/ to trigger @register_agent decorators.

    Called once at startup (e.g., in train.py) to ensure all agents are registered
    before get_agent_class() is used.
    """
    import sys

    algorithms_dir = os.path.dirname(os.path.abspath(__file__))
    usr_dir = os.path.dirname(algorithms_dir)
    src_dir = os.path.dirname(usr_dir)
    project_root = os.path.dirname(src_dir)
    for p in (project_root, src_dir, usr_dir, algorithms_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Discover in src/usr/algorithms
    for module_info in pkgutil.iter_modules([algorithms_dir]):
        if module_info.name.startswith("_") or module_info.name in (
            "agent_registry",
            "base_agent",
        ):
            continue
        try:
            importlib.import_module(f"src.usr.algorithms.{module_info.name}")
        except Exception as e:
            print(f"[Warning] Failed to auto-discover algorithm module '{module_info.name}': {e}")


def list_registered_agents():
    """Return sorted list of registered algorithm prefixes."""
    auto_discover()
    agents = set(AGENT_REGISTRY.keys())
    # Discover configured agent algorithms from in/config/agent/
    algorithms_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(algorithms_dir)))
    agent_cfg_dir = os.path.join(project_root, "in", "config", "agent")
    if os.path.isdir(agent_cfg_dir):
        for root, _, files in os.walk(agent_cfg_dir):
            for f in files:
                if (f.endswith(".yaml") or f.endswith(".yml")) and not f.startswith("_"):
                    stem = f.rsplit(".", 1)[0]
                    agents.add(stem)
    return sorted(agents)


# Semantic aliases aligning Algorithm terminology
register_algorithm = register_agent
get_algorithm_class = get_agent_class
list_registered_algorithms = list_registered_agents
ALGORITHM_REGISTRY = AGENT_REGISTRY
