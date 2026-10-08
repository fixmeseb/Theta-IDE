"""Component and Model Protocols for NeSyRL.

Defines protocol interfaces and discovery helpers for extensible models, agents,
and environments. Models implement these protocols to declare runtime capabilities
(such as dynamic topology evolution, custom trainer callbacks, and checkpoint state)
without hardcoding class checks in base agent loops.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class DynamicTopologyProtocol(Protocol):
    """Protocol for models with mutable parameter topologies (e.g. neuro-fuzzy or neuro-evolutionary models).

    When a model reorganizes its architecture (adding/removing parameters or changing shapes),
    it signals this change so that the agent can rebind optimizers and synchronize target networks.
    """

    def has_topology_changed(self) -> bool:
        """Return True if the model's architecture or parameter set has changed since last reset."""
        ...

    def clone_topology_to(self, target: nn.Module) -> None:
        """Synchronize the target network's architecture and weights to match this model."""
        ...


@runtime_checkable
class ExtraStateProtocol(Protocol):
    """Protocol for models that require saving non-parameter state into checkpoints (e.g. fuzzy rules)."""

    def extra_state(self) -> dict[str, Any]:
        """Return a dictionary of state to be saved in checkpoints."""
        ...

    def load_extra_state(self, state: dict[str, Any]) -> None:
        """Restore non-parameter state from checkpoint dict."""
        ...


@runtime_checkable
class HasModelCallbacks(Protocol):
    """Protocol for models that provide their own PyTorch Lightning callbacks."""

    def get_callbacks(self) -> list[Any]:
        """Return a list of Lightning Callback instances required by this model."""
        ...


def walk_model_modules(root: Any) -> list[Any]:
    """Recursively collect a model and all constituent sub-models.

    Supports:
      - Standalone models (returns [root])
      - Composite models (e.g. BlendRL BlenderActorCritic via `policy_modules`)
      - Standard torch.nn.Module hierarchies
    """
    if root is None:
        return []

    visited = set()
    modules = []

    def _traverse(obj: Any):
        if obj is None or id(obj) in visited or len(visited) > 500:
            return
        visited.add(id(obj))
        modules.append(obj)

        # Check for explicit policy_modules (e.g. BlendRL composite model)
        policy_modules = getattr(obj, "policy_modules", None)
        if policy_modules is not None and isinstance(policy_modules, (list, tuple)):
            for sub_m in policy_modules:
                _traverse(sub_m)

        # Check for target models or constituent actor/critic networks
        for attr in (
            "model",
            "target_model",
            "q_model",
            "q_network",
            "target_q_model",
            "target_q_network",
            "blender",
            "actor",
            "critic",
        ):
            # Guard against unconstrained mock objects synthesizing attributes
            if hasattr(obj, "_mock_return_value"):
                if attr not in getattr(obj, "__dict__", {}):
                    continue
            sub_m = getattr(obj, attr, None)
            if sub_m is not None:
                has_pol = hasattr(sub_m, "policy_modules") and isinstance(
                    getattr(sub_m, "policy_modules", None), (list, tuple)
                )
                if isinstance(sub_m, nn.Module) or has_pol:
                    _traverse(sub_m)

    _traverse(root)
    return modules
