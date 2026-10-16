"""Component and Model Protocols for NeSyRL.

Defines protocol interfaces and discovery helpers for extensible models, agents,
and environments. Models implement these protocols to declare runtime capabilities
(such as dynamic topology evolution, custom trainer callbacks, and checkpoint state)
without hardcoding class checks in base agent loops.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

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


@dataclass
class ModelSlot:
    """Defines a slot for a submodel inside a composite architecture.

    Attributes:
        name: Unique identifier for the slot (e.g. 'neural', 'symbolic', 'encoder').
        required: Whether this slot must be filled for the model to instantiate.
        default: Default model identifier to populate if none is provided.
        tags: List of semantic tags to filter candidate models in the GUI/registry.
        description: User-facing explanation of the subcomponent's role.
    """

    name: str
    required: bool = True
    default: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""


@runtime_checkable
class CompositeModelProtocol(Protocol):
    """Protocol for composite models that assemble subcomponents/submodels."""

    @classmethod
    def get_model_slots(cls) -> list[ModelSlot]:
        """Declare submodel slots to the GUI and configuration pipeline."""
        ...

    @classmethod
    def build_from_config(
        cls,
        cfg: dict[str, Any],
        env: Any,
        device: torch.device,
        submodels: dict[str, nn.Module],
        **kwargs: Any,
    ) -> nn.Module:
        """Plugin-owned constructor receiving instantiated child modules."""
        ...


@runtime_checkable
class EnvironmentWrapperProtocol(Protocol):
    """Protocol for plugins requiring specialized environment wrappers (e.g. logic predicates)."""

    @classmethod
    def get_env_wrappers(cls, env: Any, cfg: dict[str, Any]) -> list[Any]:
        """Return wrapper classes or callables to apply to the environment upon initialization."""
        ...


@runtime_checkable
class AuxiliaryLossProtocol(Protocol):
    """Protocol for models that produce auxiliary losses during training."""

    def compute_auxiliary_losses(self, batch: Any, output: Any) -> dict[str, torch.Tensor]:
        """Return dictionary of named auxiliary losses (e.g. {'aux_loss': tensor})."""
        ...


@runtime_checkable
class OptimizerGroupsProtocol(Protocol):
    """Protocol for models requiring custom PyTorch optimizer parameter groups."""

    def configure_optimizer_groups(
        self, base_lr: float, weight_decay: float, cfg: Any = None
    ) -> list[dict[str, Any]]:
        """Return custom parameter group dictionaries with individual learning rates."""
        ...


def walk_model_modules(root: Any) -> list[Any]:
    """Recursively collect a model and all constituent sub-models.

    Supports:
      - Standalone models (returns [root])
      - Composite models declaring policy_modules or CompositeModelProtocol
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

        # Check for explicit policy_modules
        policy_modules = getattr(obj, "policy_modules", None)
        if policy_modules is not None and isinstance(policy_modules, (list, tuple)):
            for sub_m in policy_modules:
                _traverse(sub_m)

        # Check subcomponents if object implements CompositeModelProtocol
        if hasattr(obj, "get_subcomponents") and callable(getattr(obj, "get_subcomponents")):
            try:
                for sub_m in obj.get_subcomponents().values():
                    _traverse(sub_m)
            except Exception:
                pass

        # Check for target models or constituent actor/critic networks
        for attr in (
            "model",
            "target_model",
            "q_model",
            "q_network",
            "target_q_model",
            "target_q_network",
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
