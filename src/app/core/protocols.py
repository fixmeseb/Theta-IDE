"""Generic component and model protocols for Theta-IDE.

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
    """Protocol for models with mutable or evolvable parameter topologies.

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
    """Protocol for models that save non-parameter state alongside checkpoint weights."""

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
    """Protocol for plugins that require environment wrappers or observation adapters."""

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

    def configure_optimizer_groups(self, base_lr: float, weight_decay: float, cfg: Any = None) -> list[dict[str, Any]]:
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


@runtime_checkable
class DataModulePluginProtocol(Protocol):
    """Protocol for pluggable DataModules providing PyTorch datasets and DataLoaders."""

    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets and splits."""
        ...

    def train_dataloader(self) -> Any:
        """Return PyTorch DataLoader for training batches."""
        ...

    def val_dataloader(self) -> Any:
        """Return PyTorch DataLoader for validation batches."""
        ...


@runtime_checkable
class TrajectorySamplerProtocol(Protocol):
    """Protocol for trajectory sequence and window samplers (e.g. Decision Transformers, RNNs)."""

    def sample_trajectory_window(self, batch_size: int, context_len: int) -> dict[str, torch.Tensor]:
        """Return a batch of variable- or fixed-length trajectory slices."""
        ...


@runtime_checkable
class EnvironmentPluginProtocol(Protocol):
    """Protocol for domain environment providers and custom Gym environments."""

    @classmethod
    def make_env(cls, env_id: str, **kwargs: Any) -> Any:
        """Instantiate and return the Gymnasium environment."""
        ...

    @classmethod
    def get_default_wrappers(cls) -> list[Callable[[Any], Any]]:
        """Return list of default wrapper callables to wrap the environment."""
        ...

    @classmethod
    def get_reward_shaping_hook(cls) -> Callable[..., Any] | None:
        """Return optional reward shaping callable."""
        ...

    @classmethod
    def get_logic_valuation_hook(cls) -> Callable[..., Any] | None:
        """Return optional logic valuation callable."""
        ...


@runtime_checkable
class AgentPluginProtocol(Protocol):
    """Protocol for RL agent training algorithms extending PyTorch Lightning."""

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Execute one training step and return loss dictionary."""
        ...

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """Execute one validation step."""
        ...

    def configure_optimizers(self) -> Any:
        """Configure and return optimizers."""
        ...


@runtime_checkable
class EvaluationHookProtocol(Protocol):
    """Protocol for custom evaluation metrics (formal verification, safety bounds, saliency, OOD)."""

    def on_eval_start(self, agent: Any, env: Any, stage: str = "val") -> None:
        """Called before evaluation rollouts begin."""
        ...

    def evaluate_step(self, agent: Any, batch_or_transition: Any) -> dict[str, float]:
        """Return step-level evaluation metrics."""
        ...

    def on_eval_end(self) -> dict[str, float]:
        """Return aggregated epoch-level evaluation metrics."""
        ...


class StandardModelActorCriticWrapper(nn.Module):
    """Adapter that gives a standard PyTorch model an actor-critic interface."""

    def __init__(self, backbone: nn.Module, obs_dim: int = 4, n_actions: int = 2):
        super().__init__()
        self.backbone = backbone
        self.n_actions = n_actions
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.backbone(obs, **kwargs)

    def get_value(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if hasattr(self.backbone, "get_value"):
            return self.backbone.get_value(obs, **kwargs)
        return self.critic(obs.float().reshape(obs.shape[0], -1))

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None) -> Any:
        from torch.distributions.categorical import Categorical

        from src.app.core.types import ActionResult

        if hasattr(self.backbone, "get_action_and_value"):
            return self.backbone.get_action_and_value(obs, action=action)
        logits = self.backbone(obs.float().reshape(obs.shape[0], -1))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.get_value(obs)
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=value,
        )

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "get_action_probs"):
            return self.backbone.get_action_probs(obs)
        logits = self.backbone(obs.float().reshape(obs.shape[0], -1))
        return torch.softmax(logits, dim=-1)
