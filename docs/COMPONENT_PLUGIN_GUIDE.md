# Theta-IDE Component Plugin Authoring Guide

This guide describes how to author, register, and distribute model architecture plugins for **Theta-IDE**. It details the protocols that allow complex, composite, and hybrid model architectures to integrate seamlessly with the IDE, pipeline, algorithms, and configuration viewer without modifying core code.

---

## 1. Overview & Architectural Principles

In Theta-IDE, learning algorithms (such as PPO, CQL, and IQL) are decoupled from model architectures:
- **Agents / Algorithms** manage the reinforcement learning or supervised loss functions, replay buffers, and environment interactions.
- **Models** define policy neural networks, Q-networks, feature extractors, and reasoners.
- **Plugins** introduce new model architectures, subcomponents, auxiliary losses, optimizer parameter groups, and environment wrappers through standard Python runtime protocols.

A user who installs Theta-IDE without specialized plugins can run standard RL algorithms with neural models (such as `NatureCNN`, `StandardMLP`, `DuelingResNet`, or `Transformer`). When a component plugin (such as `BlendRL` or `CEW`) is installed, it provides its metadata, hierarchical parameter slots, auxiliary losses, and custom optimizer configurations declaratively.

```
                    ┌──────────────────────────────┐
                    │      Theta-IDE Core          │
                    │ (PPO, CQL, IQL, Train Loop)  │
                    └──────────────┬───────────────┘
                                   │
              implements runtime protocols via duck-typing
                                   │
                    ┌──────────────▼───────────────┐
                    │      Component Plugin        │
                    │  (e.g., BlendRL Model Arch)  │
                    └──────────────┬───────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
[ModelSlot / Composite]   [AuxiliaryLossProtocol]   [OptimizerGroupsProtocol]
Declares submodel slots   Dynamic loss terms &      Custom param groups &
for GUI & Pipeline        auxiliary metric logs     per-module learning rates
```

---

## 2. Core Protocols Reference

All plugin protocols are defined in [`src/app/core/protocols.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/protocols.py).

### 2.1 ModelSlot & CompositeModelProtocol

A **composite model** is composed of multiple submodules (e.g. an encoder and a decoder, or a neural policy and a symbolic reasoner).

```python
@dataclass(frozen=True)
class ModelSlot:
    name: str                           # Slot key, e.g. "neural", "symbolic", "encoder"
    description: str = ""               # Human-readable description shown in GUI
    candidate_types: tuple[str, ...] = ()  # Recommended candidate models (empty = all models)
    default_type: str | None = None     # Default model identifier
    required: bool = True               # Whether this slot is required
```

To declare composite subcomponents, implement `CompositeModelProtocol`:

```python
class CompositeModelProtocol(Protocol):
    @classmethod
    def get_model_slots(cls) -> tuple[ModelSlot, ...]:
        """Return the submodel slots required or supported by this model architecture."""
        ...
```

When implemented:
1. **Pipeline Resolver (`src/app/pipeline/config.py`)**: Automatically resolves each declared slot against Tier 1 defaults (`in/config/model/<candidate>.yaml`) and Tier 2 overrides.
2. **GUI ConfigViewer (`frontend/config_viewer.py`)**: Automatically renders dynamic slot cards, candidate dropdown selectors, and subcomponent parameter adders.

---

### 2.2 AuxiliaryLossProtocol

Certain model architectures require auxiliary objectives during training (e.g., entropy regularization over blending weights, sparsity constraints, or representation consistency):

```python
class AuxiliaryLossProtocol(Protocol):
    def compute_auxiliary_losses(
        self,
        batch: Any,
        policy_output: Any,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return auxiliary loss terms added to the total algorithm loss.
        
        Returns:
            Dictionary of loss name -> scalar tensor.
            Example: {"blend_entropy": 0.05 * loss}
        """
        ...
```

Any registered agent (e.g., `PPOAgent`, `CQLAgent`) calls `compute_auxiliary_losses` during each training step and logs all returned auxiliary terms to CSV metrics automatically.

---

### 2.3 OptimizerGroupsProtocol

Complex models often require different learning rates or weight decays for different subcomponents (e.g., slower learning rate for symbolic logic rules, faster for neural heads):

```python
class OptimizerGroupsProtocol(Protocol):
    def configure_optimizer_groups(self, default_lr: float) -> list[dict[str, Any]]:
        """Return parameter groups for the PyTorch optimizer.
        
        Returns:
            List of param group dictionaries, e.g.:
            [
                {"params": self.neural.parameters(), "lr": default_lr},
                {"params": self.logic.parameters(), "lr": self.logic_lr},
            ]
        """
        ...
```

---

### 2.4 EnvironmentWrapperProtocol

If a model architecture requires specific observation formats, symbolic state extraction, or frame wrapping, it can declare required environment wrappers:

```python
class EnvironmentWrapperProtocol(Protocol):
    @classmethod
    def get_required_env_wrappers(cls, env_name: str) -> list[Callable[[Any], Any]]:
        """Return wrapper callables to wrap Gymnasium environments on instantiation."""
        ...
```

---

## 3. Reference Case Study: BlendRL as a Component Plugin

**BlendRL** is a reference implementation of a composite, hybrid neural-symbolic component plugin located in `src/usr/models/blendrl/`.

### 3.1 Plugin Manifest (`.theta_component.json`)

```json
{
  "name": "blendrl",
  "version": "1.0.0",
  "component_type": "model",
  "description": "BlendRL hybrid neural-symbolic policy architecture",
  "author": "Theta Research",
  "entry_point": "src.usr.models.blendrl.agents.blender_agent:build_blender_module",
  "protocols": [
    "CompositeModelProtocol",
    "AuxiliaryLossProtocol",
    "OptimizerGroupsProtocol"
  ]
}
```

### 3.2 Registration

In `src/usr/models/blendrl/agents/blender_agent.py`:

```python
from src.app.core.model_registry import register_model
from src.app.core.protocols import ModelSlot

@register_model("blendrl", "blender")
class BlenderActorCritic(nn.Module):
    """Hybrid Actor-Critic combining neural and symbolic policy heads."""

    @classmethod
    def get_model_slots(cls) -> tuple[ModelSlot, ...]:
        return (
            ModelSlot(
                name="neural",
                description="Neural feature extractor / policy head",
                candidate_types=(),  # Accepts any non-composite neural model
                default_type="dnn",
            ),
            ModelSlot(
                name="symbolic",
                description="Symbolic logic reasoning module",
                candidate_types=("nsfr", "neumann", "cew"),
                default_type="nsfr",
            ),
            ModelSlot(
                name="blender",
                description="Logic-neural policy blender",
                candidate_types=("neural", "logic", "hybrid"),
                default_type="neural",
            ),
        )
```

### 3.3 Factory Builder

```python
def build_blender_module(cfg, env=None, action_dim=None, device="cpu"):
    """Instantiate BlenderActorCritic from hierarchical Hydra config."""
    # Resolves neural and symbolic subcomponents via build_model
    ...
```

### 3.4 Custom Auxiliary Losses & Optimizer Parameter Groups

```python
    def compute_auxiliary_losses(self, batch, policy_output, step=0):
        losses = {}
        if self.blend_ent_coef > 0.0 and hasattr(policy_output, "get_aux"):
            blend_weights = policy_output.get_aux("blend_weights")
            if blend_weights is not None:
                # Maximize entropy of blender gating weights
                p = blend_weights.clamp(min=1e-8)
                ent = -(p * torch.log(p)).sum(dim=-1).mean()
                losses["blend_entropy"] = -self.blend_ent_coef * ent
        return losses

    def configure_optimizer_groups(self, default_lr: float):
        groups = []
        if hasattr(self, "neural_agent") and self.neural_agent is not None:
            groups.append({"params": self.neural_agent.parameters(), "lr": default_lr})
        if hasattr(self, "logic_agent") and self.logic_agent is not None:
            groups.append({"params": self.logic_agent.parameters(), "lr": self.logic_lr})
        if hasattr(self, "blender") and self.blender is not None:
            groups.append({"params": self.blender.parameters(), "lr": self.blender_lr})
        return groups
```

---

## 4. Configuring Composite Models in YAML

Theta-IDE's 3-tier configuration natively supports composite models:

### Tier 1: Base Profile (`in/config/model/blendrl.yaml`)

```yaml
actor_mode: hybrid
blender:
  mode: neural
  blend_function: softmax
blend_q_values: true
blend_ent_coef: 0.05
logic_lr: 0.001
blender_lr: 0.0005

neural:
  type: dnn
  hidden_sizes: [256, 256]

symbolic:
  type: nsfr
  rules: default
```

### Tier 2: Universal Hyperparameters (`params.model.blendrl`)

```yaml
methods:
  params:
    gamma: 0.99
    model:
      blendrl:
        blend_ent_coef: 0.02
```

### Tier 3: Experiment Method

```yaml
methods:
  cql_blendrl_cew_resnet:
    agent: cql
    model:
      blendrl:
        neural: dueling_resnet
        symbolic:
          cew:
            ecm_dthr: 0.03
```

---

## 5. Summary Checklist for Plugin Authors

1. **Implement `@register_model("<name>")`** on your model class or factory function.
2. If your model composes submodules, implement `get_model_slots()` returning `tuple[ModelSlot, ...]`.
3. If your model requires extra losses, implement `compute_auxiliary_losses(batch, policy_output, step)`.
4. If your model requires separate learning rates, implement `configure_optimizer_groups(default_lr)`.
5. Provide a base YAML profile in `in/config/model/<name>.yaml`.
6. Add unit tests verifying `build_model(cfg, env=env)` and protocol compliance.
