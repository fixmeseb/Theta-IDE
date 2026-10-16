# Theta-IDE Component Plugin Authoring Guide

This guide describes how to author, register, package, and distribute research components and plugins for **Theta-IDE**. It details the protocols that allow complex model architectures, reinforcement learning algorithms, custom environments, trajectory samplers, safety evaluators, and complete experiments to integrate seamlessly with the IDE, execution engine, and Community Hub without modifying core code.

---

## 1. Overview & Architectural Principles

In Theta-IDE, all components are cleanly decoupled through declarative runtime protocols:
- **Models & Composite Architectures**: Define policy networks, Q-functions, feature extractors, and symbolic reasoners.
- **Agents & Algorithms**: Implement RL or supervised loss functions, update steps, and optimization procedures.
- **Environments & Groundings**: Define simulators, observation wrappers, potential-based reward shaping, and logic valuations.
- **DataModules & Samplers**: Manage trajectory sequence windows, offline datasets, and replay buffer sampling.
- **Evaluators & Verification**: Compute formal verification bounds, safety metrics, and policy attributions.
- **Experiment Recipes**: Bundle complete reproducible benchmarks (configs, envs, models, rules, weights) for 1-click sharing.

```
                    ┌──────────────────────────────────────────────┐
                    │               Theta-IDE Core                 │
                    │   (Pipeline, GUI Viewer, Slurm, Lightning)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        Discovers via standard protocols & entry points
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   [Model Plugins]                 [Agent Plugins]                 [Env & Data Plugins]
• CompositeModelProtocol        • AgentPluginProtocol           • EnvironmentPluginProtocol
• AuxiliaryLossProtocol         • Dynamic loss steps            • DataModulePluginProtocol
• OptimizerGroupsProtocol       • Paradigm validation           • TrajectorySamplerProtocol
```

---

## 2. Component Plugin Archetypes & Protocols Reference

All plugin contracts are defined in [`src/app/core/protocols.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/protocols.py).

### 2.1 Model Architectures & Composite Models

#### `ModelSlot` & `CompositeModelProtocol`
Composite models assemble multiple submodules (e.g. an encoder and a decoder, or a neural policy and a symbolic reasoner).

```python
from dataclasses import dataclass, field
from src.app.core.protocols import ModelSlot, CompositeModelProtocol

class MyCompositePolicy(nn.Module):
    @classmethod
    def get_model_slots(cls) -> list[ModelSlot]:
        return [
            ModelSlot(
                name="neural",
                required=True,
                default="dnn",
                tags=["neural", "actor"],
                description="Neural feature extractor & policy head",
            ),
            ModelSlot(
                name="symbolic",
                required=False,
                default="nsfr",
                tags=["symbolic", "logic"],
                description="Symbolic logic reasoning module",
            ),
        ]
```

When implemented:
1. **Pipeline Resolver (`src/app/pipeline/config.py`)**: Automatically resolves each declared slot against Tier 1 defaults (`in/config/model/<candidate>.yaml`) and Tier 2 overrides.
2. **GUI ConfigViewer (`frontend/config_viewer.py`)**: Automatically renders dynamic slot cards, candidate dropdown selectors, and subcomponent parameter editors.

#### `AuxiliaryLossProtocol`
Models can declare auxiliary objectives (e.g., entropy regularization over blending weights, sparsity constraints, or representation consistency):

```python
class AuxiliaryLossProtocol(Protocol):
    def compute_auxiliary_losses(self, batch: Any, output: Any) -> dict[str, torch.Tensor]:
        """Return dictionary of named auxiliary losses added to the algorithm loss."""
        ...
```

#### `OptimizerGroupsProtocol`
Models configure custom parameter groups with per-module learning rates (e.g., slower learning rate for symbolic logic rules, faster for neural heads):

```python
class OptimizerGroupsProtocol(Protocol):
    def configure_optimizer_groups(self, base_lr: float, weight_decay: float, cfg: Any = None) -> list[dict[str, Any]]:
        """Return parameter group dictionaries for the PyTorch optimizer."""
        ...
```

#### `EnvironmentWrapperProtocol`
Models declare required observation wrappers or predicate extractors needed by the simulator:

```python
class EnvironmentWrapperProtocol(Protocol):
    @classmethod
    def get_env_wrappers(cls, env: Any, cfg: dict[str, Any]) -> list[Any]:
        """Return Gymnasium wrappers to wrap the environment upon initialization."""
        ...
```

---

### 2.2 Agent & RL Algorithm Plugins

Agents implement RL algorithms (e.g., SAC, TD3, AWAC, Offline Multi-Agent) as PyTorch Lightning Modules:

```python
from src.usr.methods.agent_registry import register_agent
from src.app.core.protocols import AgentPluginProtocol

@register_agent("my_algorithm", "my_algo")
class MyCustomAgent(L.LightningModule):
    """Custom reinforcement learning agent."""

    def training_step(self, batch, batch_idx):
        loss_dict = self.compute_algorithm_loss(batch)
        # Auxiliary losses from models are automatically collected and added
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        # Queries model.configure_optimizer_groups() if implemented
        ...
```

---

### 2.3 Environment & Grounding Plugins

Environment plugins package Gymnasium simulators, vectorization hooks, domain logic rules, and potential-based reward shaping:

```python
from src.app.core.protocols import EnvironmentPluginProtocol

class RoboticsSimulatorPlugin:
    @classmethod
    def make_env(cls, env_id: str, **kwargs):
        import gymnasium as gym
        return gym.make(f"Robotics/{env_id}", **kwargs)

    @classmethod
    def get_default_wrappers(cls):
        return [NormalizeObservationWrapper, StateExtractionWrapper]

    @classmethod
    def get_reward_shaping_hook(cls):
        return potential_based_shaping_function

    @classmethod
    def get_logic_valuation_hook(cls):
        return predicate_valuation_function
```

---

### 2.4 DataModule & Trajectory Sampler Plugins

DataModules manage offline datasets, clinical cohorts, and sequence window sampling:

```python
from src.app.core.paradigm_loader import register_component
from src.app.core.protocols import DataModulePluginProtocol, TrajectorySamplerProtocol

@register_component("DecisionTransformerDataModule", "TrajectoryDataModule")
class TrajectorySequenceDataModule:
    def setup(self, stage=None):
        self.dataset = load_trajectory_npz(...)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=64, shuffle=True)

    def sample_trajectory_window(self, batch_size: int, context_len: int) -> dict[str, torch.Tensor]:
        """Returns sequence batches (s_t, a_t, r_t, s_{t+1}) of length context_len."""
        return self.dataset.sample_slices(batch_size, context_len)
```

---

### 2.5 Evaluator, Safety & Verification Plugins

Evaluator plugins log formal safety bounds, barrier certificates, and explainability attributions without altering the training loop:

```python
from src.app.core.protocols import EvaluationHookProtocol

class SafetyVerificationEvaluator:
    def on_eval_start(self, agent, env, stage="val"):
        self.violations = 0

    def evaluate_step(self, agent, batch_or_transition) -> dict[str, float]:
        obs = batch_or_transition["obs"]
        action = agent.act(obs)
        is_safe = self.check_barrier_certificate(obs, action)
        return {"eval/barrier_safety": float(is_safe)}

    def on_eval_end(self) -> dict[str, float]:
        return {"eval/safety_violation_rate": self.violations / self.total_steps}
```

All metrics returned by evaluation hooks automatically log into `metrics.csv` and render in the Theta-IDE Results Browser.

---

## 3. Universal Python Packaging via `pyproject.toml` Entry Points

Any component can be packaged as a standard Python library and published to PyPI (`pip install my-theta-plugin`). Theta-IDE discovers and registers them automatically at startup via [`src/app/core/plugin_discovery.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/plugin_discovery.py).

Declare standard entry-point groups in your `pyproject.toml`:

```toml
[project]
name = "my-theta-plugin"
version = "1.0.0"
dependencies = [
    "torch>=2.0.0",
    "gymnasium>=0.29.0",
]

[project.entry-points."theta.models"]
my_composite_model = "my_plugin.models:MyCompositePolicy"

[project.entry-points."theta.agents"]
my_rl_algorithm = "my_plugin.agents:MyCustomAgent"

[project.entry-points."theta.environments"]
my_custom_env = "my_plugin.env:RoboticsSimulatorPlugin"

[project.entry-points."theta.datamodules"]
my_trajectory_dm = "my_plugin.data:TrajectorySequenceDataModule"

[project.entry-points."theta.evaluators"]
safety_verifier = "my_plugin.eval:SafetyVerificationEvaluator"

[project.entry-points."theta.plugins"]
custom_gui_panel = "my_plugin.ui:MyCustomPanelPlugin"
```

Once installed, your components appear immediately in:
- The Config Viewer dropdown selectors in the GUI.
- The pipeline execution engine (`python run_pipeline.py <group>/<exp>`).
- Built-in registries (`MODEL_REGISTRY`, `AGENT_REGISTRY`, `_COMPONENT_REGISTRY`).

---

## 4. 1-Click Experiment-to-Plugin Exporter

Researchers can package any experiment into a self-contained, shareable, installable Theta Hub component archive (`.zip`) with one click.

### 4.1 Exporting from the GUI

1. Open Theta-IDE and select any experiment recipe in the Config Tree.
2. Click the **Export** button on the Action Bar (or press `Ctrl+Shift+S`):
   - **Theta Hub Component Archive (`.zip`)** (Recommended): Packages the recipe YAML, base templates, referenced environment definitions, valuation rules, custom model architectures, and optional checkpoint weights.
   - **Standalone Recipe YAML (`.yaml`)**: Exports only the composed recipe YAML for manual reproduction.
3. Configure packaging options (include rules, models, weights) and click **Export**.

### 4.2 Exporting from the CLI

Export any experiment directly from the command line:

```bash
# Export experiment with all dependencies
python run_pipeline.py export cartpole/quick_test

# Include trained model checkpoints & specify output archive
python run_pipeline.py export cartpole/quick_test --include-weights --out results/exports/cartpole_release.zip

# Or via Hydra override during execution
python run_pipeline.py cartpole/quick_test export=true
```

### 4.3 What Gets Packaged

The [`ExperimentExporter`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/services/experiment_exporter.py) automatically resolves the full dependency closure:
```
cartpole_quick_test_v1.0.0.zip
├── .theta_component.json                      # Hub component manifest
├── README.md                                  # Auto-generated reproduction guide
├── in/config/experiment/cartpole/
│   ├── quick_test.yaml                        # Experiment recipe
│   └── _base.yaml                             # Group base template
├── in/config/env/cartpole.yaml                # Environment configuration
├── in/envs/cartpole/                          # Custom environment code & hooks
├── in/rules/cartpole/                         # Domain logic rules
├── src/usr/models/blendrl/                    # Referenced model architecture
└── results/checkpoints/cartpole/quick_test/   # Checkpoint weights (optional)
```

### 4.4 1-Click Roundtrip Installation via Theta Hub

Any researcher receiving the `.zip` package can install it into their Theta-IDE in one click:
- Drag and drop or import the `.zip` in **Components &rarr; Hub &rarr; Install Local Package**.
- The [`HubInstaller`](file:///Users/cameronegbert/Research/Theta-IDE/frontend/hub/installer.py) automatically unpacks configs to `in/config/experiment/`, environments to `in/envs/`, and rules to `in/rules/`.
- The experiment is immediately runnable with:
  ```bash
  python run_pipeline.py cartpole/quick_test
  ```

---

## 5. Reference Case Study: BlendRL as a Component Plugin

**BlendRL** is the reference implementation of a composite, hybrid neural-symbolic component plugin located in `src/usr/models/blendrl/`.

### 5.1 Plugin Manifest (`.theta_component.json`)

```json
{
  "id": "blendrl",
  "name": "BlendRL",
  "kind": "model",
  "version": "1.0.0",
  "description": "BlendRL hybrid neural-symbolic policy architecture",
  "author": {"name": "Theta Research"},
  "target_path": "src/usr/models/blendrl",
  "compatibility": {"theta": ">=2.0.0"}
}
```

### 5.2 Implementation Highlights

```python
from src.app.core.model_registry import register_model
from src.app.core.protocols import ModelSlot

@register_model("blendrl", "blender")
class BlenderActorCritic(nn.Module):
    """Hybrid Actor-Critic combining neural and symbolic policy heads."""

    @classmethod
    def get_model_slots(cls) -> list[ModelSlot]:
        return [
            ModelSlot(
                name="neural",
                description="Neural feature extractor / policy head",
                default="dnn",
                tags=["neural", "actor"],
            ),
            ModelSlot(
                name="symbolic",
                description="Symbolic logic reasoning module",
                default="nsfr",
                tags=["symbolic", "logic"],
            ),
            ModelSlot(
                name="blender",
                description="Logic-neural policy blender",
                default="neural",
                tags=["blender"],
            ),
        ]

    def compute_auxiliary_losses(self, batch, output) -> dict[str, torch.Tensor]:
        losses = {}
        be = getattr(output, "blend_entropy", None)
        if isinstance(be, torch.Tensor):
            losses["blend_entropy"] = be.mean()
        return losses

    def configure_optimizer_groups(self, base_lr: float, weight_decay: float = 0.0, cfg=None):
        groups = []
        logic_lr = float(self.get_cfg("logic_lr", base_lr))
        blender_lr = float(self.get_cfg("blender_lr", base_lr))
        if hasattr(self, "actor") and hasattr(self.actor, "policy_modules"):
            for m, m_type in zip(self.actor.policy_modules, self.module_types):
                lr = base_lr if m_type == "neural" else logic_lr
                groups.append({"params": m.parameters(), "lr": lr, "weight_decay": weight_decay})
        return groups
```

---

## 6. Authoring Checklist

1. **Choose your component archetype**: Model, Agent, Environment, DataModule, Evaluator, or Experiment.
2. **Implement the matching runtime protocol** in [`src/app/core/protocols.py`](file:///Users/cameronegbert/Research/Theta-IDE/src/app/core/protocols.py).
3. **Register your component**:
   - Models: `@register_model("<name>")`
   - Agents: `@register_agent("<name>")`
   - DataModules: `@register_component("<Name>")`
4. **Publishing options**:
   - **Local / Hub package**: Include `.theta_component.json` and export via `python run_pipeline.py export`.
   - **PyPI Python package**: Declare entry points under `[project.entry-points."theta.*"]` in `pyproject.toml`.
5. **Verify with test suite**: Run `pytest tests/test_plugin_discovery.py` to verify protocol conformance.
