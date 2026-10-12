# Antigravity Context: BlendRL (Refactored)

This document is the foundational source of truth for Antigravity in this workspace. It defines the architecture, data standards, and critical logic rules that must be followed.

[TOC]
- [1. Project Overview](#1-project-overview)
- [2. Learning Paradigms](#2-learning-paradigms)
- [3. Execution Pipeline & Configuration](#3-execution-pipeline--configuration)
- [4. Data Saving & Dataset Schema](#4-data-saving--dataset-schema)
- [5. Operational Environment](#5-operational-environment)
- [6. Hydra & Debugging Heuristics](#6-hydra--debugging-heuristics)
- [7. Unified Logging & Reproducibility](#7-unified-logging--reproducibility)
- [8. Environment Customization & Wrappers](#8-environment-customization--wrappers)
- [9. Agent Implementation Guide](#9-agent-implementation-guide)
- [10. Logic-Neural Bridge (The "Blender")](#10-logic-neural-bridge-the-blender)
- [11. Method Style Registry](#11-method-style-registry)
- [12. Plotting Architecture](#12-plotting-architecture)
- [13. Pipeline Architecture](#13-pipeline-architecture)
- [14. Reciprocal Refinement (EP ↔ CQL Co-Training)](#14-reciprocal-refinement-ep--cql-co-training)
- [15. Maintenance & Evolution](#15-maintenance--evolution)

---

## 1. Project Overview & Pipeline Rules

BlendRL jointly learns symbolic (logic) and neural policies. It integrates logic agents (NSFR/Neumann) with RL (PPO, IQL) to create hybrid agents.

**No standalone scripts.** Do not write `if __name__ == "__main__":` Python scripts, bash scripts, or one-off files in the root or `scripts/` directories. Use the modular pipeline:
- **Run experiments:** `python run_pipeline.py <group>/<exp>` — configure via Hydra in `in/config/`.
- **Add agents/plotters/tasks:** Follow the `pipeline-crud` skill.
- **Temporary debugging scripts:** Only in `<AppDataDir>/brain/<conversation-id>/scratch/`.

---

## 2. Learning Paradigms

An experiment's declared **paradigm** determines what training, evaluation, and plots are valid. The pipeline validates paradigm compatibility at startup and raises an error on mismatches. Plots declared by the experiment config drive the course of training and evaluation — not hardcoded logic.

### `online_rl`
Trains agents via live simulator rollouts (e.g. CartPole, Seaquest).
- **Agents:** Online RL algorithms (`allowed_agents: [ppo]`, forbidding `cql, iql, cew`).
- **Evaluation:** Fixed-episode average reward in the live simulator via `EnvironmentEvaluatorCallback`.
- **Required env keys:** `offline_only: false`.

### `offline_rl`
Trains agents on static transition datasets (e.g. MIMIC, Pyrenees, pre-collected datasets).
- **Agents:** Offline RL algorithms (`allowed_agents: [cql, iql, cew]`, forbidding `ppo`).
- **Data & Evaluation:** Replay buffer data module; validation loss and Bellman error via Lightning.
- **Constraints:** `intervals_count: 1`, `eval_episodes: 0`.

### `supervised`
Supervised representation learning and early prediction tasks.
- **Models:** Predictive architectures (`allowed_models: [dnn, cnn, resnet, lstm, gru, transformer, ecm, ...], forbidding standalone RL agents).

> **Paradigm enforcement:** Every experiment strictly declares its `paradigm: <name>`. Pre-flight validation enforces that declared methods and environments are valid for that paradigm. Multi-paradigm workflows (e.g. online collection feeding offline RL) are orchestrated at the workflow task level, keeping individual paradigms clean and decoupled.

---

## 3. Execution Pipeline & Configuration

### Modular Specification
- **Configuration:** All experiment parameters are in YAML files in `in/config/`.
- **Precedence:** CLI arguments > experiment YAML > group `_base.yaml` > env/mode/site defaults > `config.yaml` root defaults.
- **Entry Point:** `python run_pipeline.py <experiment_name> [optional Hydra overrides]` is the sole entry point.
- **Dispatch Model:** `task:` has been completely removed. Routing is governed solely by `paradigm:` (e.g., `online_rl`, `offline_rl`, `supervised`) for single experiments, or `workflow:` (e.g., `sepsis_reciprocal`) for multi-paradigm DAG workflows. Stale `task:` declarations trigger a deprecation warning at startup.
- **Method Execution:** Driven by the structured `methods:` dictionary. Methods are dispatched sequentially (or swept via Hydra/Optuna) for the declared paradigm.

### 3-Tier Hierarchical Configuration & Composite Models
Configurations follow a 3-tier hierarchy that eliminates parameter repetition and cleanly separates algorithm, model, and training hyperparameters:
1. **Tier 1: Defaults & Base Profiles**: Base defaults defined in `in/config/agent/<algo>.yaml` and `in/config/model/<arch>.yaml`.
2. **Tier 2: Universal Experiment Parameters (`methods.params`)**:
   - Universal scalars and training hyperparameters applied across all methods (e.g. `epochs_per_interval: 25`, `gamma: 0.95`).
   - Algorithmic hyperparameter groups (`params.agent.<algo>` or `params.<algo>`).
   - Model architecture hyperparameter groups (`params.model.<arch>` or `params.<arch>`).
3. **Tier 3: Method-Level Declarations (`methods.<method_name>`)**:
   - Method declarations inherit from Tier 1 & Tier 2, overriding only differences.
   - **Composite Model Specification (BlendRL)**: BlendRL composite models (`model: blendrl` or `model.blendrl`) assemble 3 constituent modules (`neural`, `symbolic`, `blender`). Sub-models automatically inherit from their respective `in/config/model/<type>.yaml` files (e.g. `dueling_resnet.yaml`, `cew.yaml`, `nsfr.yaml`, `neumann.yaml`) and can be specified flat or deeply nested:
     ```yaml
     cql_blendrl_cew_dueling_resnet:
       agent: cql
       model:
         blendrl:
           neural: dueling_resnet
           symbolic:
             cew:
               ecm_dthr: 0.03
     ```

### Declarative Environment Config Keys
Environment YAMLs (`in/config/env/*.yaml`) define behavior declaratively — no hardcoded env-name checks in Python:
- `offline_only: true/false` — drives paradigm selection
- `monitor_metric: "val/loss"` or `"eval/reward"` — checkpointing/tuning target
- `preprocess_on_load: true/false` — whether dataset needs preprocessing on load
- `default_plots: [...]` — default plotters for this environment; **experiments can override via `plots:` in their YAML**

### Cluster Resources in Config
```yaml
resources:
  time: "04:00:00"
  gpus: 1
  cores: 16
  memory: "32G"
```
Priority: CLI flags > experiment `resources:` > site config > hardcoded fallbacks.

### Dataset & Experiment Mapping
- **Experiment ID & Group:** Inferred from file path `in/config/experiment/<group>/<experiment_id>.yaml`.
- **Method Lists:**
    - `online_methods`: Comma-separated (e.g., `ppo/cp_tuned,blendrl/cp_tuned`).
    - `offline_methods`: Comma-separated (e.g., `iql/cp_tuned,blendrl_iql/cp_tuned`).
    - `offline_datasets`: (Optional) Explicit dataset names. Slashes normalized to underscores internally. Many-to-many execution: every dataset × every offline method.
        - Matches online method datasets from the same run, else searches `results/datasets/[GROUP]/[EXP_ID]/[NORMALIZED_NAME]`. Exits if not found.
        - Default: all offline methods run against all online-generated datasets.
- **Standard Paths:**
    - Logs: `results/logs/[GROUP]/[EXP_ID]/[AGENT]`
    - Datasets: `results/datasets/[GROUP]/[EXP_ID]/[AGENT]`
    - Plots: `results/plots/[GROUP]/[EXP_ID]/`
- **Auto-Plotting:** `plot/manager.py` runs automatically at end of cycle unless `no_plot=true`.

---

## 4. Data Saving & Dataset Schema
- **Writer:** `DatasetWriter` (`src/app/dataset_utils.py`) serializes transitions during online phase.
- **Compression:** Observations converted to `uint8` (images) or `float32` (vectors) on CPU before saving. Chunked `.pkl` files (default `chunk_size=100,000`).
- **Dataset Manifest:** `DatasetWriter.close()` writes `dataset_manifest.json` with: agent, experiment_id, group, env, seed, total transitions, chunk count, timestamp (ISO-8601), git commit, branch, dirty flag.
- **Transition Schema:**
    - `obs` / `next_obs`: Neural input.
    - `logic_obs` / `next_logic_obs`: Symbolic input (hybrid agents).
    - `action`, `reward`, `done`: Standard RL fields.
- **Downstream Usage:** `RLDataModule` (`src/app/data/rl_data_module.py`) reads chunks into the offline replay buffer.

---

## 5. Operational Environment
- **Site Profiles (`site`):**
    - `site=local` (default): Interactive CLI execution, standard subprocess.
    - `site=ncshare` / `site=arc`: Generates and submits Slurm batch scripts.
    - **Cluster Push Mandate:** Push all changes to GitHub before submitting cluster jobs.
- **Filesystem:**
    - All datasets and training workloads must use repository paths (`in/datasets/`) or local user storage (`/hpc/home/`).
- **State Recovery:** Use `recover=true` to resume from the latest checkpoint. States saved in `results/checkpoints/[GROUP]/[EXP_ID]/[AGENT]/`.

---

## 6. Hydra & Debugging Heuristics
- **Hydra Outputs:** Timestamped directories in `results/hydra/outputs/YYYY-MM-DD/HH-MM-SS/` with `.hydra/config.yaml` and logs.
- **Debugging Failures:**
    - **Logs:** `results/logs/[GROUP]/[EXP_ID]/[AGENT]/version_X/metrics.csv`.
    - **Config audit:** `.hydra/config.yaml` in the hydra outputs folder.
- **Logic Rule Changes:** Always investigate Python valuation code first before touching rules in `in/rules/`. Ask before modifying rules — they are domain ground truths.
- **Surgical Updates:** Do not refactor unrelated code.
- **Verification:** Ensure `run_pipeline.py` orchestration is never broken and `src/app/train.py` remains compatible with Hydra.

---

## 7. Unified Logging & Reproducibility
- **Logger:** `CSVLogger` (`metrics.csv`) is the foundational logger. `TensorBoardLogger` is optional, disabled by default (`tensorboard: true` to enable).
- **Metrics Source:** `results/logs/[GROUP]/[EXP_ID]/[AGENT]/version_X/metrics.csv`.
- **Universal X-Axis:** `transitions` column — aligns online and offline agents at identical data exposure.
- **Reproducibility (`src/app/core/metadata.py`):** Every run captures git provenance (commit, branch, dirty flag + patch), system info, CLI command, seed. Written to `runtime.json` in both checkpoint and log directories.
- **Hydra Overrides:** `.hydra/overrides.yaml` copied into the log version directory.

---

## 8. Environment Customization & Wrappers
- **Interface:** All environments wrapped via `VectorizedNudgeBaseEnv` (`src/app/core/env_vectorized.py`).
- **Evaluation:** `EnvironmentEvaluatorCallback` (`src/app/core/callbacks.py`) executes fixed episode counts (default 100) at transition intervals.
- **Environment Logic:** `in/envs/[ENV]/` — custom reward shaping (`blenderl_reward.py`) and logic valuation (`valuation.py`). Environments do not define models.

---

## 9. Agent & Model Implementation Guide
Agents are PyTorch Lightning Modules in `src/usr/methods/` (registered in `agent_registry.py`):
- `PPOAgent`: Standard online actor-critic RL.
- `IQLAgent`: Offline RL using Implicit Q-Learning.
- `BlendRLAgent` / `BlendRLIQLAgent`: Hybrid logic-neural agents.
- **Model Architecture Selection:** All models live in `src/usr/models/` and are registered via `@register_model`. Instantiated through `build_model()` (`src/app/core/model_registry.py`).

See `pipeline-crud` skill for step-by-step instructions on adding new agents.

---

## 10. Logic-Neural Bridge (The "Blender")
- **Reasoners:** NSFR (Neural Symbolic Forward Reasoner) or Neumann.
- **Rule Loading:** `get_blender` (`src/app/core/factories.py`) initializes from `in/rules/`.
- **Hybrid Forward Pass:** Raw observations processed by neural encoders and logic reasoners simultaneously; outputs blended into a single policy.

---

## 11. Config-First Method Styling & Plot Overrides
- **Config-First:** Method visual styles are specified directly in experiment YAMLs (e.g. `label`, `color`, `linestyle`, `marker`, or `style: { ... }`) under `methods.<name>`.
- **Zero-Boilerplate Default:** When nothing is specified, the plotter uses the method name in the legend and automatically assigns distinct colors from a high-contrast palette (`QUALITATIVE_PALETTE`) to all methods on the same plot.
- **Plot Overrides:** Group defaults (`_base.yaml`) or experiment YAMLs can declare plot overrides (e.g. `plots.title`, `plots.xlabel`, `plots.smoothing_window`) at the top level or per-plotter (`plots.convergence`).
- **Template Inheritance:** Methods can declare `base: <template>` to inherit from `in/config/experiment/<group>/methods/<template>.yaml` under any custom method key.

---

## 12. Plotting Architecture
- **Auto-Discovery:** `plot/manager.py` scans `plot/` for `BasePlotter` subclasses. No manual registry needed.
- **Environment Defaults:** `default_plots:` in `in/config/env/*.yaml`. Experiments override via `plots:` in their YAML.
- **Config Resolution:** Plotters read the saved `config.yaml` from `results/logs/` or `results/checkpoints/`, falling back to the experiment YAML.
- **Shared Utilities:** `BasePlotter.plot_metric_series()` handles multi-method plotting with multi-version mean±SEM.

See `pipeline-crud` skill for adding new plotters.

---

## 13. Pipeline Architecture (`run_pipeline.py` & `src/app/pipeline/`)

**Dispatch model (no `task:` key):** `run_pipeline.py` routes directly based on config keys:
1. `workflow: <id>` → `src/app/pipeline/workflow/executor.py` runs the named DAG from `in/config/workflow/<id>.yaml`.
2. Otherwise → `local_runner.run_local_training()` (site=local) or `slurm_runner.run_slurm_training()` (any cluster site).

> [!IMPORTANT]
> `task:` has been completely removed. If a stale `task:` key is found at startup, the pipeline emits a deprecation warning and ignores it. Routing is governed by `paradigm:` (single experiment) and `workflow:` (multi-experiment DAG).

- **`src/app/pipeline/workflow/executor.py`**: Workflow DAG executor. Loads `WorkflowGraph` from `in/config/workflow/*.yaml`, walks topological levels, dispatches paradigm nodes as `run_pipeline.py` subprocesses.
- **`src/app/pipeline/workflow/model.py`**: `WorkflowGraph`, `WorkflowNode`, `WorkflowString`, `PortType` — string diagram data model.
- **`src/app/pipeline/config.py`**: Name normalization, method list parsing, CLI arg sanitization.
- **`src/app/pipeline/datasets.py`**: Dataset path resolution, online symlinking, plotting dispatch.
- **`src/app/pipeline/slurm.py`**: Slurm header builder (`generate_sbatch_header`), job submission (`submit_sbatch`). Mail: `--mail-type=END,FAIL`, `--mail-user=egbertcm23@gmail.com`.
- **`src/app/pipeline/local_runner.py`**: Local sequential phase execution (`_setup_output_dirs` → `run_methods` → `run_plotting_phase`).
- **`src/app/pipeline/slurm_runner.py`**: Cluster batch script generation and job dependency orchestration.
- **`src/app/pipeline/shape_rewards.py`**: Offline ETL utility — `run_shape_rewards(cfg, context=None)`. Not a registered task; called as a library function from workflow transform nodes.
- **`src/app/pipeline/optuna_utils.py`**: SQLite URL constants, study management, dashboard launching.
- **`src/app/pipeline/validation.py`**: Pre-flight paradigm compatibility and config validation.

---

## 14. Reciprocal Refinement (EP ↔ CQL Co-Training)
- **Concept:** EP septic shock predictor and CQL policy iteratively improve each other.
- **Workflow Orchestration:** Reciprocal refinement is orchestrated as a declarative Workflow DAG (`in/config/workflow/sepsis_reciprocal.yaml`) executed via `workflow: sepsis_reciprocal` in `in/config/experiment/mimic/reciprocal_refinement.yaml`.
- **Reward Shaping (`src/usr/eval/reward_shaping.py`):** Potential-based (Ng et al. 1999):
    - Φ(s) = −P_EP(shock | observation window ending at s)
    - r_shaped = r_TQN + λ * (γ * Φ(s') − Φ(s))
- **Config:** `in/config/env/mimic.yaml` under `reward_shaping:`. `reward_type: ep_shaped` activates it via `in/envs/mimic/hooks.py`.
- **Results:** `results/checkpoints/[GROUP]/[EXP_ID]/roundN/`, `results/plots/[GROUP]/[EXP_ID]/convergence_log.json`.

---

## 15. Maintenance & Evolution
- **Update this document** when new standard paths, evaluation standards, or core algorithms are introduced.
- **Workflow reference:** [`docs/WORKFLOW_GUIDE.md`](file:///Users/cameronegbert/Research/NeSyRL/docs/WORKFLOW_GUIDE.md).

---
*Last Updated: 2026-09-16*
