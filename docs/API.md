# ThetaIDE ↔ NeSyRL Pipeline Interface & Team Onboarding Guide

This document describes how **ThetaIDE** drives the **NeSyRL** training pipeline (in-process services plus a QProcess job manager; there is no HTTP server), and is a guide for onboarding new team members.

---

## 1. Quick Start for Team Members

### 1.1 Development Setup

```bash
# 1. Clone repository
git clone https://github.com/fixmeseb/Theta-IDE.git
cd Theta-IDE

# 2. Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies + dev tools + the desktop app
pip install -e ".[dev,gui]"

# 4. Verify test suite
pytest tests/ -v
```

### 1.2 Running a Smoke Test (Simulator vs. Offline Datasets)

Because `results/` and `in/datasets/` are `.gitignore`d to prevent committing large binary files, teammates should understand how datasets work:

1. **Simulator-Backed Benchmarks (`online_v_offline` paradigm):**
   Environments like `CartPole` or `MountainCar` **do not require any downloaded datasets**. 
   Running an experiment automatically generates the replay buffer in Phase 1 and trains offline agents on it in Phase 2:
   ```bash
   # Fast local smoke test (~15-30 seconds)
   python run_pipeline.py cp_final total_timesteps=1000 eval_episodes=5 site=local
   ```

2. **Static Clinical / Offline Datasets (`offline_rl` paradigm):**
   Datasets like `MIMIC` or `Pyrenees` reside in `in/datasets/mimic/` or `in/datasets/pyrenees/`. 
   If working on clinical offline RL, obtain the dataset chunks (`.pkl`) from the team storage or run the preprocessing utilities in `scripts/preprocess_pyrenees.py`.

---

## 2. Running ThetaIDE

```bash
python -m frontend          # from the repository root, with the venv active
```

ThetaIDE runs from the same venv as the pipeline. It imports the pipeline's composition and validation code
directly and launches `run_pipeline.py` as a child process, so there is nothing else to start.

---

## 3. Interface Reference

Everything the GUI used to request over HTTP is now a direct call. The pure-Python services live in
`src/app/services/` and have no Qt dependency, so scripts and tests can use them too. The Qt side lives in
`frontend/job_manager.py`, `frontend/local_backend.py` and `frontend/tensorboard.py`.

### 3.1 Services (`src/app/services/`)

| Function | Returns | Replaces |
|---|---|---|
| `config_service.config_schema()` | The builder's fields with defaults read from the composed base experiment | `GET /api/config/schema` |
| `config_service.compose_report(experiment, overrides)` | `{valid, argv, config, config_yaml, methods, methods_yaml, notices, errors}`, composed exactly as `run_pipeline.py` would | `POST /api/config/compose` |
| `config_service.prepare_launch(experiment, overrides, overwrite, active_jobs)` | The job description (group, experiment ID, agents, effective timesteps); raises `LaunchError` (`.conflict` for existing results or a clashing job) | launch validation |
| `catalog.list_runs()` | Every `results/logs/<group>/<id>` folder with config summary and per-agent timing/reward | `GET /api/runs` |
| `catalog.run_metrics(group, experiment_id, agent)` | All rows of the newest `metrics.csv`; raises `FileNotFoundError` | `GET /api/runs/.../metrics` |
| `catalog.list_plots(group, experiment_id)` | Plot file names under `results/plots/` | `GET /api/runs/.../plots` |
| `catalog.list_environments()`, `list_experiments()`, `list_methods()` | Config catalog | `GET /api/environments`, `/experiments`, `/methods` |
| `metrics.MetricsTail(path).read()` | `(new rows, reset)`: only rows appended since the last read; `reset` when Lightning rewrote the file | incremental telemetry |
| `processes.kill_process_tree(pid)` | Stops the pipeline and every training/plotting subprocess it started | cancel |
| `job_store.JobStore(path)` | SQLite persistence for jobs and queue order | unchanged |

### 3.2 Async calls from the GUI (`frontend/local_backend.py`)

`LocalBackend` runs `config_schema`, `compose`, `list_runs` and `run_metrics` on one background thread
(Hydra's global state is not thread-safe, so calls never overlap) and calls back on the GUI thread with
`(data, error)`, the same shape the old network client used.

### 3.3 Training jobs (`frontend/job_manager.py`)

`JobManager` launches `python -u run_pipeline.py <experiment> <overrides…>` as a `QProcess` from the project
root. Nothing polls: output arrives through `readyReadStandardOutput`, metrics through a
`QFileSystemWatcher` on the run's folders and `metrics.csv`, and status changes as signals.

| Method | Behaviour |
|---|---|
| `launch(experiment, overrides, queue=False, overwrite=False)` | Validates via `prepare_launch`, then starts the job (or queues it). Raises `LaunchError`. |
| `cancel(job_id)` | Queued: removed from the queue. Running: kills the process tree; the job ends as `cancelled`. |
| `start_queue()` / `pause_queue()` | Queued jobs train one at a time, in order, only while the queue is started. It pauses itself when drained. Starting an empty queue leaves it paused. |
| `move(job_id, position)` | Reorders a queued job (raises `LaunchError` for a job that is not queued). |
| `queue_state()` | `{running, active, queued (with position), finished (ten most recent)}` |
| `job(job_id)`, `active_jobs()`, `is_busy()` | Current records. |
| `shutdown()` | Stops every running job (used when the app closes). |

| Signal | Arguments |
|---|---|
| `job_changed` | the job's public record after any status change (`pending` → `running` → `completed` / `failed` / `cancelled` / `error`) |
| `job_output` | `job_id`, new log lines (also written to `<data dir>/jobs/<job_id>.log`) |
| `job_metrics` | `job_id`, agent, new metric rows, `reset` |
| `job_hardware` | `job_id`, `{cpu_percent, memory_mb}`, sampled at most once a second while output arrives |
| `queue_changed` | the same dictionary as `queue_state()` |

Jobs are children of the app. While one is training, closing the window hides ThetaIDE to the system tray (`frontend/tray.py`) so it keeps running; Quit asks before stopping a running job. A job still marked
running from a previous session is recorded as `interrupted`. Queued jobs are kept across sessions.

### 3.4 TensorBoard (`frontend/tensorboard.py`)

`TensorBoardServer` runs `python -m tensorboard.main --logdir results/tensorboard` on a free local port. It
emits `ready(url)` when TensorBoard prints that it is serving and `stopped(error)` when it exits (`""` after
a deliberate `stop()`).
