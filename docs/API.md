# NeSyRL Backend API & Team Onboarding Guide

This document serves as the interface specification for integrating **ThetaIDE** (or any custom frontend/client) with the **NeSyRL** backend engine, as well as a guide for onboarding new team members to the repository.

---

## 1. Quick Start for Team Members

### 1.1 Development Setup

```bash
# 1. Clone repository
git clone https://github.com/CameronEgb/Offline-BlendRL.git
cd Offline-BlendRL

# 2. Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies + dev tools + API service
pip install -e ".[dev,api]"

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
   python run_pipeline.py cp_final total_timesteps=1000 intervals_count=2 eval_episodes=5 site=local
   ```

2. **Static Clinical / Offline Datasets (`offline_only` paradigm):**
   Datasets like `MIMIC` or `Pyrenees` reside in `in/datasets/mimic/` or `in/datasets/pyrenees/`. 
   If working on clinical offline RL, obtain the dataset chunks (`.pkl`) from the team storage or run the preprocessing utilities in `scripts/preprocess_pyrenees.py`.

---

## 2. Launching the Backend API Server

The backend exposes a lightweight FastAPI service located in `src/app/api/app.py`.

```bash
# Start API server on localhost:8000 with hot-reloading
uvicorn src.app.api.app:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 3. API Endpoint Reference

### 3.1 System Health
* **`GET /api/health`**
  * **Description:** Health-check endpoint to confirm server availability.
  * **Response:**
    ```json
    {"status": "ok"}
    ```

---

### 3.2 Metadata & Exploration (Experiment Builder Panel)

* **`GET /api/environments`**
  * **Description:** Discovers all declared environment YAML specifications under `in/config/env/`.
  * **Response:**
    ```json
    {
      "environments": [
        {
          "name": "cartpole",
          "config": {"name": "cartpole", "offline_only": false, ...}
        }
      ]
    }
    ```

* **`GET /api/methods`**
  * **Description:** Lists all registered agent architectures (`AGENT_REGISTRY`). Method visualization styles are configured directly in experiment YAMLs.
  * **Response:**
    ```json
    {
      "registered_agents": ["blendrl_iql", "cql", "iql", "ppo"],
      "method_styles": {}
    }
    ```

* **`GET /api/experiments`**
  * **Description:** Lists pre-configured experiment recipes in `in/config/experiment/**/*.yaml`.
  * **Response:**
    ```json
    {
      "experiments": [
        {
          "path": "cartpole/cp_final.yaml",
          "name": "cartpole/cp_final",
          "config": {"paradigm": "online_v_offline", "online_methods": "ppo/cp_tuned", ...}
        }
      ]
    }
    ```

* **`GET /api/config/schema`**
  * **Description:** Describes the fields of the ThetaIDE experiment builder (currently CartPole/PPO). Defaults come from composing the builder's base recipe `thetaide/_base`, so the GUI never keeps its own copy of backend defaults. Each field's `key` is the Hydra override path it sets.
  * **Response:**
    ```json
    {
      "base_experiment": "thetaide/_base",
      "paradigm": "online_rl",
      "environment": {"name": "cartpole", "env_id": "CartPole-v1"},
      "method": {"name": "ppo", "agent": "ppo", "model": "dnn"},
      "fixed_overrides": ["++methods.ppo.agent=ppo", "++methods.ppo.model=dnn"],
      "fields": [
        {"key": "methods.ppo.lr", "label": "Learning rate", "type": "float", "default": 0.0003,
         "min": 1e-06, "max": 1.0, "step": 0.0001, "help": "Adam learning rate for the PPO policy."}
      ]
    }
    ```

* **`POST /api/config/compose`**
  * **Description:** Composes and validates an experiment exactly as `run_pipeline.py` would (both use `src/app/pipeline/compose.py`), without running it. Invalid configs still return HTTP 200 with `valid: false`; each error's `stage` is `compose` (unknown recipe or bad override), `validation` (paradigm checks) or `methods` (method parsing).
  * **Request Body:**
    ```json
    {"experiment": "thetaide/_base", "overrides": ["seed=7", "++methods.ppo.agent=ppo", "++methods.ppo.model=dnn", "++methods.ppo.lr=0.001"]}
    ```
  * **Response:**
    ```json
    {
      "valid": true,
      "experiment": "thetaide/_base",
      "argv": ["python", "run_pipeline.py", "thetaide/_base", "seed=7", "..."],
      "config": {"seed": 7, "agent": {"lr": 0.0003, "...": "..."}, "methods": {"ppo": {"lr": 0.001, "...": "..."}}},
      "config_yaml": "...",
      "methods": {"ppo": {"settings": {"agent": "ppo", "model": "dnn", "lr": 0.001}, "train_overrides": ["...", "++agent.lr=0.001"]}},
      "methods_yaml": "...",
      "notices": [],
      "errors": []
    }
    ```
    `config` is the composed config; per-method values only reach `agent.*` in each method's `train_overrides`, which are the exact arguments its training subprocess receives.

---

### 3.3 Execution & Job Control (Training Monitor Panel)

* **`POST /api/experiments/launch`**
  * **Description:** Composes and validates the experiment exactly as `run_pipeline.py` will, then runs `run_pipeline.py` in the background. Output is unbuffered, so logs stream live. The job records where its metrics will appear.
  * **Request Body:**
    ```json
    {"experiment": "thetaide/_base", "overrides": ["++experiment_id='my_run'", "total_timesteps=20000", "++methods.ppo.agent=ppo", "++methods.ppo.model=dnn"], "overwrite": false}
    ```
  * **Errors:**
    * `422`: the config does not compose or validate. `detail` holds the message.
    * `409`: `results/logs/<group>/<experiment_id>/` already has results, and the pipeline would purge them. Choose another `experiment_id` or pass `"overwrite": true`.
  * **Response:**
    ```json
    {"job_id": "cc912a03-…", "status": "pending", "experiment": "thetaide/_base", "group": "thetaide",
     "experiment_id": "my_run", "agents": ["ppo"], "total_timesteps": 20000, "created": 1790000000.0, "request": {"…": "…"}}
    ```

* **`GET /api/experiments/{job_id}/status?since=N`**
  * **Description:** Job state plus the pipeline's output lines after line `N` (0-based, counted from job start). Pass the previous `log_total` as `since` to receive only new lines. The full log is also written to `results/jobs/<job_id>.log`.
  * **Response:**
    ```json
    {"status": "running", "pid": 48210, "returncode": null, "log": ["Evaluation at 5000 transitions: Avg Reward = 30.49 (+/- 14.79)"],
     "log_total": 131, "stdout": "…last lines…", "group": "thetaide", "experiment_id": "my_run", "agents": ["ppo"]}
    ```
  * *Status values:* `"pending"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`, `"error"`.

* **`GET /api/experiments/{job_id}/metrics?since=N`**
  * **Description:** Numeric rows from each agent's newest `results/logs/<group>/<experiment_id>/<agent>/version_N/metrics.csv`, read live while training. It returns rows after index `N`. A line still being written is skipped. If Lightning rewrote the file with fewer rows, `reset` is `true` and all rows are returned. Evaluation rows carry `eval/reward` and training rows carry `losses/*`, both keyed by `transitions`.
  * **Response:**
    ```json
    {"job_id": "…", "status": "running", "agents": {"ppo": {"source": "results/logs/thetaide/my_run/ppo/version_0/metrics.csv",
      "total": 42, "reset": false, "rows": [{"eval/reward": 30.49, "transitions": 5000.0, "step": 16.0}]}}}
    ```

* **`GET /api/experiments/jobs`**
  * **Description:** Lists the jobs this server process knows about. Jobs are held in memory, so restarting the API forgets them. Their logs and metrics stay on disk.

* **`POST /api/experiments/{job_id}/cancel`**
  * **Description:** Stops a pending or running job. A job that has not started never spawns. A running job has its whole process tree terminated: the pipeline plus the `train.py` and plotting subprocesses it started (`taskkill /T` on Windows, process-group `SIGTERM` elsewhere).
  * **Response:** `{"job_id": "…", "status": "cancelled"}`, or `{"status": "completed", "message": "Job is already completed"}` for a finished job.

---

### 3.3b Job queue (Queue tab)

Launching with `"queue": true` adds a job to a first-in, first-out queue instead of starting it immediately. It is validated exactly as a direct launch is (422 and 409 as described above). Queued jobs do not start until the queue is started with `POST /api/queue/start`. While started, a background worker starts the next queued job only when no job is pending or running, including jobs launched directly, so queued jobs train one at a time, in order. Once the queue has drained and its last job has finished, it pauses itself, so jobs added later wait for the next start. The queue lives in the API process: it keeps running when GUI clients close, and it is lost if the API restarts. A launch or enqueue whose `group/experiment_id` is already used by a queued, pending or running job returns `409`.

* **`POST /api/experiments/launch`** with `{"experiment": "...", "overrides": [...], "queue": true}` returns the job with `"status": "queued"`, its 0-based `position` and `queue_running`.
* **`POST /api/queue/start`** starts the queue and returns `{"running": true, "queued": N}`. Starting an empty queue leaves it paused (`"running": false`).
* **`POST /api/queue/pause`** stops new jobs from starting. A job that is already training keeps running.
* **`GET /api/queue`** returns `{"running": false, "active": [...], "queued": [...], "finished": [...]}`. `queued` is in run order, and each entry has a `position`; `finished` holds the ten most recent finished jobs. Jobs carry `created`, `started` and `finished` timestamps.
* **`POST /api/queue/{job_id}/move`** with `{"position": 0}` moves a queued job. The position is clamped to the queue's length. Returns `409` if the job is not queued.
* **`POST /api/experiments/{job_id}/cancel`** on a queued job removes it from the queue. It never starts, and its status becomes `cancelled`.

---

### 3.3a TensorBoard (TensorBoard tab)

The API manages a single local TensorBoard server over `results/tensorboard/`. Runs write there when launched with `tensorboard=true`, which `thetaide/_base` sets by default. The server binds to 127.0.0.1 on a free port. Its output goes to `results/jobs/tensorboard.log`, and it is stopped when the API exits normally.

* **`GET /api/tensorboard`**: `{"available": true, "running": true, "ready": true, "url": "http://127.0.0.1:55860/", "logdir": "results/tensorboard", "pid": 1234}`. `ready` becomes true when the server answers, about a second after start. After an unexpected exit, `exit_code` and `error` (the tail of its output) are included.
* **`POST /api/tensorboard/start`**: starts the server if it isn't running and returns immediately with the same shape. Poll `GET` until `ready`. Returns `503` if TensorBoard isn't installed in the backend environment.
* **`POST /api/tensorboard/stop`**: stops the server and its child processes.

---

### 3.4 Results & Visualization (Results Browser Panel)

* **`GET /api/runs`**
  * **Description:** Scans `results/logs/` and returns all completed experiment runs along with `runtime.json` metadata (git commit, hardware, timing).
  * **Response:**
    ```json
    {
      "runs": [
        {
          "group": "cartpole",
          "experiment_id": "cp_final",
          "metadata": {
            "agent": "ppo_cp_tuned",
            "training_time_seconds": 23.4,
            "git_commit": "31b56ad"
          }
        }
      ]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/{agent}/metrics`**
  * **Description:** Parses the latest `metrics.csv` for the given agent and experiment into structured JSON for frontend graphing.
  * **Response:**
    ```json
    {
      "source": "results/logs/cartpole/cp_final/ppo_cp_tuned/version_0/metrics.csv",
      "metrics": [
        {"epoch": "0", "step": "0", "eval/reward": "18.2", "transitions": "0.0"},
        {"epoch": "1", "step": "256", "eval/reward": "194.5", "transitions": "50000.0"}
      ]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/plots`**
  * **Description:** Lists all generated visual plots (PNG files) in `results/plots/{group}/{experiment_id}/`.
  * **Response:**
    ```json
    {
      "plots": ["convergence_reward.png", "losses_actor.png"]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/plots/{filename}`**
  * **Description:** Streams the actual plot image file (e.g., `image/png`) directly to the client for rendering.
