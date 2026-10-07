import atexit
import csv
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.app.pipeline.compose import compose_experiment, effective_config, method_plans, validate_composed
from src.app.pipeline.config import normalize_agent_name

try:
    from src.usr.methods.method_style_registry import METHOD_STYLE
except ImportError:
    METHOD_STYLE = {}
try:
    from src.usr.methods.agent_registry import list_registered_agents
except ImportError:

    def list_registered_agents():
        return []


try:
    import psutil
except ImportError:
    psutil = None

from src.app.api.job_store import JobStore

app = FastAPI(title="NeSyRL API")

# Enable CORS for frontend clients (Vite, Next.js, Electron, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Browsers reject a wildcard origin when credentials are allowed, and this
    # API has no cookies or auth to send, so credentials stay off.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict[str, Any]] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/environments")
def list_environments():
    envs = []
    env_dir = Path("in/config/env")
    if env_dir.exists():
        for f in env_dir.glob("*.yaml"):
            if f.name.startswith("_"):
                continue
            try:
                with open(f) as yaml_file:
                    envs.append({"name": f.stem, "config": yaml.safe_load(yaml_file)})
            except Exception:
                pass
    return {"environments": envs}


@app.get("/api/methods")
def list_methods():
    return {
        "registered_agents": list_registered_agents(),
        "method_styles": METHOD_STYLE,
    }


@app.get("/api/experiments")
def list_experiments():
    exps = []
    exp_dir = Path("in/config/experiment")
    if exp_dir.exists():
        for f in sorted(exp_dir.glob("**/*.yaml")):
            # Ignore internal base / template configs
            if f.name.startswith("_"):
                continue
            rel_path = f.relative_to(exp_dir)
            try:
                with open(f) as yaml_file:
                    exps.append(
                        {
                            "path": str(rel_path),
                            "name": str(rel_path.with_suffix("")),
                            "config": yaml.safe_load(yaml_file),
                        }
                    )
            except Exception:
                pass
    return {"experiments": exps}


@app.get("/api/runs")
def list_runs():
    runs = []
    logs_dir = Path("results/logs")
    if logs_dir.exists():
        for run_dir in logs_dir.glob("*/*"):
            if run_dir.is_dir():
                runtime_file = run_dir / "runtime.json"
                metadata = {}
                if runtime_file.exists():
                    try:
                        with open(runtime_file) as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                runs.append(
                    {
                        "group": run_dir.parent.name,
                        "experiment_id": run_dir.name,
                        "metadata": metadata,
                    }
                )
    return {"runs": runs}


@app.get("/api/runs/{group}/{experiment_id}/{agent}/metrics")
def get_metrics(group: str, experiment_id: str, agent: str):
    agent_dir = Path(f"results/logs/{group}/{experiment_id}/{agent}")
    # Metrics live in version_N subdirectories
    metrics_file = None
    if agent_dir.exists():
        versions = sorted(agent_dir.glob("version_*/metrics.csv"), reverse=True)
        if versions:
            metrics_file = versions[0]
    if not metrics_file or not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")

    metrics = []
    try:
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"metrics": metrics, "source": str(metrics_file)}


@app.get("/api/runs/{group}/{experiment_id}/plots")
def list_plots(group: str, experiment_id: str):
    plots = []
    plot_dir = Path(f"results/plots/{group}/{experiment_id}")
    if plot_dir.exists():
        for f in plot_dir.glob("*"):
            if f.is_file():
                plots.append(f.name)
    return {"plots": plots}


@app.get("/api/runs/{group}/{experiment_id}/plots/{filename}")
def get_plot_image(group: str, experiment_id: str, filename: str):
    file_path = Path(f"results/plots/{group}/{experiment_id}/{filename}")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path)


# Experiment builder contract. The GUI composes GUI_BASE_EXPERIMENT plus one override per
# field; defaults are read from the composed backend config rather than duplicated in the GUI.
GUI_BASE_EXPERIMENT = "thetaide/_base"
GUI_METHOD = {"name": "ppo", "agent": "ppo", "model": "dnn"}
GUI_FIELDS = [
    {
        "key": "experiment_id",
        "label": "Experiment name",
        "type": "str",
        "default": "cartpole_ppo_experiment",
        "pattern": r"^[A-Za-z0-9_\-]+$",
        "help": "Results directory name. Letters, digits, '_' and '-' only.",
    },
    {
        "key": "seed",
        "label": "Random seed",
        "type": "int",
        "default_from": "seed",
        "min": 0,
        "max": 2147483647,
        "help": "Seeds environments and network initialization.",
    },
    {
        "key": "total_timesteps",
        "label": "Total timesteps",
        "type": "int",
        "default_from": "total_timesteps",
        "min": 1000,
        "max": 10000000,
        "step": 1000,
        "help": "Environment steps across all training intervals.",
    },
    {
        "key": "methods.ppo.lr",
        "label": "Learning rate",
        "type": "float",
        "default_from": "agent.lr",
        "min": 1e-6,
        "max": 1.0,
        "step": 1e-4,
        "help": "Adam learning rate for the PPO policy.",
    },
    {
        "key": "methods.ppo.batch_size",
        "label": "Batch size",
        "type": "int",
        "default_from": "agent.batch_size",
        "choices": [32, 64, 128, 256],
        "help": "Minibatch size for each PPO update.",
    },
    {
        "key": "methods.ppo.gamma",
        "label": "Discount factor · γ",
        "type": "float",
        "default_from": "env.gamma",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "help": "Weight of future rewards.",
    },
    {
        "key": "tensorboard",
        "label": "Log to TensorBoard",
        "type": "bool",
        "default_from": "tensorboard",
        "help": "Also write TensorBoard logs to results/tensorboard/ for the TensorBoard tab.",
    },
]


def _rollout_notices(plans: dict[str, dict], total_timesteps) -> list[str]:
    notices = []
    for name, plan in plans.items():
        rollout = plan.get("rollout")
        if rollout and rollout["timesteps"] != total_timesteps:
            notices.append(
                f"Method '{name}' trains {rollout['timesteps']:,} steps, not {total_timesteps:,}: PPO collects whole "
                f"rollouts of {rollout['size']} steps ({rollout['num_envs']} envs × {rollout['num_steps']} steps), "
                f"so it runs {rollout['rollouts']} rollouts."
            )
    return notices


def _effective_timesteps(plans: dict[str, dict], total_timesteps):
    """Environment steps training will actually run: the largest method's rounded-up PPO budget."""
    rollouts = [plan["rollout"]["timesteps"] for plan in plans.values() if plan.get("rollout")]
    return max(rollouts, default=total_timesteps)


class ComposeRequest(BaseModel):
    experiment: str
    overrides: list[str] = []


@app.get("/api/config/schema")
def config_schema():
    composed = compose_experiment(GUI_BASE_EXPERIMENT)
    fields = []
    for field in GUI_FIELDS:
        field = dict(field)
        source = field.pop("default_from", None)
        if source:
            field["default"] = OmegaConf.select(composed.cfg, source)
        fields.append(field)
    method_prefix = f"methods.{GUI_METHOD['name']}"
    return {
        "base_experiment": composed.experiment,
        "paradigm": composed.cfg.get("paradigm"),
        "environment": {"name": composed.cfg.env.name, "env_id": composed.cfg.env.get("env_id")},
        "method": GUI_METHOD,
        "fixed_overrides": [
            f"++{method_prefix}.agent={GUI_METHOD['agent']}",
            f"++{method_prefix}.model={GUI_METHOD['model']}",
        ],
        "fields": fields,
    }


@app.post("/api/config/compose")
def compose_config(req: ComposeRequest):
    """Compose and validate an experiment exactly as `run_pipeline.py` would, without running it."""
    argv = ["python", "run_pipeline.py", req.experiment, *req.overrides]
    try:
        composed = compose_experiment(req.experiment, req.overrides)
    except Exception as e:
        return {
            "valid": False,
            "experiment": req.experiment,
            "argv": argv,
            "config": None,
            "config_yaml": "",
            "notices": [],
            "errors": [{"stage": "compose", "message": str(e)}],
        }

    errors, notices, methods = [], [], {}
    try:
        notices = validate_composed(composed)
    except Exception as e:
        errors.append({"stage": "validation", "message": str(e).strip()})
    try:
        methods = method_plans(composed)
        notices.extend(_rollout_notices(methods, composed.cfg.get("total_timesteps")))
    except Exception as e:
        errors.append({"stage": "methods", "message": str(e).strip()})
    config = effective_config(composed.cfg)
    return {
        "valid": not errors,
        "experiment": composed.experiment,
        "argv": composed.argv,
        "config": config,
        "config_yaml": yaml.safe_dump(config, sort_keys=False),
        "methods": methods,
        "methods_yaml": yaml.safe_dump({name: plan["settings"] for name, plan in methods.items()}, sort_keys=False),
        "notices": notices,
        "errors": errors,
    }


class LaunchRequest(BaseModel):
    experiment: str
    overrides: list[str] = []
    overwrite: bool = False
    """Allow reusing an experiment ID whose results already exist (the pipeline purges them)."""
    queue: bool = False
    """Add to the job queue instead of starting now. Queued jobs run one at a time, in order,
    once the queue is started."""


class MoveRequest(BaseModel):
    position: int
    """New 0-based position among the queued jobs (clamped to the queue's length)."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = PROJECT_ROOT / "results" / "jobs"
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "error")
MAX_LOG_LINES = 20000
ACTIVE_STATUSES = ("pending", "running")
_jobs_lock = threading.Lock()
_jobs_changed = threading.Condition(_jobs_lock)  # notified when a job finishes or the queue changes
queue_order: list[str] = []  # job IDs with status "queued", next to run first
# Jobs only start from the queue after POST /api/queue/start; the queue pauses itself again once it drains.
queue_state = {"running": False}
_queue_worker: threading.Thread | None = None

job_store = JobStore(JOBS_DIR / "jobs.db")

# Recover dead active jobs before populating in-memory state
job_store.recover_active_jobs()
for _stored in job_store.list_jobs():
    jobs[_stored["job_id"]] = _stored
queue_order.extend(job_store.get_queue())


def _sync_job(job_id: str) -> None:
    if job_id in jobs:
        job = jobs[job_id]
        job.setdefault("job_id", job_id)
        job_store.save_job(job, job_id=job_id)


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _append_log(job: dict[str, Any], line: str) -> None:
    log = job.setdefault("_log", [])
    log.append(line)
    if len(log) > MAX_LOG_LINES:
        dropped = len(log) - MAX_LOG_LINES
        del log[:dropped]
        job["log_dropped"] = job.get("log_dropped", 0) + dropped


def _kill_process_tree(pid: int) -> None:
    """Terminate the pipeline and every training/plotting subprocess it started."""
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs(children + [parent], timeout=2.0)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)  # the job was started as its own process group
        except ProcessLookupError:
            pass


def run_experiment_task(job_id: str, req: LaunchRequest):
    job = jobs[job_id]
    cmd = [sys.executable, "-u", "run_pipeline.py", job["experiment"], *req.overrides]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    popen_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True
    log_file = None
    try:
        with _jobs_lock:
            if job["status"] == "cancelled":
                return  # cancelled before it started
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )
            job.update(status="running", pid=process.pid, _process=process, started=time.time())
            _sync_job(job_id)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(JOBS_DIR / f"{job_id}.log", "w", encoding="utf-8")
        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                with _jobs_lock:
                    _append_log(job, line)
                log_file.write(line + "\n")
                log_file.flush()
        process.wait()
        with _jobs_lock:
            job["returncode"] = process.returncode
            if job["status"] != "cancelled":
                job["status"] = "completed" if process.returncode == 0 else "failed"
            _sync_job(job_id)
    except Exception as e:
        with _jobs_lock:
            if job.get("status") != "cancelled":
                job["status"] = "error"
            job["error"] = str(e)
            _sync_job(job_id)
    finally:
        with _jobs_changed:
            job["finished"] = time.time()
            _sync_job(job_id)
            _jobs_changed.notify_all()  # the queue worker may start the next job
        if log_file:
            log_file.close()


@app.post("/api/experiments/launch")
def launch_experiment(req: LaunchRequest):
    """Validate the experiment exactly as the pipeline will, then run it in the background."""
    try:
        composed = compose_experiment(req.experiment, req.overrides)
        validate_composed(composed)
        plans = method_plans(composed)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e).strip())

    cfg = composed.cfg
    log_dir = PROJECT_ROOT / "results" / "logs" / cfg.group / cfg.experiment_id
    if log_dir.exists() and any(log_dir.iterdir()) and not req.overwrite and not cfg.get("recover", False):
        raise HTTPException(
            status_code=409,
            detail=f"Results already exist for {cfg.group}/{cfg.experiment_id}; launching would delete them. "
            "Choose another experiment_id or pass overwrite=true.",
        )

    with _jobs_lock:
        clash = next(
            (
                j
                for j in jobs.values()
                if j.get("status") in ("queued", *ACTIVE_STATUSES)
                and (j.get("group"), j.get("experiment_id")) == (cfg.group, cfg.experiment_id)
            ),
            None,
        )
    if clash:
        raise HTTPException(
            status_code=409,
            detail=f"Job {clash['job_id']} ({clash['status']}) already uses {cfg.group}/{cfg.experiment_id}; "
            "running both would overwrite one run's results with the other's.",
        )

    job_id = str(uuid.uuid4())
    agents = [
        settings.get("name", normalize_agent_name(name))
        for name, settings in ((name, plan["settings"]) for name, plan in plans.items())
    ]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "request": req.model_dump(),
        "experiment": composed.experiment,
        "group": cfg.group,
        "experiment_id": cfg.experiment_id,
        "agents": agents,
        "total_timesteps": cfg.get("total_timesteps"),
        "effective_timesteps": _effective_timesteps(plans, cfg.get("total_timesteps")),
        "created": time.time(),
    }
    _sync_job(job_id)

    if req.queue:
        with _jobs_changed:
            jobs[job_id].update(status="queued", _request=req)
            _sync_job(job_id)
            queue_order.append(job_id)
            job_store.set_queue(queue_order)
            _ensure_queue_worker()
            _jobs_changed.notify_all()
            return dict(_public(jobs[job_id]), position=queue_order.index(job_id), queue_running=queue_state["running"])

    thread = threading.Thread(target=run_experiment_task, args=(job_id, req))
    thread.daemon = True
    thread.start()

    return _public(jobs[job_id])


# ── Job queue ────────────────────────────────────────────────────────────────


def _ensure_queue_worker() -> None:
    """Start the queue worker thread once. Call with _jobs_lock held."""
    global _queue_worker
    if _queue_worker is None or not _queue_worker.is_alive():
        _queue_worker = threading.Thread(target=_queue_worker_loop, name="job-queue", daemon=True)
        _queue_worker.start()


def _queue_worker_loop() -> None:
    """Run queued jobs one at a time while the queue is started.

    The next job starts only when no job (queued or launched directly) is active. When the queue has
    drained and the last job has finished, the queue pauses itself, so jobs added later wait for Start.
    """
    while True:
        with _jobs_changed:
            while True:
                idle = not any(j.get("status") in ACTIVE_STATUSES for j in jobs.values())
                if queue_state["running"] and idle and not queue_order:
                    queue_state["running"] = False  # drained
                if queue_state["running"] and idle and queue_order:
                    break
                _jobs_changed.wait(timeout=1.0)
            job_id = queue_order.pop(0)
            job_store.set_queue(queue_order)
            job = jobs[job_id]
            job["status"] = "pending"
            _sync_job(job_id)
            req = job.pop("_request", None)
            if req is None and job.get("request"):
                req = LaunchRequest(**job["request"])
        run_experiment_task(job_id, req)


@app.get("/api/queue")
def queue_status():
    """The active job(s), the queued jobs in the order they will run, and the ten most recent finished jobs."""
    with _jobs_lock:
        active = [_public(j) for j in jobs.values() if j.get("status") in ACTIVE_STATUSES]
        queued = [dict(_public(jobs[job_id]), position=n) for n, job_id in enumerate(queue_order) if job_id in jobs]
        finished = sorted(
            (j for j in jobs.values() if j.get("status") in TERMINAL_STATUSES),
            key=lambda j: j.get("finished") or 0,
            reverse=True,
        )[:10]
        return {
            "running": queue_state["running"],
            "active": active,
            "queued": queued,
            "finished": [_public(j) for j in finished],
        }


@app.post("/api/queue/start")
def queue_start():
    """Start running queued jobs, one at a time, in order. The queue pauses itself when it drains."""
    with _jobs_changed:
        queue_state["running"] = bool(queue_order)  # starting an empty queue leaves it paused
        if queue_state["running"]:
            _ensure_queue_worker()
        _jobs_changed.notify_all()
        return {"running": queue_state["running"], "queued": len(queue_order)}


@app.post("/api/queue/pause")
def queue_pause():
    """Stop starting queued jobs. A job that is already training keeps running."""
    with _jobs_changed:
        queue_state["running"] = False
        _jobs_changed.notify_all()
        return {"running": False, "queued": len(queue_order)}


@app.post("/api/queue/{job_id}/move")
def queue_move(job_id: str, req: MoveRequest):
    with _jobs_changed:
        if job_id not in queue_order:
            raise HTTPException(status_code=409, detail="Only queued jobs can be moved.")
        queue_order.remove(job_id)
        queue_order.insert(max(0, min(req.position, len(queue_order))), job_id)
        job_store.set_queue(queue_order)
        _jobs_changed.notify_all()
        return {"job_id": job_id, "position": queue_order.index(job_id)}


@app.get("/api/experiments/jobs")
def list_jobs():
    with _jobs_lock:
        return {"jobs": [_public(job) for job in jobs.values()]}


@app.get("/api/experiments/{job_id}/status")
def check_status(job_id: str, since: int = 0):
    """Job state plus log lines after line number `since` (0-based, counted from job start)."""
    with _jobs_lock:
        if job_id not in jobs:
            stored = job_store.get_job(job_id)
            if stored:
                jobs[job_id] = stored
            else:
                raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        log = job.get("_log", [])
        dropped = job.get("log_dropped", 0)
        job_info = _public(job)
        job_info["log"] = log[max(since - dropped, 0) :]
        job_info["log_total"] = dropped + len(log)
        job_info["stdout"] = "\n".join(log[-40:])[-1000:]
    return job_info


@app.get("/api/experiments/{job_id}/metrics")
def job_metrics(job_id: str, since: int = 0, since_byte: int = 0):
    """Numeric metrics rows for each agent of a job, read live from Lightning's metrics.csv.

    Rows after index `since` are returned. When a file was rewritten with fewer rows (Lightning
    rewrites it when new columns appear) `reset` is true and all rows are returned.
    """
    with _jobs_lock:
        if job_id not in jobs:
            stored = job_store.get_job(job_id)
            if stored:
                jobs[job_id] = stored
            else:
                raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id]
    agents = {}
    for agent in job.get("agents", []):
        path = _latest_metrics_csv(PROJECT_ROOT / "results" / "logs" / job["group"] / job["experiment_id"] / agent)
        if since_byte > 0 and path:
            rows, next_byte, reset = _read_metrics_incremental(path, last_byte=since_byte)
            agents[agent] = {
                "source": str(path),
                "total": len(rows),
                "next_byte": next_byte,
                "reset": reset,
                "rows": rows,
            }
        else:
            rows = _read_metrics_rows(path) if path else []
            reset = since > len(rows)
            agents[agent] = {
                "source": str(path) if path else None,
                "total": len(rows),
                "reset": reset,
                "rows": rows if reset else rows[since:],
            }
    return {"job_id": job_id, "status": job.get("status"), "agents": agents}


_csv_headers: dict[str, list[str]] = {}


def _read_metrics_incremental(path: Path, last_byte: int = 0) -> tuple[list[dict[str, float]], int, bool]:
    """Read newly appended lines from a metrics CSV starting at last_byte offset.

    Returns (rows, next_byte_offset, reset_occurred).
    """
    if not path.is_file():
        return [], 0, False

    try:
        current_size = path.stat().st_size
    except OSError:
        return [], 0, False

    reset = False
    if current_size < last_byte:
        last_byte = 0
        reset = True

    try:
        with open(path, "rb") as f:
            path_key = str(path)
            if last_byte == 0 or path_key not in _csv_headers:
                header_line = f.readline().decode("utf-8", errors="replace")
                if not header_line.endswith("\n"):
                    return [], 0, False  # header still being written
                reader = csv.reader([header_line.rstrip("\r\n")])
                header_fields = next(reader, [])
                _csv_headers[path_key] = header_fields
                last_byte = f.tell()
            else:
                f.seek(last_byte)

            header_fields = _csv_headers.get(path_key, [])
            if not header_fields:
                return [], 0, False

            raw = f.read()
            if not raw:
                return [], last_byte, reset

            last_nl = raw.rfind(b"\n")
            if last_nl == -1:
                return [], last_byte, reset

            valid_chunk = raw[: last_nl + 1]
            next_byte = last_byte + len(valid_chunk)

            text = valid_chunk.decode("utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]

            rows = []
            for line in lines:
                reader = csv.reader([line])
                vals = next(reader, [])
                numeric: dict[str, float] = {}
                for key, val in zip(header_fields, vals):
                    if key and val not in (None, ""):
                        try:
                            numeric[key] = float(val)
                        except ValueError:
                            pass
                if numeric:
                    rows.append(numeric)

            return rows, next_byte, reset
    except OSError:
        return [], last_byte, reset


def _latest_metrics_csv(agent_dir: Path) -> Path | None:
    candidates = []
    for path in agent_dir.glob("version_*/metrics.csv"):
        suffix = path.parent.name.removeprefix("version_")
        candidates.append((int(suffix) if suffix.isdigit() else -1, path))
    return max(candidates)[1] if candidates else None


def _read_metrics_rows(path: Path) -> list[dict[str, float]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = text.split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]  # the last line is still being written
    rows = []
    for row in csv.DictReader(lines):
        numeric = {}
        for key, value in row.items():
            if key and value not in (None, ""):
                try:
                    numeric[key] = float(value)
                except ValueError:
                    pass
        rows.append(numeric)
    return rows


@app.get("/api/experiments/{job_id}/telemetry")
def job_telemetry(job_id: str, since_log: int = 0, since_byte: int = 0):
    """Unified telemetry: job state, incremental log lines, incremental metrics, and hardware stats."""
    with _jobs_lock:
        if job_id not in jobs:
            stored = job_store.get_job(job_id)
            if stored:
                jobs[job_id] = stored
            else:
                raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        log = job.get("_log", [])
        dropped = job.get("log_dropped", 0)
        job_info = _public(job)
        job_info["log"] = log[max(since_log - dropped, 0) :]
        job_info["log_total"] = dropped + len(log)
        job_info["stdout"] = "\n".join(log[-40:])[-1000:]

    # Hardware stats
    pid = job.get("pid")
    hw: dict[str, Any] = {}
    if pid and psutil is not None:
        try:
            proc = psutil.Process(pid)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                hw["cpu_percent"] = proc.cpu_percent()
                mem = proc.memory_info()
                hw["memory_mb"] = round(mem.rss / (1024 * 1024), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Incremental metrics for each agent
    agents = {}
    for agent in job.get("agents", []):
        path = _latest_metrics_csv(PROJECT_ROOT / "results" / "logs" / job["group"] / job["experiment_id"] / agent)
        if path:
            rows, next_byte, reset = _read_metrics_incremental(path, last_byte=since_byte)
            agents[agent] = {
                "source": str(path),
                "rows": rows,
                "next_byte": next_byte,
                "reset": reset,
            }
        else:
            agents[agent] = {"source": None, "rows": [], "next_byte": 0, "reset": False}

    return {
        "job_id": job_id,
        "job": job_info,
        "hardware": hw,
        "agents": agents,
    }


@app.post("/api/experiments/{job_id}/cancel")
def cancel_experiment(job_id: str):
    if job_id not in jobs:
        stored = job_store.get_job(job_id)
        if stored:
            jobs[job_id] = stored
        else:
            raise HTTPException(status_code=404, detail="Job not found")

    with _jobs_changed:
        job = jobs[job_id]
        if job.get("status") == "queued":  # never started: take it out of the queue
            if job_id in queue_order:
                queue_order.remove(job_id)
                job_store.set_queue(queue_order)
            job.pop("_request", None)
            job.update(status="cancelled", finished=time.time())
            _sync_job(job_id)
            _jobs_changed.notify_all()
            return {"job_id": job_id, "status": "cancelled"}
        if job.get("status") not in ("pending", "running"):
            return {"status": job.get("status"), "message": f"Job is already {job.get('status')}"}
        job["status"] = "cancelled"
        _sync_job(job_id)
        process = job.get("_process")

    if process and process.poll() is None:
        _kill_process_tree(process.pid)
    return {"job_id": job_id, "status": "cancelled"}


# ── TensorBoard ──────────────────────────────────────────────────────────────
# One local TensorBoard server over results/tensorboard/ (runs launched with tensorboard=true),
# managed here so GUI clients need no TensorBoard install of their own.

TENSORBOARD_DIR = Path("results/tensorboard")
_tensorboard: dict[str, Any] = {}
_tensorboard_lock = threading.Lock()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tensorboard_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "data/environment", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def _tensorboard_state() -> dict[str, Any]:
    process = _tensorboard.get("process")
    running = process is not None and process.poll() is None
    state: dict[str, Any] = {
        "available": importlib.util.find_spec("tensorboard") is not None,
        "running": running,
        "ready": False,
        "url": None,
        "logdir": TENSORBOARD_DIR.as_posix(),
    }
    if running:
        state.update(url=_tensorboard["url"], pid=process.pid, ready=_tensorboard_ready(_tensorboard["url"]))
    elif process is not None and not _tensorboard.get("stopped"):  # it exited on its own
        state["exit_code"] = process.returncode
        state["error"] = (
            _tensorboard.get("log_path")
            and Path(_tensorboard["log_path"]).read_text(encoding="utf-8", errors="replace")[-1500:]
        )
    return state


@app.get("/api/tensorboard")
def tensorboard_status():
    """Whether the TensorBoard server runs and answers. `ready` turns true about a second after start."""
    with _tensorboard_lock:
        return _tensorboard_state()


@app.post("/api/tensorboard/start")
def tensorboard_start():
    """Start the TensorBoard server if it is not running. Returns at once; poll GET until `ready`."""
    with _tensorboard_lock:
        state = _tensorboard_state()
        if state["running"]:
            return state
        if not state["available"]:
            raise HTTPException(status_code=503, detail="TensorBoard is not installed in the backend environment.")
        logdir = PROJECT_ROOT / TENSORBOARD_DIR
        logdir.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        port = _free_port()
        log_path = JOBS_DIR / "tensorboard.log"
        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--reload_interval",
            "5",
        ]
        popen_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, **popen_kwargs)
        _tensorboard.update(process=process, url=f"http://127.0.0.1:{port}/", log_path=str(log_path), stopped=False)
        return _tensorboard_state()


@app.post("/api/tensorboard/stop")
def tensorboard_stop():
    with _tensorboard_lock:
        _stop_tensorboard()
        return _tensorboard_state()


def _stop_tensorboard() -> None:
    process = _tensorboard.get("process")
    _tensorboard["stopped"] = True
    if process is not None and process.poll() is None:
        _kill_process_tree(process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


atexit.register(_stop_tensorboard)  # do not leave the server behind when the API exits
