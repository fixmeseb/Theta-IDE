"""Portable demo records. This module does not import the training backend."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from uuid import uuid4

# Backend experiment every builder config is composed from (in/config/experiment/thetaide/_base.yaml).
BASE_EXPERIMENT = "thetaide/_base"
METHOD, AGENT, MODEL = "ppo", "ppo", "dnn"


def _num(value):
    """Format a float so both the Hydra override grammar and YAML read it back as a float."""
    text = repr(float(value))
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    return text


@dataclass
class Config:
    name: str = "cartpole_ppo_baseline"
    env: str = "cartpole"
    seed: int = 42
    total_timesteps: int = 10000
    lr: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    tensorboard: bool = True

    def overrides(self):
        """Hydra overrides applied to BASE_EXPERIMENT, exactly as typed after `run_pipeline.py`."""
        method = f"++methods.{METHOD}"
        return [f"++experiment_id='{self.name}'", f"seed={self.seed}", f"total_timesteps={self.total_timesteps}",
                f"{method}.agent={AGENT}", f"{method}.model={MODEL}", f"{method}.lr={_num(self.lr)}",
                f"{method}.batch_size={self.batch_size}", f"{method}.gamma={_num(self.gamma)}",
                f"tensorboard={str(self.tensorboard).lower()}"]

    def command(self):
        return " ".join(["python", "run_pipeline.py", BASE_EXPERIMENT, *self.overrides()])

    def recipe_yaml(self):
        """A standalone experiment recipe equivalent to BASE_EXPERIMENT + overrides()."""
        return (
            "# @package _global_\n"
            "# Exported from ThetaIDE. Save under in/config/experiment/thetaide/ and run:\n"
            "#   python run_pipeline.py thetaide/<file name without .yaml>\n"
            f"defaults:\n- {BASE_EXPERIMENT}\n\n"
            f"experiment_id: {json.dumps(self.name)}\n"
            f"seed: {self.seed}\ntotal_timesteps: {self.total_timesteps}\n"
            f"tensorboard: {str(self.tensorboard).lower()}\n\n"
            f"methods:\n  {METHOD}:\n    agent: {AGENT}\n    model: {MODEL}\n"
            f"    lr: {_num(self.lr)}\n    batch_size: {self.batch_size}\n    gamma: {_num(self.gamma)}\n"
        )


def sample(config, index):
    rng = random.Random(config.seed * 101 + index)
    progress = index / 60
    reward = max(0, min(500, 20 + 475 * (1 - math.exp(-3.6 * progress)) + rng.uniform(-18, 18)))
    return {"step": round(config.total_timesteps * progress),
            "reward": round(reward, 2),
            "loss": round(0.8 * math.exp(-4 * progress) + rng.uniform(0.015, 0.055), 4)}


def new_run(config, simulated=True):
    return {"id": uuid4().hex[:10], "created": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "status": "running", "simulated": simulated,
            "metrics": [], "notes": ""}


# Statuses of a backend training job that is still in progress, and those that are final.
LIVE_STATUSES = ("starting", "running")
FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")


# Run-record metric keys and the metrics.csv columns they come from.
METRIC_COLUMNS = {
    "reward": "eval/reward",
    "reward_std": "eval/reward_std",
    "loss": "losses/total_loss",
    "policy_loss": "losses/policy_loss",
    "value_loss": "losses/value_loss",
    "entropy": "losses/entropy",
    "approx_kl": "losses/approx_kl",
}


def metric_points(rows):
    """Convert Lightning metrics.csv rows into chart points keyed by environment transitions.

    Evaluation rows carry eval/* and training rows carry losses/*; a point keeps None for
    the metrics its row does not report.
    """
    points = []
    for row in rows:
        if "transitions" not in row:
            continue
        point = {key: row.get(column) for key, column in METRIC_COLUMNS.items()}
        if all(value is None for value in point.values()):
            continue
        points.append({"step": round(row["transitions"]), **point})
    return points


def available_metrics(run):
    """Metric keys with at least one value in the run (demo runs only have reward and loss)."""
    return {key for m in run["metrics"] for key, value in m.items() if key != "step" and value is not None}


def run_name(run):
    """The name a run is listed under: the experiment ID the backend wrote results to, or the demo name."""
    return run["config"]["name"] if run["simulated"] else run["backend"]["experiment_id"]


# Format of episode rewards in cards, tables and the run picker.
REWARD_FORMAT = ".1f"
# Chart series colours, in the order they are assigned (legacy palette values the theme maps to its roles).
SERIES_COLORS = ("#b8bb26", "#83a598", "#fabd2f", "#d3869b", "#8ec07c")
BASELINE_COLOR = "#d3869b"  # pinned baseline run


def latest(run, key):
    """The most recent non-missing value of a metric, or None."""
    return next((m[key] for m in reversed(run["metrics"]) if m.get(key) is not None), None)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run):
        folder = self.root / run["id"]
        folder.mkdir(exist_ok=True)
        temporary = folder / "record.tmp"
        temporary.write_text(json.dumps(run, indent=2), encoding="utf-8")
        temporary.replace(folder / "record.json")
        (folder / "config.yaml").write_text(Config(**run["config"]).recipe_yaml(), encoding="utf-8")
        columns = ["step"] + [key for key in METRIC_COLUMNS if any(key in m for m in run["metrics"])]
        cell = lambda value: "" if value is None else value
        rows = [",".join(columns)] + [",".join(str(cell(m.get(key))) for key in columns) for m in run["metrics"]]
        (folder / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def load(self):
        runs, errors = [], []
        for path in self.root.glob("*/record.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                Config(**run["config"])
                if run["id"] != path.parent.name or not isinstance(run["metrics"], list):
                    raise ValueError("Invalid run record")
                for key in ("status", "notes", "created", "simulated"):
                    run[key]
                if run["status"] == "running" and run["simulated"]:
                    run["status"] = "interrupted"  # live runs are reconnected to the backend instead
                runs.append(run)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"Could not load {path}: {exc}")
        return sorted(runs, key=lambda r: r["created"], reverse=True), errors


def example_runs():
    runs = []
    for seed, name in ((42, "cartpole_ppo_baseline"), (7, "cartpole_seed_study")):
        config = Config(name=name, seed=seed)
        run = new_run(config)
        run.update(id=f"example-{seed}", status="example", metrics=[sample(config, i) for i in range(1, 61)],
                   notes="## Hypothesis\nA small learning rate should produce stable improvement.\n\n"
                         "## Next iteration\nCompare a second seed before changing the policy.\n\n"
                         "These are synthetic example metrics, not training results.")
        runs.append(run)
    return runs
