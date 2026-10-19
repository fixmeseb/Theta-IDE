"""Reads the Hydra config tree and answers experiment-compatibility questions.

Pure Python: no Qt and no torch imports, so it loads in headless CI and can be
unit tested without a display or the ML stack.

The panel asks this module three things: which options are valid for a given
paradigm, which form fields that paradigm permits, and what command line a
selection turns into.
"""

from __future__ import annotations

import copy
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Paradigms opt into fields with an `allows_<flag>` key whose name does not
# always match the Hydra key the field maps to.
_ALLOW_FLAGS = {
    "eval_episodes": "allows_eval_episodes",
}

_GROUPS = ("paradigms", "env", "agent", "model", "site")


# path -> (file bytes, parsed value). The tree is rebuilt often (every save invalidates it) but
# usually only one file changed, and parsing is what costs: reading the bytes is ~100x cheaper.
# Keyed on the exact contents, not mtime, so two quick same-size writes can't return stale data.
_PARSED: dict[Path, tuple[bytes, Any]] = {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        cached = _PARSED.get(path)
        if cached is not None and cached[0] == data:
            parsed = cached[1]
        else:
            parsed = yaml.safe_load(data) or {}
            _PARSED[path] = (data, parsed)
    except (OSError, yaml.YAMLError):
        return {}
    return copy.deepcopy(parsed)  # callers may mutate what they get; the cache must not change


def config_fingerprint(config_root: Path) -> int:
    """A hash of every YAML file under `config_root` (paths and contents): equal means unchanged."""
    entries = []
    for path in sorted(Path(config_root).rglob("*.yaml")):
        try:
            entries.append((str(path), path.read_bytes()))
        except OSError:
            continue
    return hash(tuple(entries))


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*.yaml")
        if not p.name.startswith("_") and not any(part.startswith(".") for part in p.parts)
    )


@dataclass(frozen=True)
class Environment:
    name: str
    compatible_paradigms: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Environment:
        raw = _read_yaml(path)
        rel_name = path.stem
        if root is not None:
            try:
                rel_name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                rel_name = path.stem

        env_name = raw.get("name")
        if not env_name and isinstance(raw.get("env"), dict):
            env_name = raw["env"].get("name")

        if env_name and "/" in env_name:
            chosen_name = env_name
        elif "/" in rel_name:
            chosen_name = rel_name
        else:
            chosen_name = env_name or rel_name

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if not paradigms and isinstance(raw.get("env"), dict):
            paradigms = raw["env"].get("paradigms") or raw["env"].get("compatible_paradigms")

        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            cp = ()

        return cls(
            name=chosen_name,
            compatible_paradigms=cp,
            raw=raw,
        )


@dataclass(frozen=True)
class Agent:
    name: str
    algorithm: str
    compatible_paradigms: tuple[str, ...] = ()
    hyperparameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Agent:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            engine = str(raw.get("engine", "")).lower()
            if engine == "sb3" or name.startswith("sb3/"):
                cp = ("online_rl",)
            elif engine == "d3rlpy" or name.startswith("d3rlpy/"):
                cp = ("offline_rl",)
            else:
                cp = ()

        return cls(
            name=name,
            algorithm=raw.get("algorithm") or path.stem,
            compatible_paradigms=cp,
            hyperparameters={k: v for k, v in raw.items() if k not in ("algorithm", "paradigms", "compatible_paradigms")},
            raw=raw,
        )


@dataclass(frozen=True)
class Model:
    name: str
    architecture: str
    compatible_paradigms: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Model:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            cp = ()

        return cls(
            name=name,
            architecture=raw.get("architecture") or path.stem,
            compatible_paradigms=cp,
            parameters={k: v for k, v in raw.items() if k not in ("architecture", "paradigms", "compatible_paradigms")},
            raw=raw,
        )

    @property
    def is_composite(self) -> bool:
        return bool(self.subcomponents) or "slots" in self.parameters

    @property
    def subcomponents(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.parameters.items() if isinstance(v, dict)}

    @property
    def slots(self) -> list[str]:
        if "slots" in self.parameters and isinstance(self.parameters["slots"], (list, tuple)):
            return list(self.parameters["slots"])
        return list(self.subcomponents.keys())


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    path: Path
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path) -> Experiment:
        relative = path.relative_to(root)
        if relative.parts[0] == "experiment" and len(relative.parts) > 2:
            group = relative.parts[1]
        elif len(relative.parts) > 1:
            group = relative.parts[0]
        else:
            group = relative.parent.name or "ungrouped"

        exp_name = relative.with_suffix("").as_posix()
        if exp_name.startswith("experiment/"):
            exp_name = exp_name[len("experiment/"):]

        return cls(
            name=exp_name,
            group=group,
            path=path,
            raw=_read_yaml(path),
        )


@dataclass(frozen=True)
class Paradigm:
    name: str
    description: str
    allowed_agents: tuple[str, ...]
    forbidden_agents: tuple[str, ...]
    requires: Mapping[str, Any] = field(default_factory=dict, repr=False)
    forbids: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Paradigm:
        raw = _read_yaml(path)
        constraints = raw.get("constraints") or {}
        return cls(
            name=path.stem,
            description=raw.get("description", ""),
            allowed_agents=tuple(raw.get("allowed_agents") or ()),
            forbidden_agents=tuple(raw.get("forbidden_agents") or ()),
            requires=constraints.get("requires") or {},
            forbids=constraints.get("forbids") or {},
            raw=raw,
        )

    def permits_agent(self, agent: Agent | str) -> bool:
        algo = (agent.algorithm if isinstance(agent, Agent) else str(agent)).lower()
        name = (agent.name if isinstance(agent, Agent) else str(agent)).lower()
        clean = (algo, name, name.split("/")[-1], algo.split("/")[-1])
        if any(
            f.lower() in clean
            or name.startswith(f.lower() + "/")
            or name.endswith("/" + f.lower())
            for f in self.forbidden_agents
        ):
            return False

        if isinstance(agent, Agent) and agent.compatible_paradigms:
            return self.name in agent.compatible_paradigms

        # Fallback for agents without explicit compatible_paradigms
        if "allowed_agents" in self.raw and not self.allowed_agents:
            # Explicitly declared empty allow list (e.g. supervised)
            return False

        if self.allowed_agents:
            for allowed in self.allowed_agents:
                al = allowed.lower()
                if al in (algo, name) or name.startswith(al + "/") or algo.startswith(al + "_") or name.endswith("/" + al):
                    return True
            return False

        return True

    def permits_model(self, model: Model | str) -> bool:
        if isinstance(model, Model) and model.compatible_paradigms:
            return self.name in model.compatible_paradigms
        return True

    def permits_environment(self, environment: Environment) -> bool:
        if not environment.compatible_paradigms:
            return True
        return self.name in environment.compatible_paradigms

    def field_enabled(self, name: str) -> bool:
        if name in self.forbids:
            return False
        flag = _ALLOW_FLAGS.get(name)
        if flag is not None:
            # These are opt-in: validation.py reads paradigm_def.get(flag, False),
            # so a paradigm that does not declare the flag forbids the field.
            return bool(self.raw.get(flag, False))
        return True

    def disabled_reason(self, name: str) -> str | None:
        if self.field_enabled(name):
            return None
        if name in self.forbids:
            return f"{self.name} forbids {name} ({self.forbids[name]})"
        return f"{self.name} does not enable {name}"


class ConfigTree:
    """The `in/config/` directory, parsed into queryable objects."""

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self._experiment_groups: dict[str, ExperimentGroup] | None = None
        self.paradigms = {p.stem: Paradigm.from_file(p) for p in _yaml_files(self.config_root / "paradigms")}
        agent_dir = self.config_root / "agent"
        self.agents = {}
        for a in _yaml_files(agent_dir):
            rel_name = a.relative_to(agent_dir).with_suffix("").as_posix()
            agent_obj = Agent.from_file(a, agent_dir)
            self.agents[rel_name] = agent_obj
            if a.stem not in self.agents:
                self.agents[a.stem] = agent_obj

        model_dir = self.config_root / "model"
        self.models = {}
        for m in _yaml_files(model_dir):
            rel_name = m.relative_to(model_dir).with_suffix("").as_posix()
            model_obj = Model.from_file(m, model_dir)
            self.models[rel_name] = model_obj
            if m.stem not in self.models:
                self.models[m.stem] = model_obj

        env_dir = self.config_root / "env"
        self.environments = {}
        for e in _yaml_files(env_dir):
            rel_name = e.relative_to(env_dir).with_suffix("").as_posix()
            env_obj = Environment.from_file(e, env_dir)
            self.environments[rel_name] = env_obj
            if e.stem not in self.environments:
                self.environments[e.stem] = env_obj

        self.sites = tuple(s.stem for s in _yaml_files(self.config_root / "site"))

        self._discover_plugin_manifests()

        experiment_root = self.config_root / "experiment"
        self.experiments: dict[str, Experiment] = {}
        if experiment_root.is_dir():
            for path in sorted(experiment_root.glob("**/*.yaml")):
                if path.name.startswith("_") or path.parent.name == "methods" or "/methods/" in path.as_posix():
                    continue
                experiment = Experiment.from_file(path, experiment_root)
                self.experiments[experiment.name] = experiment

    def _discover_plugin_manifests(self) -> None:
        """Scan workspace for installed plugin manifests (.theta_component.json / plugin.json)
        and register them into self.agents, self.models, and self.environments if not already present.
        """
        import json

        # Infer workspace root
        if self.config_root.name == "config" and self.config_root.parent.name == "in":
            ws_root = self.config_root.parent.parent
        else:
            ws_root = self.config_root.parent

        scan_dirs = [
            (ws_root / "src" / "usr" / "algorithms", "agent"),
            (ws_root / "src" / "usr" / "methods", "agent"),
            (ws_root / "src" / "usr" / "models", "model"),
            (ws_root / "in" / "envs", "env"),
        ]

        for sdir, default_kind in scan_dirs:
            if not sdir.is_dir():
                continue
            for meta_file in sorted(list(sdir.glob("*/.theta_component.json")) + list(sdir.glob("*/plugin.json"))):
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    kind = str(meta.get("kind", default_kind)).lower()
                    comp_id = str(meta.get("id") or meta_file.parent.name)
                    paradigms = meta.get("paradigms") or meta.get("compatible_paradigms") or ()
                    if isinstance(paradigms, list):
                        paradigms_tuple = tuple(str(p) for p in paradigms)
                    elif isinstance(paradigms, str):
                        paradigms_tuple = (paradigms,)
                    else:
                        paradigms_tuple = ()

                    if kind in ("agent", "algorithm") and comp_id not in self.agents:
                        self.agents[comp_id] = Agent(
                            name=comp_id,
                            algorithm=str(meta.get("algorithm") or comp_id),
                            compatible_paradigms=paradigms_tuple,
                            hyperparameters=dict(meta.get("hyperparameters", {})),
                            raw=meta,
                        )
                    elif kind == "model" and comp_id not in self.models:
                        self.models[comp_id] = Model(
                            name=comp_id,
                            architecture=str(meta.get("architecture") or comp_id),
                            compatible_paradigms=paradigms_tuple,
                            parameters=dict(meta.get("parameters", {})),
                            raw=meta,
                        )
                    elif kind == "env" and comp_id not in self.environments:
                        self.environments[comp_id] = Environment(
                            name=comp_id,
                            compatible_paradigms=paradigms_tuple,
                            raw=meta,
                        )
                except Exception:
                    continue

    def create_folder(self, group: str, subpath: str) -> Path:
        """Create a new organizational folder under an experiment group."""
        folder = self.config_root / "experiment" / group / subpath
        folder.mkdir(parents=True, exist_ok=True)
        self.refresh_groups()
        return folder

    def rename_folder(self, old_path: Path | str, new_name: str) -> Path:
        """Rename an organizational folder under an experiment group."""
        old_p = Path(old_path)
        new_p = old_p.with_name(new_name)
        if new_p.exists() and not old_p.samefile(new_p):
            raise FileExistsError(f"A folder named '{new_name}' already exists.")
        old_p.rename(new_p)
        self.refresh_groups()
        return new_p

    def delete_folder(self, folder_path: Path | str) -> bool:
        """Delete an organizational folder under an experiment group."""
        import shutil
        folder_p = Path(folder_path)
        if folder_p.is_dir():
            shutil.rmtree(folder_p)
            self.refresh_groups()
            return True
        return False

    @classmethod
    def discover(cls, start: Path | None = None) -> ConfigTree:
        """Walk upward from `start` looking for an `in/config` directory."""
        current = Path(start or Path(__file__).resolve().parent)
        for candidate in (current, *current.parents):
            config_root = candidate / "in" / "config"
            if config_root.is_dir():
                return cls(config_root)
        raise FileNotFoundError(f"No in/config directory found above {current}")

    def agents_for(self, paradigm: str) -> list[Agent]:
        if paradigm not in self.paradigms:
            return []
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Agent] = []
        for a in self.agents.values():
            if id(a) not in seen:
                seen.add(id(a))
                if rules.permits_agent(a):
                    result.append(a)
        return result

    def environments_for(self, paradigm: str) -> list[Environment]:
        if paradigm not in self.paradigms:
            return []
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Environment] = []
        for e in self.environments.values():
            if id(e) not in seen:
                seen.add(id(e))
                if rules.permits_environment(e):
                    result.append(e)
        return result

    def models_for(self, paradigm: str) -> list[Model]:
        if paradigm not in self.paradigms:
            return list(self.models.values())
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Model] = []
        for m in self.models.values():
            if id(m) not in seen:
                seen.add(id(m))
                if rules.permits_model(m):
                    result.append(m)
        return result

    def field_enabled(self, paradigm: str, name: str) -> bool:
        if paradigm not in self.paradigms:
            return True
        return self.paradigms[paradigm].field_enabled(name)

    def experiments_in(self, group: str) -> list[Experiment]:
        return [e for e in self.experiments.values() if e.group == group]

    @property
    def groups(self) -> list[str]:
        return sorted({e.group for e in self.experiments.values()} | set(self.experiment_groups))

    @property
    def experiment_groups(self) -> dict[str, ExperimentGroup]:
        # Built once per tree: group() alone reads it up to three times. Folder edits through this
        # tree reset it; anything else that changes in/config invalidates the whole tree.
        if self._experiment_groups is None:
            root = self.config_root / "experiment"
            self._experiment_groups = (
                {d.name: ExperimentGroup.from_dir(d) for d in sorted(root.iterdir()) if d.is_dir()}
                if root.is_dir() else {}
            )
        return self._experiment_groups

    def refresh_groups(self) -> None:
        """Re-read the experiment groups on next use (after writing a group's folder or _base.yaml)."""
        self._experiment_groups = None

    def group(self, name: str) -> ExperimentGroup:
        if not name:
            return ExperimentGroup(name="")
        if name in self.experiment_groups:
            return self.experiment_groups[name]
        for k, g in self.experiment_groups.items():
            if k.lower() == name.lower():
                return g
        return ExperimentGroup(name=name)

    def agent_hyperparameters(self, agent_name: str | None) -> dict[str, Any]:
        if not agent_name or agent_name not in self.agents:
            return {}
        return dict(self.agents[agent_name].hyperparameters)

    def model_parameters(self, model_name: str | None) -> dict[str, Any]:
        if not model_name or model_name not in self.models:
            return {}
        return dict(self.models[model_name].parameters)

    def model_subcomponents(self, model_name: str | None) -> dict[str, dict[str, Any]]:
        if not model_name or model_name not in self.models:
            return {}
        return {
            k: dict(v)
            for k, v in self.models[model_name].parameters.items()
            if isinstance(v, dict)
        }

    def model_slots(self, model_name: str | None) -> list[str]:
        if not model_name or model_name not in self.models:
            return []
        return self.models[model_name].slots


@dataclass(frozen=True)
class ExperimentGroup:
    """A directory under in/config/experiment/, described by its _base.yaml.

    The base is what binds an environment and a paradigm, so every experiment in
    the group inherits them; individual experiments only choose methods and a
    training budget.
    """

    name: str
    paradigm: str | None = None
    env: str | None = None
    has_base: bool = False
    base_methods: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, directory: Path) -> ExperimentGroup:
        base = directory / "_base.yaml"
        if not base.is_file():
            return cls(name=directory.name)
        raw = _read_yaml(base)
        env = raw.get("env")
        for entry in raw.get("defaults") or []:
            if isinstance(entry, dict):
                env = entry.get("override /env", env)
        methods = raw.get("methods")
        return cls(
            name=directory.name,
            paradigm=raw.get("paradigm"),
            env=env,
            has_base=True,
            base_methods=methods if isinstance(methods, dict) else {},
        )


def default_method_model(
    group: ExperimentGroup | None,
    known: Any = (),
    fallback: str = "dnn",
    paradigm: Paradigm | None = None,
) -> str:
    """The architecture a new agent-less method should name.

    A group whose base already declares methods has settled on an architecture
    (early_prediction uses lstm and transformer), so follow that rather than
    impose a default on a group that has already chosen.

    Failing that, the paradigm's own `default_model`. This matters because the
    bare fallback is `dnn`, which is MLPQNetwork - an RL Q-network built around
    n_actions. Offering it for a supervised or unsupervised group points at the
    one model whose training is certain to fail.
    """
    for spec in (group.base_methods if group else {}).values():
        if isinstance(spec, dict) and isinstance(spec.get("model"), str):
            return spec["model"]
    names = set(known or ())
    declared = paradigm.raw.get("default_model") if paradigm is not None else None
    if isinstance(declared, str) and (not names or declared in names):
        return declared
    if not names or fallback in names:
        return fallback
    return sorted(names)[0]


def group_base_yaml(env: str | None, paradigm: str, rules: Paradigm | None = None, seed: int = 1) -> str:
    """The _base.yaml for a new group, which is what binds env and paradigm."""
    lines = [
        "# @package _global_",
    ]
    if env and str(env).strip().lower() not in ("", "none"):
        lines += [
            "defaults:",
            f"  - override /env: {env}",
            "",
        ]
    else:
        lines += [
            "defaults: []",
            "",
        ]
    lines += [
        f"paradigm: {paradigm}",
        "",
    ]
    if rules is not None and not rules.field_enabled("eval_episodes"):
        lines.append("eval_episodes: 0")
    lines += [f"seed: {seed}", "save_dataset: false", "recover: false"]
    return "\n".join(lines) + "\n"


def experiment_yaml(
    group: str,
    experiment_id: str,
    paradigm: Paradigm | None = None,
    agent: str | None = None,
    model: str | None = None,
    seed: int = 42,
    total_timesteps: int = 10000,
) -> str:
    """A valid experiment for `group`, omitting whatever its paradigm forbids."""
    lines = [
        "# @package _global_",
        "defaults:",
        f"  - {group}/_base",
        "",
        f"experiment_id: {experiment_id}",
        f"seed: {seed}",
        f"total_timesteps: {total_timesteps}",
    ]
    if paradigm is None or paradigm.field_enabled("eval_episodes"):
        lines.append("eval_episodes: 100")

    # A paradigm with no agents still needs a method, so a model on its own is
    # enough to write one: supervised groups name methods for the architecture.
    if agent or model:
        model = model or "dnn"
        lines += ["", "methods:", f"  {unique_method_name((), agent, model)}:"]
        if agent:
            lines.append(f"    agent: {agent}")
        lines.append(f"    model: {model}")
    return "\n".join(lines) + "\n"


@dataclass
class Selection:
    """A chosen experiment plus the overrides the panel will apply to it."""

    experiment: str
    paradigm: str | None = None
    environment: str | None = None
    agent: str | None = None
    model: str | None = None
    site: str = "local"
    experiment_id: str | None = None
    seed: int | None = None
    total_timesteps: int | None = None
    eval_episodes: int | None = None
    extra: list[str] = field(default_factory=list)

    _GROUP_FIELDS = (
        ("paradigm", "paradigm"),
        ("environment", "env"),
        ("agent", "agent"),
        ("model", "model"),
        ("site", "site"),
    )
    _SCALAR_FIELDS = (
        ("experiment_id", "experiment_id"),
        ("seed", "seed"),
        ("total_timesteps", "total_timesteps"),
        ("eval_episodes", "eval_episodes"),
    )

    def to_overrides(self, tree: ConfigTree | None = None) -> list[str]:
        """Render the selection as Hydra `key=value` strings.

        Passing `tree` drops overrides the chosen paradigm forbids, so the panel
        cannot submit a combination the pipeline would reject at startup.
        """
        overrides: list[str] = []
        for attribute, key in self._GROUP_FIELDS:
            value = getattr(self, attribute)
            if value is not None:
                overrides.append(f"{key}={value}")
        for attribute, key in self._SCALAR_FIELDS:
            value = getattr(self, attribute)
            if value is None:
                continue
            if tree is not None and self.paradigm and not tree.field_enabled(self.paradigm, key):
                continue
            overrides.append(f"{key}={value}")
        overrides.extend(self.extra)
        return overrides

    def command(
        self,
        tree: ConfigTree | None = None,
        python: str = "python",
        script: str = "run_pipeline.py",
        dry_run: bool = False,
    ) -> list[str]:
        argv = [python, script, self.experiment, *self.to_overrides(tree)]
        if dry_run:
            argv.append("dry_run=true")
        return argv

    def command_line(self, **kwargs: Any) -> str:
        return shlex.join(self.command(**kwargs))


def unique_method_name(existing: Any, agent: str | None, model: str) -> str:
    """A method name free in `existing`, following the <agent>_<model> convention.

    The RL configs name methods after the agent they run, sometimes with a
    suffix (ppo_dnn, ppo_blendrl, iql_blendrl_arch1), so a collision gets a
    numeric suffix rather than a new scheme. Supervised methods have no agent
    at all (ep_lstm, ep_transformer), so they are named for the model.
    """
    base = f"{agent}_{model}" if agent and model else (agent or model or "method")
    taken = set(existing or ())
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def add_method(
    data: dict[str, Any],
    agent: str | None = None,
    model: str | None = None,
    paradigm: Paradigm | None = None,
) -> str:
    """Add a method to an experiment's `methods` block. Returns its name.

    `agent` is omitted for paradigms that declare none: supervised learning has
    no policy to choose actions with, so its methods name an architecture only.
    """
    if model is None:
        model = (paradigm.raw.get("default_model") if paradigm else None) or "dnn"
    methods = data.setdefault("methods", {})
    name = unique_method_name(methods, agent, model)
    methods[name] = {"agent": agent, "model": model} if agent else {"model": model}
    return name


def remove_method(data: dict[str, Any], name: str) -> bool:
    """Drop a method. Returns False if it was not there.

    `params` holds settings shared by every method rather than a method of its
    own, so it is never removable this way.
    """
    methods = data.get("methods")
    if not isinstance(methods, dict) or name == "params" or name not in methods:
        return False
    del methods[name]
    if not [k for k in methods if k != "params"]:
        # A methods block with only `params` left satisfies no paradigm, and an
        # absent one lets the group base supply its own.
        data.pop("methods", None)
    return True


def deletion_blocked_reason(path: Path) -> str | None:
    """Why deleting this config would break something, or None if it is safe.

    A group's _base.yaml binds the environment and paradigm every experiment in
    the group inherits. Removing one while experiments remain leaves them
    pointing at nothing, and Hydra cannot even load the result:
    "Could not load 'experiment/<group>/_base'".
    """
    path = Path(path)
    if not path.exists():
        return f"{path.name} no longer exists"
    if path.name != "_base.yaml":
        return None
    siblings = [p for p in path.parent.glob("*.yaml") if p.name != "_base.yaml"]
    if siblings:
        names = ", ".join(sorted(p.stem for p in siblings)[:4])
        return (
            f"{path.parent.name}/_base.yaml defines the environment and paradigm for "
            f"{len(siblings)} experiment(s) that inherit it ({names}). "
            f"Delete those first."
        )
    return None


def rename_experiment_text(text: str, new_id: str) -> str:
    """The file's contents with experiment_id updated to match a new filename.

    Rewriting the one line rather than round-tripping through yaml keeps the
    comments and key order the rest of the team wrote.
    """
    pattern = re.compile(r"^(\s*experiment_id\s*:\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{new_id}", text, count=1)
    return text
