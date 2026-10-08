"""Reads the Hydra config tree and answers experiment-compatibility questions.

Pure Python: no Qt and no torch imports, so it loads in headless CI and can be
unit tested without a display or the ML stack.

The panel asks this module three things: which options are valid for a given
paradigm, which form fields that paradigm permits, and what command line a
selection turns into.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Paradigms opt into fields with an `allows_<flag>` key whose name does not
# always match the Hydra key the field maps to.
_ALLOW_FLAGS = {
    "eval_episodes": "allows_eval_episodes",
    "intervals_count": "allows_intervals",
}

_GROUPS = ("paradigms", "env", "agent", "model", "site")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


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
    offline_only: bool
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
        if env_name and "/" in env_name:
            chosen_name = env_name
        elif "/" in rel_name:
            chosen_name = rel_name
        else:
            chosen_name = env_name or rel_name

        return cls(
            name=chosen_name,
            offline_only=bool(raw.get("offline_only", False)),
            raw=raw,
        )


@dataclass(frozen=True)
class Agent:
    name: str
    algorithm: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Agent:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem
        return cls(
            name=name,
            algorithm=raw.get("algorithm") or path.stem,
            hyperparameters={k: v for k, v in raw.items() if k != "algorithm"},
        )


@dataclass(frozen=True)
class Model:
    name: str
    architecture: str
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Model:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem
        return cls(
            name=name,
            architecture=raw.get("architecture") or path.stem,
            parameters={k: v for k, v in raw.items() if k != "architecture"},
        )


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    path: Path
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path) -> Experiment:
        relative = path.relative_to(root)
        return cls(
            name=relative.with_suffix("").as_posix(),
            group=relative.parent.name or "ungrouped",
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
        name = agent.algorithm if isinstance(agent, Agent) else agent
        if name in self.forbidden_agents:
            return False
        # An empty allow-list means nothing has been declared valid yet, which is
        # the current state of the supervised paradigm.
        return name in self.allowed_agents

    def permits_environment(self, environment: Environment) -> bool:
        required = self.requires.get("env.offline_only")
        if required is None:
            return True
        return bool(required) == environment.offline_only

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

        experiment_root = self.config_root / "experiment"
        self.experiments: dict[str, Experiment] = {}
        if experiment_root.is_dir():
            for path in sorted(experiment_root.glob("**/*.yaml")):
                if path.name.startswith("_"):
                    continue
                experiment = Experiment.from_file(path, experiment_root)
                self.experiments[experiment.name] = experiment

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
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Environment] = []
        for e in self.environments.values():
            if id(e) not in seen:
                seen.add(id(e))
                if rules.permits_environment(e):
                    result.append(e)
        return result

    def field_enabled(self, paradigm: str, name: str) -> bool:
        return self.paradigms[paradigm].field_enabled(name)

    def experiments_in(self, group: str) -> list[Experiment]:
        return [e for e in self.experiments.values() if e.group == group]

    @property
    def groups(self) -> list[str]:
        return sorted({e.group for e in self.experiments.values()} | set(self.experiment_groups))

    @property
    def experiment_groups(self) -> dict[str, ExperimentGroup]:
        root = self.config_root / "experiment"
        if not root.is_dir():
            return {}
        return {d.name: ExperimentGroup.from_dir(d) for d in sorted(root.iterdir()) if d.is_dir()}

    def group(self, name: str) -> ExperimentGroup:
        return self.experiment_groups.get(name) or ExperimentGroup(name=name)


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

    @classmethod
    def from_dir(cls, directory: Path) -> ExperimentGroup:
        base = directory / "_base.yaml"
        if not base.is_file():
            return cls(name=directory.name)
        raw = _read_yaml(base)
        env = None
        for entry in raw.get("defaults") or []:
            if isinstance(entry, dict):
                env = entry.get("override /env", env)
        return cls(name=directory.name, paradigm=raw.get("paradigm"), env=env, has_base=True)


def group_base_yaml(env: str, paradigm: str, rules: Paradigm | None = None, seed: int = 1) -> str:
    """The _base.yaml for a new group, which is what binds env and paradigm.

    in/config/config.yaml defaults intervals_count to 4 and eval_episodes to 100
    for the online case, so a base whose paradigm forbids them must pin them to
    their only legal values - exactly as the existing offline bases do. Leaving
    them to the root defaults makes every experiment in the group invalid.
    """
    lines = [
        "# @package _global_",
        "defaults:",
        f"  - override /env: {env}",
        "",
        f"paradigm: {paradigm}",
        "",
    ]
    if rules is not None and not rules.field_enabled("intervals_count"):
        lines.append("intervals_count: 1")
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
    """A valid experiment for `group`, omitting whatever its paradigm forbids.

    Offline and supervised paradigms reject intervals_count > 1 and non-zero
    eval_episodes, and their group bases already pin both to legal values, so
    the experiment must not restate them.
    """
    lines = [
        "# @package _global_",
        "defaults:",
        f"  - {group}/_base",
        "",
        f"experiment_id: {experiment_id}",
        f"seed: {seed}",
        f"total_timesteps: {total_timesteps}",
    ]
    if paradigm is None or paradigm.field_enabled("intervals_count"):
        lines.append("intervals_count: 4")
    if paradigm is None or paradigm.field_enabled("eval_episodes"):
        lines.append("eval_episodes: 100")
    lines += ["", "tensorboard: true"]

    if agent:
        model = model or "dnn"
        lines += ["", "methods:", f"  {agent}_{model}:", f"    agent: {agent}", f"    model: {model}"]
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
    intervals_count: int | None = None
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
        ("intervals_count", "intervals_count"),
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
        return " ".join(self.command(**kwargs))


def unique_method_name(existing: Any, agent: str, model: str) -> str:
    """A method name free in `existing`, following the <agent>_<model> convention.

    The configs name methods after the agent they run, sometimes with a suffix
    (ppo_dnn, ppo_blendrl, iql_blendrl_arch1), so a collision gets a numeric
    suffix rather than a new scheme.
    """
    base = f"{agent}_{model}" if model else str(agent)
    taken = set(existing or ())
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def add_method(data: dict[str, Any], agent: str, model: str = "dnn") -> str:
    """Add a method to an experiment's `methods` block. Returns its name."""
    methods = data.setdefault("methods", {})
    name = unique_method_name(methods, agent, model)
    methods[name] = {"agent": agent, "model": model}
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
