"""Experiment export service for Theta-IDE.

Packages experiment configurations, base templates, environment hooks/valuations,
domain rules, custom model architectures, and optional checkpoint weights into a
self-contained, installable Theta Hub component package (.zip).
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from . import PROJECT_ROOT

log = logging.getLogger(__name__)


def _component_names(value: Any) -> Set[str]:
    """Names from a recipe's scalar, name/params mapping, or list selection."""
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        name = value.get("name") or value.get("class") or value.get("type")
        return {str(name)} if name else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_component_names(item) for item in value)) if value else set()
    return set()


def _plot_names(value: Any) -> Set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, dict):
        return {
            str(name)
            for name, options in value.items()
            if isinstance(options, dict) and name not in ("style", "figsize")
        }
    return set()


@dataclass
class ExportOptions:
    """Configuration options for exporting an experiment or experiment group plugin."""

    include_checkpoints: bool = False
    include_environment: bool = True
    include_models: bool = True
    include_rules: bool = True
    include_evaluators: bool = True
    include_plotters: bool = True
    include_data_modules: bool = True
    include_all_group_experiments: bool = True
    output_path: Path | None = None
    author: str = "Theta Researcher"
    version: str = "1.0.0"
    description: str | None = None


@dataclass
class ExportResult:
    """Result of an experiment export operation."""

    success: bool
    archive_path: Path | None = None
    manifest: Dict[str, Any] | None = None
    files_included: List[str] = field(default_factory=list)
    error: str | None = None


class ExperimentExporter:
    """Extracts, traces dependencies, and packages experiments into installable Hub components."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else PROJECT_ROOT

    def export(
        self,
        experiment_id_or_path: str | Path,
        options: ExportOptions | None = None,
    ) -> ExportResult:
        """Export an experiment and its full dependency closure to a .zip archive.

        Args:
            experiment_id_or_path: e.g. "group/experiment", "group/experiment.yaml",
                or absolute Path to the YAML recipe.
            options: Export configuration options.

        Returns:
            ExportResult containing archive path and manifest if successful.
        """
        opts = options or ExportOptions()

        try:
            group, exp_name, exp_yaml_path = self._resolve_experiment_path(experiment_id_or_path)
            if not exp_yaml_path.exists():
                return ExportResult(
                    success=False,
                    error=f"Experiment config file not found: {exp_yaml_path}",
                )

            # 1. Parse experiment YAML and trace templates
            try:
                raw_cfg = yaml.safe_load(exp_yaml_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                return ExportResult(
                    success=False,
                    error=f"Failed to parse experiment YAML '{exp_yaml_path}': {e}",
                )

            # Dictionary mapping workspace-relative posix path -> source absolute Path
            files_to_pack: Dict[str, Path] = {}

            # Add primary experiment config
            exp_rel = exp_yaml_path.relative_to(self.workspace_root).as_posix()
            files_to_pack[exp_rel] = exp_yaml_path

            # 2. Trace base configs and method templates in the experiment group
            self._trace_group_templates(group, raw_cfg, files_to_pack)

            # 3. Environment configs, wrappers, and valuation rules
            env_name = self._resolve_env_name(raw_cfg, group)
            if opts.include_environment and env_name:
                self._trace_environment_assets(env_name, files_to_pack)

            # 4. Domain rules (in/rules/<env>/)
            if opts.include_rules and env_name:
                self._trace_rules(env_name, files_to_pack)

            # 5. Model architectures and agent configs
            agents_found: Set[str] = set()
            models_found: Set[str] = set()
            self._trace_methods_and_models(
                raw_cfg,
                files_to_pack,
                agents_found,
                models_found,
                include_models=opts.include_models,
            )

            # 6. Evaluators
            if opts.include_evaluators:
                self._trace_evaluators(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 7. Plotters
            if opts.include_plotters:
                self._trace_plotters(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 8. Data modules
            if opts.include_data_modules:
                self._trace_data_modules(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 9. Workflows
            self._trace_workflows(group, files_to_pack, raw_cfg=raw_cfg)

            # 10. Checkpoints (optional)
            if opts.include_checkpoints:
                self._trace_checkpoints(group, exp_name, files_to_pack)

            # Determine output archive path
            output_path = opts.output_path
            if output_path is None:
                exports_dir = self.workspace_root / "results" / "exports"
                exports_dir.mkdir(parents=True, exist_ok=True)
                clean_exp = exp_name.replace("/", "_")
                output_path = exports_dir / f"{group}_{clean_exp}_v{opts.version}.zip"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # 7. Generate metadata manifest (.theta_component.json)
            manifest = self._generate_manifest(
                group=group,
                exp_name=exp_name,
                env_name=env_name,
                agents=sorted(agents_found),
                models=sorted(models_found),
                files=sorted(files_to_pack.keys()),
                options=opts,
            )

            # Validate manifest against formal JSON schema
            from src.app.core.manifest_schema import validate_manifest_data

            is_valid, schema_err = validate_manifest_data(manifest)
            if not is_valid:
                return ExportResult(
                    success=False,
                    error=f"Export manifest failed schema validation: {schema_err}",
                )

            # 8. Generate README.md
            readme_content = self._generate_readme(group, exp_name, env_name, opts)

            # 9. Write archive
            with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # Store manifest at root of zip
                zf.writestr(".theta_component.json", json.dumps(manifest, indent=2) + "\n")
                zf.writestr("README.md", readme_content + "\n")

                # Store all packaged assets with their relative workspace path
                for rel_path, abs_path in sorted(files_to_pack.items()):
                    if abs_path.is_file():
                        zf.write(abs_path, arcname=rel_path)

            file_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
            log.info("Exported experiment '%s/%s' (%d files) to %s", group, exp_name, len(file_list), output_path)

            return ExportResult(
                success=True,
                archive_path=output_path,
                manifest=manifest,
                files_included=file_list,
            )

        except Exception as exc:
            log.exception("Error exporting experiment %s", experiment_id_or_path)
            return ExportResult(
                success=False,
                error=str(exc),
            )

    def _resolve_experiment_path(self, target: str | Path) -> tuple[str, str, Path]:
        """Resolve group, experiment name, and file path from target."""
        exp_dir = self.workspace_root / "in" / "config" / "experiment"

        if isinstance(target, Path):
            if target.is_absolute():
                rel = target.relative_to(exp_dir)
            else:
                rel = target
        else:
            s = str(target).strip()
            if s.startswith("in/config/experiment/"):
                s = s[len("in/config/experiment/") :]
            rel = Path(s)

        parts = rel.parts
        if len(parts) >= 2:
            group = parts[0]
            exp_stem = Path("/".join(parts[1:])).stem
            exp_yaml = exp_dir / group / f"{exp_stem}.yaml"
        else:
            group = "general"
            exp_stem = rel.stem
            exp_yaml = exp_dir / f"{exp_stem}.yaml"

        return group, exp_stem, exp_yaml

    def _trace_group_templates(
        self,
        group: str,
        cfg: Dict[str, Any],
        files_to_pack: Dict[str, Path],
    ) -> None:
        """Trace _base.yaml and any referenced method templates in the group."""
        group_dir = self.workspace_root / "in" / "config" / "experiment" / group
        if not group_dir.is_dir():
            return

        base_yaml = group_dir / "_base.yaml"
        if base_yaml.is_file():
            rel = base_yaml.relative_to(self.workspace_root).as_posix()
            files_to_pack[rel] = base_yaml

        # Check defaults list for specific includes
        defaults = cfg.get("defaults", [])
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, str) and not item.startswith("override"):
                    cand = group_dir / f"{item}.yaml"
                    if cand.is_file():
                        rel = cand.relative_to(self.workspace_root).as_posix()
                        files_to_pack[rel] = cand

        # Methods directory templates
        methods_dir = group_dir / "methods"
        if methods_dir.is_dir():
            for mfile in sorted(methods_dir.glob("*.yaml")):
                rel = mfile.relative_to(self.workspace_root).as_posix()
                files_to_pack[rel] = mfile

    def _resolve_env_name(self, cfg: Dict[str, Any], group: str) -> str | None:
        """Extract environment name from configuration or defaults."""
        if "env" in cfg and isinstance(cfg["env"], str):
            return cfg["env"]

        # Check defaults list for an environment selection.
        defaults = cfg.get("defaults", [])
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, dict):
                    if "override /env" in item:
                        return str(item["override /env"])
                    if "/env" in item:
                        return str(item["/env"])
                    if "env" in item:
                        return str(item["env"])

        # Check group _base.yaml if present
        base_yaml = self.workspace_root / "in" / "config" / "experiment" / group / "_base.yaml"
        if base_yaml.is_file():
            try:
                base_cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
                if "env" in base_cfg and isinstance(base_cfg["env"], str):
                    return base_cfg["env"]
                b_defaults = base_cfg.get("defaults", [])
                if isinstance(b_defaults, list):
                    for item in b_defaults:
                        if isinstance(item, dict):
                            if "override /env" in item:
                                return str(item["override /env"])
                            if "/env" in item:
                                return str(item["/env"])
            except Exception:
                pass

        return None

    def _find_config_file(self, folder: str, name: str) -> Path | None:
        """Find a configuration file in in/config/<folder>/ directly or recursively."""
        root = self.workspace_root / "in" / "config" / folder
        if not root.is_dir():
            return None
        direct = root / f"{name}.yaml"
        if direct.is_file():
            return direct
        direct_yml = root / f"{name}.yml"
        if direct_yml.is_file():
            return direct_yml
        for p in sorted(root.rglob(f"{name}.yaml")):
            if p.is_file():
                return p
        for p in sorted(root.rglob(f"{name}.yml")):
            if p.is_file():
                return p
        return None

    def _trace_environment_assets(self, env_name: str, files_to_pack: Dict[str, Path]) -> None:
        """Trace environment YAML config and custom environment code directory."""
        # 1. in/config/env/<env_name>.yaml (direct or recursive)
        env_yaml = self._find_config_file("env", env_name)
        if env_yaml and env_yaml.is_file():
            rel = env_yaml.relative_to(self.workspace_root).as_posix()
            files_to_pack[rel] = env_yaml

        # 2. in/envs/<env_name>/
        env_code_dir = self.workspace_root / "in" / "envs" / env_name
        if env_code_dir.is_dir():
            for p in sorted(env_code_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = p.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = p

    def _trace_rules(self, env_name: str, files_to_pack: Dict[str, Path]) -> None:
        """Trace domain logic rules for the environment."""
        rules_dir = self.workspace_root / "in" / "rules" / env_name
        if rules_dir.is_dir():
            for p in sorted(rules_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = p.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = p

    def _trace_evaluators(
        self,
        group: str,
        env_name: str | None,
        files_to_pack: Dict[str, Path],
        raw_cfg: Dict[str, Any] | None = None,
    ) -> None:
        """Trace evaluator implementations and YAML configurations."""
        candidates = {group}
        if env_name:
            candidates.add(env_name)
        if raw_cfg:
            for k in ("evaluator", "evaluators"):
                candidates.update(_component_names(raw_cfg.get(k)))

        for name in candidates:
            eval_dir = self.workspace_root / "src" / "usr" / "eval" / name
            if eval_dir.is_dir():
                for p in sorted(eval_dir.rglob("*")):
                    if p.is_file() and "__pycache__" not in p.parts:
                        files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p
            else:
                eval_file = self.workspace_root / "src" / "usr" / "eval" / f"{name}.py"
                if eval_file.is_file():
                    files_to_pack[eval_file.relative_to(self.workspace_root).as_posix()] = eval_file

            cfg1 = self.workspace_root / "in" / "config" / "eval" / f"{name}.yaml"
            if cfg1.is_file():
                files_to_pack[cfg1.relative_to(self.workspace_root).as_posix()] = cfg1
            cfg_dir = self.workspace_root / "in" / "config" / "eval" / name
            if cfg_dir.is_dir():
                for p in sorted(cfg_dir.rglob("*.yaml")):
                    files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p

    def _trace_plotters(
        self, group: str, env_name: str | None, files_to_pack: Dict[str, Path], raw_cfg: Dict[str, Any] | None = None
    ) -> None:
        """Trace plotters and visualization configs."""
        candidates = {group}
        if env_name:
            candidates.add(env_name)
        if raw_cfg:
            candidates.update(_plot_names(raw_cfg.get("plots")))

        for name in candidates:
            plot_dir = self.workspace_root / "plot" / name
            if plot_dir.is_dir():
                for p in sorted(plot_dir.rglob("*")):
                    if p.is_file() and "__pycache__" not in p.parts:
                        files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p
            else:
                plot_file = self.workspace_root / "plot" / f"{name}.py"
                if plot_file.is_file():
                    files_to_pack[plot_file.relative_to(self.workspace_root).as_posix()] = plot_file
                else:
                    for p in sorted((self.workspace_root / "plot").rglob(f"{name}.py")):
                        files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p

            pcfg_dir = self.workspace_root / "in" / "config" / "plot" / name
            if pcfg_dir.is_dir():
                for p in sorted(pcfg_dir.rglob("*.yaml")):
                    files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p
            else:
                for p in sorted((self.workspace_root / "in" / "config" / "plot").rglob(f"{name}.yaml")):
                    files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p

    def _trace_data_modules(
        self,
        group: str,
        env_name: str | None,
        files_to_pack: Dict[str, Path],
        raw_cfg: Dict[str, Any] | None = None,
    ) -> None:
        """Trace data modules and configurations."""
        candidates = {group}
        if env_name:
            candidates.add(env_name)
        if raw_cfg:
            for k in ("data_module", "datamodule", "data"):
                for name in _component_names(raw_cfg.get(k)):
                    candidates.add(name)
                    candidates.add(re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())

        for name in candidates:
            data_dir = self.workspace_root / "src" / "usr" / "data" / name
            if data_dir.is_dir():
                for p in sorted(data_dir.rglob("*")):
                    if p.is_file() and "__pycache__" not in p.parts:
                        files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p
            else:
                data_file = self.workspace_root / "src" / "usr" / "data" / f"{name}.py"
                if data_file.is_file():
                    files_to_pack[data_file.relative_to(self.workspace_root).as_posix()] = data_file

            dcfg_dir = self.workspace_root / "in" / "config" / "data" / name
            if dcfg_dir.is_dir():
                for p in sorted(dcfg_dir.rglob("*.yaml")):
                    files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p
            dcfg_file = self.workspace_root / "in" / "config" / "data" / f"{name}.yaml"
            if dcfg_file.is_file():
                files_to_pack[dcfg_file.relative_to(self.workspace_root).as_posix()] = dcfg_file

    def _trace_workflows(
        self,
        group: str,
        files_to_pack: Dict[str, Path],
        raw_cfg: Dict[str, Any] | None = None,
    ) -> None:
        """Trace any associated DAG workflows in in/config/workflow/."""
        wf_dir = self.workspace_root / "in" / "config" / "workflow"
        if not wf_dir.is_dir():
            return
        workflow_names = set()
        if raw_cfg and raw_cfg.get("workflow"):
            workflow_names.add(str(raw_cfg["workflow"]))

        for p in sorted(wf_dir.glob("*.yaml")):
            if group in p.stem or p.stem in workflow_names:
                files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p

    def _trace_all_group_experiments(self, group: str, files_to_pack: Dict[str, Path]) -> None:
        """Trace all experiment recipe YAML files in the group directory."""
        g_dir = self.workspace_root / "in" / "config" / "experiment" / group
        if g_dir.is_dir():
            for p in sorted(g_dir.rglob("*.yaml")):
                files_to_pack[p.relative_to(self.workspace_root).as_posix()] = p

    def export_group(
        self,
        group: str,
        options: ExportOptions | None = None,
    ) -> ExportResult:
        """Export an entire experiment group and its full component ecosystem as an installable Experiment Plugin."""
        opts = options or ExportOptions()
        group_dir = self.workspace_root / "in" / "config" / "experiment" / group
        if not group_dir.is_dir():
            return ExportResult(
                success=False,
                error=f"Experiment group directory not found: {group_dir}",
            )

        files_to_pack: Dict[str, Path] = {}
        # 1. Pack all experiment configs in the group
        for p in sorted(group_dir.rglob("*.yaml")):
            rel = p.relative_to(self.workspace_root).as_posix()
            files_to_pack[rel] = p

        # 2. Extract configs to resolve env, agents, models, paradigms
        agents_found: Set[str] = set()
        models_found: Set[str] = set()
        paradigms_found: Set[str] = set()
        workflows_found: Set[str] = set()
        data_modules_found: Set[str] = set()
        evaluators_found: Set[str] = set()
        plotters_found: Set[str] = set()
        env_name: str | None = None

        for exp_yaml in sorted(group_dir.rglob("*.yaml")):
            try:
                cfg = yaml.safe_load(exp_yaml.read_text(encoding="utf-8")) or {}
                if not env_name:
                    env_name = self._resolve_env_name(cfg, group)
                if "paradigm" in cfg and cfg["paradigm"]:
                    paradigms_found.add(str(cfg["paradigm"]))
                if "workflow" in cfg and cfg["workflow"]:
                    workflows_found.add(str(cfg["workflow"]))
                if "data_module" in cfg and cfg["data_module"]:
                    data_modules_found.update(_component_names(cfg["data_module"]))
                if "evaluator" in cfg and cfg["evaluator"]:
                    evaluators_found.update(_component_names(cfg["evaluator"]))
                plotters_found.update(_plot_names(cfg.get("plots")))
                self._trace_methods_and_models(
                    cfg, files_to_pack, agents_found, models_found, include_models=opts.include_models
                )
            except Exception:
                pass

        if not env_name:
            env_name = group

        # 3. Environment, rules, evaluators, plotters, data loaders, workflows
        if opts.include_environment and env_name:
            self._trace_environment_assets(env_name, files_to_pack)

        if opts.include_rules and env_name:
            self._trace_rules(env_name, files_to_pack)

        if opts.include_evaluators:
            self._trace_evaluators(group, env_name, files_to_pack, raw_cfg={"evaluators": list(evaluators_found)})

        if opts.include_plotters:
            self._trace_plotters(
                group, env_name, files_to_pack, raw_cfg={"plots": {name: {} for name in plotters_found}}
            )

        if opts.include_data_modules:
            self._trace_data_modules(group, env_name, files_to_pack, raw_cfg={"data": list(data_modules_found)})

        self._trace_workflows(
            group,
            files_to_pack,
            raw_cfg={"workflow": list(workflows_found)[0] if workflows_found else None},
        )

        # 4. Output archive path
        output_path = opts.output_path
        if output_path is None:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            output_path = exports_dir / f"{group}_experiment_v{opts.version}.zip"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. Manifest
        manifest = self._generate_manifest(
            group=group,
            exp_name="all",
            env_name=env_name,
            agents=sorted(agents_found),
            models=sorted(models_found),
            files=sorted(files_to_pack.keys()),
            options=opts,
            paradigms=sorted(paradigms_found) if paradigms_found else ["offline_rl"],
        )

        from src.app.core.manifest_schema import validate_manifest_data

        is_valid, schema_err = validate_manifest_data(manifest)
        if not is_valid:
            return ExportResult(
                success=False,
                error=f"Export manifest failed schema validation: {schema_err}",
            )

        readme_content = self._generate_readme(group, "all", env_name, opts)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(".theta_component.json", json.dumps(manifest, indent=2) + "\n")
            zf.writestr("README.md", readme_content + "\n")
            for rel_path, abs_path in sorted(files_to_pack.items()):
                if abs_path.is_file():
                    zf.write(abs_path, arcname=rel_path)

        file_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
        log.info("Exported experiment group '%s' (%d files) to %s", group, len(file_list), output_path)

        return ExportResult(
            success=True,
            archive_path=output_path,
            manifest=manifest,
            files_included=file_list,
        )

    def _trace_methods_and_models(
        self,
        cfg: Dict[str, Any],
        files_to_pack: Dict[str, Path],
        agents_found: Set[str],
        models_found: Set[str],
        include_models: bool = True,
    ) -> None:
        """Extract declared agents and models, tracing their configs and implementation code."""
        methods = cfg.get("methods", {})
        if not isinstance(methods, dict):
            return

        for m_name, m_cfg in methods.items():
            if not isinstance(m_cfg, dict):
                continue

            agent_name = m_cfg.get("agent")
            if agent_name and isinstance(agent_name, str):
                agents_found.add(agent_name)
                # in/config/agent/<agent>.yaml (direct or recursive)
                agent_yaml = self._find_config_file("agent", agent_name)
                if agent_yaml and agent_yaml.is_file():
                    rel = agent_yaml.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = agent_yaml

            # Extract model names (handle flat or nested composite specs)
            extracted_models: List[str] = []

            def _collect_models(val: Any):
                if isinstance(val, str):
                    clean = val.strip()
                    if clean and clean.lower() not in (
                        "default",
                        "none",
                        "true",
                        "false",
                        "identity",
                        "neural",
                        "learned",
                        "heuristic",
                        "linear",
                        "softmax",
                    ):
                        extracted_models.append(clean)
                elif isinstance(val, dict):
                    arch = val.get("architecture") or val.get("name") or val.get("type") or val.get("base")
                    if arch and isinstance(arch, str):
                        extracted_models.append(arch)
                    for k, sub in val.items():
                        if k in ("architecture", "name", "type", "base", "slots", "rules", "mode", "lr"):
                            continue
                        if isinstance(sub, dict):
                            if len(sub) == 1 and not any(
                                ik in sub for ik in ("lr", "hidden_sizes", "activation", "dim")
                            ):
                                first_k = next(iter(sub))
                                if isinstance(sub[first_k], dict):
                                    extracted_models.append(first_k)
                                    _collect_models(sub[first_k])
                                    continue
                            _collect_models(sub)
                        elif isinstance(sub, str):
                            _collect_models(sub)

            _collect_models(m_cfg.get("model"))
            for m in sorted(set(extracted_models)):
                models_found.add(m)
                # in/config/model/<m>.yaml (direct or recursive)
                model_yaml = self._find_config_file("model", m)
                if model_yaml and model_yaml.is_file():
                    rel = model_yaml.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = model_yaml

                # If include_models is True and model lives in src/usr/models/<m>
                if include_models:
                    model_dir = self.workspace_root / "src" / "usr" / "models" / m
                    if model_dir.is_dir():
                        for p in sorted(model_dir.rglob("*")):
                            if p.is_file() and "__pycache__" not in p.parts:
                                rel = p.relative_to(self.workspace_root).as_posix()
                                files_to_pack[rel] = p
                    else:
                        model_file = self.workspace_root / "src" / "usr" / "models" / f"{m}.py"
                        if model_file.is_file():
                            rel = model_file.relative_to(self.workspace_root).as_posix()
                            files_to_pack[rel] = model_file

    def _trace_checkpoints(
        self,
        group: str,
        exp_name: str,
        files_to_pack: Dict[str, Path],
    ) -> None:
        """Find and package saved checkpoints if available."""
        ckpt_dir = self.workspace_root / "results" / "checkpoints" / group / exp_name
        if ckpt_dir.is_dir():
            for p in sorted(ckpt_dir.rglob("*")):
                if p.is_file() and p.suffix in (".ckpt", ".pt", ".pth", ".json"):
                    rel = p.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = p

    def _generate_manifest(
        self,
        group: str,
        exp_name: str,
        env_name: str | None,
        agents: List[str],
        models: List[str],
        files: List[str],
        options: ExportOptions,
        paradigms: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Generate component manifest matching Theta Hub specifications."""
        is_group_bundle = exp_name in ("all", group)
        component_id = group if is_group_bundle else f"{group}_{exp_name}".replace("/", "_")
        config_paths = sorted([k for k in files if k.startswith("in/config/")])
        manifest = {
            "id": component_id,
            "name": f"{group.replace('_', ' ').title()} Experiment Suite" if is_group_bundle else f"{group}/{exp_name}",
            "kind": "experiment",
            "version": options.version,
            "description": options.description
            or f"Complete {group} experiment suite: recipes, group defaults, environment hooks, evaluators, and plotters.",
            "author": {"name": options.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"in/config/experiment/{group}",
            "config_paths": config_paths,
            "compatibility": {"theta": ">=2.0.0"},
            "environment": env_name,
            "agents": agents,
            "models": models,
            "package_contents": files,
        }
        if not is_group_bundle:
            manifest["entry_point"] = f"in/config/experiment/{group}/{exp_name}.yaml"
        if paradigms:
            manifest["paradigms"] = paradigms
        return manifest

    def _generate_readme(
        self,
        group: str,
        exp_name: str,
        env_name: str | None,
        options: ExportOptions,
    ) -> str:
        """Generate descriptive reproduction README."""
        lines = [
            f"# Experiment Package: {group}/{exp_name}",
            "",
            f"**Version**: {options.version}  ",
            f"**Author**: {options.author}  ",
            f"**Environment**: {env_name or 'unspecified'}  ",
            "",
            "## Overview",
            options.description or f"This is a self-contained Theta-IDE experiment package for `{group}/{exp_name}`.",
            "",
            "## Installation",
            "You can install this component directly into your Theta-IDE workspace:",
            "1. **Via Theta Hub GUI**: Open Theta-IDE -> Components -> Hub, and import this package archive.",
            "2. **Manual Extraction**: Extract the archive directly into your Theta-IDE workspace root.",
            "",
            "## Running the Experiment",
            "Execute using the unified Theta pipeline:",
            "```bash",
            f"python run_pipeline.py {group}/{exp_name}",
            "```",
            "",
            "To view the experiment parameters in the Theta-IDE GUI, launch:",
            "```bash",
            "python -m frontend",
            "```",
        ]
        return "\n".join(lines)
