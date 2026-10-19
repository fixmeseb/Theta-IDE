"""Component export service for Theta-IDE.

Packages models, algorithms/agents, environments, datamodules, and evaluators into
self-contained, installable Theta Hub component packages (.zip) or standalone
directories ready for GitHub distribution.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from . import PROJECT_ROOT

log = logging.getLogger(__name__)


@dataclass
class ComponentExportOptions:
    """Configuration options for exporting a Theta component plugin."""
    export_format: str = "zip"  # "zip" or "folder"
    output_path: Path | None = None
    version: str = "1.0.0"
    author: str = "Theta Researcher"
    description: str | None = None
    include_configs: bool = True
    include_rules: bool = True
    generate_git_scaffold: bool = True


@dataclass
class ComponentExportResult:
    """Result of a component export operation."""
    success: bool
    export_path: Path | None = None
    manifest: Dict[str, Any] | None = None
    files_included: List[str] = field(default_factory=list)
    error: str | None = None


class ComponentExporter:
    """Extracts, traces dependencies, and packages components into installable Hub packages or Git repos."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else PROJECT_ROOT

    def export(
        self,
        config_path_or_id: str | Path,
        options: ComponentExportOptions | None = None,
    ) -> ComponentExportResult:
        """Export a component plugin to a .zip archive or standalone directory.

        Args:
            config_path_or_id: Relative/absolute path to component YAML or component ID.
            options: Export configuration options.

        Returns:
            ComponentExportResult with output path, manifest, and packaged file list.
        """
        opts = options or ComponentExportOptions()

        try:
            from frontend.component_actions import find_component_assets
        except Exception:
            # Fallback inline resolution if frontend cannot be imported
            find_component_assets = None

        try:
            # 1. Resolve component config path
            cfg_path, rel_path = self._resolve_config_path(config_path_or_id)
            if not cfg_path.exists():
                return ComponentExportResult(
                    success=False,
                    error=f"Component configuration file not found: {cfg_path}",
                )

            # 2. Discover component assets
            if find_component_assets is not None:
                assets = find_component_assets(cfg_path, rel_path, workspace_dir=self.workspace_root)
            else:
                assets = self._fallback_find_assets(cfg_path, rel_path)

            comp_id = assets["comp_id"]
            comp_name = assets.get("comp_name") or comp_id
            kind = assets.get("kind") or "model"
            code_dir = assets.get("code_dir")
            primary_code_file = assets.get("primary_code_file")
            meta_existing = assets.get("meta") or {}

            # Map workspace-relative posix path -> source absolute Path
            files_to_pack: Dict[str, Path] = {}

            # 3. Pack configuration files
            config_files: List[Path] = []
            if opts.include_configs:
                for f in assets.get("files_to_delete", []):
                    if f.is_file() and f.suffix in (".yaml", ".yml"):
                        try:
                            rel = f.relative_to(self.workspace_root).as_posix()
                        except ValueError:
                            rel = f"in/config/{kind}/{f.name}"
                        files_to_pack[rel] = f
                        config_files.append(f)

                # Ensure primary config is definitely packed
                try:
                    p_rel = cfg_path.relative_to(self.workspace_root).as_posix()
                except ValueError:
                    p_rel = f"in/config/{kind}/{cfg_path.name}"
                files_to_pack[p_rel] = cfg_path
                if cfg_path not in config_files:
                    config_files.append(cfg_path)

            # 4. Pack code implementation files
            if code_dir and code_dir.is_dir():
                for item in sorted(code_dir.rglob("*")):
                    if not item.is_file():
                        continue
                    if "__pycache__" in item.parts or ".git" in item.parts or item.name == ".DS_Store":
                        continue
                    try:
                        rel = item.relative_to(self.workspace_root).as_posix()
                    except ValueError:
                        rel = f"src/usr/{kind}s/{comp_id}/{item.relative_to(code_dir).as_posix()}"
                    files_to_pack[rel] = item
            elif primary_code_file and primary_code_file.is_file():
                try:
                    rel = primary_code_file.relative_to(self.workspace_root).as_posix()
                except ValueError:
                    rel = f"src/usr/{kind}s/{primary_code_file.name}"
                files_to_pack[rel] = primary_code_file

            # 5. Pack environment rules if applicable
            if opts.include_rules and kind == "env":
                rules_dir = self.workspace_root / "in" / "rules" / comp_id
                if rules_dir.is_dir():
                    for rf in sorted(rules_dir.rglob("*")):
                        if rf.is_file() and "__pycache__" not in rf.parts and rf.name != ".DS_Store":
                            rel = rf.relative_to(self.workspace_root).as_posix()
                            files_to_pack[rel] = rf

            # 6. Generate schema-compliant manifest (.theta_component.json)
            manifest = self._generate_manifest(
                comp_id=comp_id,
                comp_name=comp_name,
                kind=kind,
                code_dir=code_dir,
                config_files=config_files,
                existing_meta=meta_existing,
                options=opts,
            )

            # 7. Generate README.md
            readme_text = self._generate_readme(
                comp_id=comp_id,
                comp_name=comp_name,
                kind=kind,
                manifest=manifest,
                options=opts,
            )

            # 8. Output Destination
            if opts.export_format == "folder":
                return self._export_to_folder(
                    files_to_pack=files_to_pack,
                    manifest=manifest,
                    readme_text=readme_text,
                    comp_id=comp_id,
                    options=opts,
                )
            else:
                return self._export_to_zip(
                    files_to_pack=files_to_pack,
                    manifest=manifest,
                    readme_text=readme_text,
                    comp_id=comp_id,
                    options=opts,
                )

        except Exception as exc:
            log.exception("Component export failed")
            return ComponentExportResult(
                success=False,
                error=f"Export failed: {exc}",
            )

    def _resolve_config_path(self, target: str | Path) -> tuple[Path, str]:
        """Resolve arbitrary input to an absolute Path and workspace-relative string."""
        p = Path(target)
        if p.is_absolute() and p.is_file():
            try:
                rel = p.relative_to(self.workspace_root).as_posix()
            except ValueError:
                rel = p.name
            return p, rel

        # Check relative to workspace
        candidate = self.workspace_root / target
        if candidate.is_file():
            return candidate, candidate.relative_to(self.workspace_root).as_posix()

        # Check candidate locations under in/config/
        target_str = str(target).replace("\\", "/").strip()
        if target_str.startswith("in/config/"):
            target_str = target_str[len("in/config/"):]
        if not target_str.endswith((".yaml", ".yml")):
            target_str_yaml = f"{target_str}.yaml"
        else:
            target_str_yaml = target_str

        search_prefixes = ["model", "agent", "env", "data", "eval"]
        for prefix in search_prefixes:
            cfg = self.workspace_root / "in" / "config" / prefix / target_str_yaml
            if cfg.is_file():
                return cfg, cfg.relative_to(self.workspace_root).as_posix()

        # Check recursively in in/config
        target_name = Path(target_str_yaml).name
        found = list((self.workspace_root / "in" / "config").rglob(target_name))
        if found:
            return found[0], found[0].relative_to(self.workspace_root).as_posix()

        # Fallback to direct path in/config/<target_str_yaml>
        cfg_fallback = self.workspace_root / "in" / "config" / target_str_yaml
        return cfg_fallback, f"in/config/{target_str_yaml}"

    def _fallback_find_assets(self, cfg_path: Path, rel_path: str) -> Dict[str, Any]:
        stem = cfg_path.stem
        kind = "model"
        if "agent" in rel_path or "method" in rel_path or "algorithm" in rel_path:
            kind = "algorithm"
        elif "env" in rel_path:
            kind = "env"
        elif "data" in rel_path or "datamodule" in rel_path:
            kind = "datamodule"
        elif "eval" in rel_path or "evaluator" in rel_path:
            kind = "evaluator"
        elif "plot" in rel_path or "plotter" in rel_path:
            kind = "plotter"

        code_dir = None
        if kind == "model":
            cd = self.workspace_root / "src" / "usr" / "models" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind in ("agent", "algorithm"):
            cd = self.workspace_root / "src" / "usr" / "algorithms" / stem
            if not cd.is_dir():
                cd = self.workspace_root / "src" / "usr" / "methods" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "env":
            cd = self.workspace_root / "in" / "envs" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "datamodule":
            cd = self.workspace_root / "src" / "usr" / "data" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "evaluator":
            cd = self.workspace_root / "src" / "usr" / "eval" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "plotter":
            cd = self.workspace_root / "plot" / stem
            if cd.is_dir():
                code_dir = cd

        return {
            "comp_id": stem,
            "comp_name": stem.capitalize(),
            "kind": kind,
            "code_dir": code_dir,
            "primary_code_file": None,
            "files_to_delete": [cfg_path],
            "meta": None,
        }

    def _generate_manifest(
        self,
        comp_id: str,
        comp_name: str,
        kind: str,
        code_dir: Path | None,
        config_files: List[Path],
        existing_meta: Dict[str, Any],
        options: ComponentExportOptions,
    ) -> Dict[str, Any]:
        """Generate a valid schema-compliant .theta_component.json manifest."""
        manifest: Dict[str, Any] = dict(existing_meta)
        manifest.pop("target_dir", None)
        manifest["id"] = comp_id
        manifest["name"] = existing_meta.get("name") or comp_name
        manifest["kind"] = existing_meta.get("kind") or kind
        manifest["version"] = options.version
        manifest["description"] = options.description or existing_meta.get("description") or f"{comp_name} component for Theta-IDE"
        author_val = options.author or existing_meta.get("author") or "Theta Researcher"
        manifest["author"] = {"name": author_val} if isinstance(author_val, str) else author_val
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()

        if code_dir:
            try:
                manifest["target_path"] = code_dir.relative_to(self.workspace_root).as_posix()
            except ValueError:
                manifest["target_path"] = f"src/usr/{kind}s/{comp_id}"

        cfg_rels = []
        for cf in config_files:
            try:
                cfg_rels.append(cf.relative_to(self.workspace_root).as_posix())
            except ValueError:
                cfg_rels.append(f"in/config/{kind}/{cf.name}")

        if cfg_rels:
            manifest["config_path"] = cfg_rels[0]
        manifest["compatibility"] = existing_meta.get("compatibility") or {"theta": ">=2.0.0"}

        # Extract and preserve supported paradigms
        paradigms = existing_meta.get("paradigms") or existing_meta.get("compatible_paradigms")
        if not paradigms and config_files:
            try:
                primary_cfg_content = yaml.safe_load(config_files[0].read_text(encoding="utf-8")) or {}
                if isinstance(primary_cfg_content, dict):
                    # Check top-level or env / agent sub-dict
                    paradigms = primary_cfg_content.get("paradigms") or primary_cfg_content.get("compatible_paradigms")
                    if not paradigms and isinstance(primary_cfg_content.get("env"), dict):
                        paradigms = primary_cfg_content["env"].get("paradigms")
            except Exception:
                pass
        if paradigms:
            manifest["paradigms"] = paradigms

        return manifest

    def _generate_readme(
        self,
        comp_id: str,
        comp_name: str,
        kind: str,
        manifest: Dict[str, Any],
        options: ComponentExportOptions,
    ) -> str:
        """Generate reproduction and integration guide for the component."""
        cfg_path = manifest.get("config_path") or f"in/config/{kind}/{comp_id}.yaml"
        target_path = manifest.get("target_path") or f"src/usr/{kind}s/{comp_id}"

        lines = [
            f"# {comp_name} ({kind.capitalize()} Component)",
            "",
            f"**Version**: {options.version}  ",
            f"**Author**: {options.author}  ",
            f"**Component ID**: `{comp_id}`  ",
            f"**Kind**: `{kind}`  ",
            "",
            "## Overview",
            manifest.get("description") or f"Standalone Theta-IDE component package for `{comp_name}`.",
            "",
            "## Package Contents",
            f"- **Source Directory**: `{target_path}`",
            f"- **Base Configuration**: `{cfg_path}`",
            "- **Component Manifest**: `.theta_component.json`",
            "",
            "## Installation",
            "### 1. Via Theta Hub GUI (Recommended)",
            "1. Open Theta-IDE and switch to the **Components** panel.",
            "2. Click **Component Hub** -> **Import / Install Component**.",
            "3. Select this archive or directory to install with 1 click.",
            "",
            "### 2. Manual Extraction",
            "Extract this package directly into your Theta-IDE workspace root directory.",
            "",
            "## Usage in Experiments",
            "Reference this component directly in any experiment YAML configuration (`in/config/experiment/<group>/<exp>.yaml`):",
            "```yaml",
        ]

        if kind == "model":
            lines.extend([
                "methods:",
                "  my_run:",
                "    agent: ppo",
                f"    model: {comp_id}",
            ])
        elif kind == "agent":
            lines.extend([
                "methods:",
                "  my_run:",
                f"    agent: {comp_id}",
                "    model: dnn",
            ])
        elif kind == "env":
            lines.extend([
                f"env: {comp_id}",
                "methods:",
                "  my_run:",
                "    agent: ppo",
            ])
        else:
            lines.extend([
                f"{kind}: {comp_id}",
            ])

        lines.extend([
            "```",
            "",
            "Run using the Theta execution engine:",
            "```bash",
            "python run_pipeline.py <group>/<experiment_name>",
            "```",
        ])
        return "\n".join(lines) + "\n"

    def _export_to_zip(
        self,
        files_to_pack: Dict[str, Path],
        manifest: Dict[str, Any],
        readme_text: str,
        comp_id: str,
        options: ComponentExportOptions,
    ) -> ComponentExportResult:
        """Write all component assets and manifests to a .zip archive."""
        if options.output_path:
            out_zip = Path(options.output_path)
        else:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            out_zip = exports_dir / f"{comp_id}_v{options.version}.zip"

        out_zip.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write all tracked files
            for rel_posix, abs_path in sorted(files_to_pack.items()):
                if abs_path.is_file():
                    zf.write(abs_path, arcname=rel_posix)

            # Write manifest at root and inside target_path
            manifest_json = json.dumps(manifest, indent=2) + "\n"
            zf.writestr(".theta_component.json", manifest_json)
            if manifest.get("target_path"):
                target_manifest = f"{manifest['target_path']}/.theta_component.json"
                if target_manifest not in files_to_pack:
                    zf.writestr(target_manifest, manifest_json)

            # Write README.md at root
            zf.writestr("README.md", readme_text)

        files_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
        return ComponentExportResult(
            success=True,
            export_path=out_zip,
            manifest=manifest,
            files_included=files_list,
        )

    def _export_to_folder(
        self,
        files_to_pack: Dict[str, Path],
        manifest: Dict[str, Any],
        readme_text: str,
        comp_id: str,
        options: ComponentExportOptions,
    ) -> ComponentExportResult:
        """Copy all component assets and generate git repository scaffolding in a folder."""
        if options.output_path:
            out_dir = Path(options.output_path)
        else:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            out_dir = exports_dir / comp_id

        out_dir.mkdir(parents=True, exist_ok=True)

        files_copied: List[str] = []
        for rel_posix, abs_path in sorted(files_to_pack.items()):
            if abs_path.is_file():
                dest = out_dir / rel_posix
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(abs_path), str(dest))
                files_copied.append(rel_posix)

        # Write root .theta_component.json and README.md
        manifest_file = out_dir / ".theta_component.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files_copied.append(".theta_component.json")

        readme_file = out_dir / "README.md"
        readme_file.write_text(readme_text, encoding="utf-8")
        files_copied.append("README.md")

        if options.generate_git_scaffold:
            gitignore_file = out_dir / ".gitignore"
            if not gitignore_file.exists():
                gitignore_file.write_text(
                    "__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.DS_Store\nbuild/\ndist/\n*.egg-info/\n.pytest_cache/\n",
                    encoding="utf-8",
                )
                files_copied.append(".gitignore")

            pyproject_file = out_dir / "pyproject.toml"
            if not pyproject_file.exists():
                pyproject_content = f"""[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "theta-component-{comp_id}"
version = "{options.version}"
description = "{manifest.get('description', '')}"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
]

[project.entry-points."theta.components"]
{comp_id} = "{manifest.get('target_path', '').replace('/', '.')}:{manifest.get('name', comp_id)}"
"""
                pyproject_file.write_text(pyproject_content, encoding="utf-8")
                files_copied.append("pyproject.toml")

        return ComponentExportResult(
            success=True,
            export_path=out_dir,
            manifest=manifest,
            files_included=files_copied,
        )
