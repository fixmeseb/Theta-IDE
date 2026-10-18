"""Component installer, archive extractor, and integrity verification."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
from typing import Callable, Optional, Tuple
from zipfile import ZipFile

from .models import AuthorInfo, HubComponent, ReleaseInfo
from frontend.plugins.dependencies import (
    check_missing_dependencies,
    install_packages,
    load_requirements_from_file,
)


class HubInstaller:
    """Manages the download, verification, extraction, and removal of Hub components."""

    def __init__(
        self,
        workspace_dir: Path,
        data_dir: Path,
        on_change_callback: Optional[Callable[[str, str], None]] = None,
        settings_manager: Optional[Any] = None,
        lookup_component_fn: Optional[Callable[[str], Optional[HubComponent]]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.on_change_callback = on_change_callback  # (component_id, action: "install" | "uninstall")
        self.settings_manager = settings_manager
        self.lookup_component_fn = lookup_component_fn

    def resolve_target_dir(self, component: HubComponent) -> Path:
        """Determine the filesystem destination for a component based on its kind."""
        if component.target_path:
            # If target_path is relative, determine root
            if component.kind == "plugin":
                return self.data_dir / component.target_path
            return self.workspace_dir / component.target_path

        # Standard conventions
        if component.kind == "plugin":
            return self.data_dir / "plugins" / component.id
        elif component.kind in ("method", "agent", "algorithm"):
            return self.workspace_dir / "src" / "usr" / "algorithms" / component.id
        elif component.kind in ("datamodule", "data"):
            return self.workspace_dir / "src" / "usr" / "data" / component.id
        elif component.kind in ("evaluator", "eval"):
            return self.workspace_dir / "src" / "usr" / "eval" / component.id
        elif component.kind == "model":
            return self.workspace_dir / "src" / "usr" / "models" / component.id
        elif component.kind == "env":
            return self.workspace_dir / "in" / "envs" / component.id
        elif component.kind == "experiment":
            return self.workspace_dir / "in" / "config" / "experiment" / component.id
        elif component.kind in ("suite", "package"):
            return self.workspace_dir / ".thetaide" / "suites" / component.id
        else:
            return self.workspace_dir / "components" / component.id

    def resolve_config_path(self, component: HubComponent) -> Optional[Path]:
        """Determine destination path for a component's default configuration in in/config/."""
        extra_cfg = getattr(component, "extra", {}).get("config_path") if hasattr(component, "extra") else None
        if not extra_cfg and hasattr(component, "config_path") and component.config_path:
            extra_cfg = component.config_path
        if extra_cfg:
            return self.workspace_dir / extra_cfg

        if component.kind == "model":
            if component.id in ("dnn", "nature_cnn", "dueling_resnet"):
                return self.workspace_dir / "in" / "config" / "model" / "basics" / f"{component.id}.yaml"
            return self.workspace_dir / "in" / "config" / "model" / f"{component.id}.yaml"
        elif component.kind in ("method", "agent", "algorithm"):
            return self.workspace_dir / "in" / "config" / "agent" / f"{component.id}.yaml"
        elif component.kind == "env":
            return self.workspace_dir / "in" / "config" / "env" / f"{component.id}.yaml"
        elif component.kind == "experiment":
            return self.workspace_dir / "in" / "config" / "experiment" / f"{component.id}.yaml"
        return None

    def lookup_component(self, component_id: str) -> Optional[HubComponent]:
        """Look up a HubComponent by ID from lookup_component_fn, local index cache, or candidate directories."""
        if self.lookup_component_fn is not None:
            comp = self.lookup_component_fn(component_id)
            if comp:
                return comp

        hub_candidates = [
            self.workspace_dir.resolve().parent / "theta-hub",
            Path(__file__).resolve().parents[3] / "theta-hub",
        ]
        for hub_dir in hub_candidates:
            if not hub_dir.is_dir():
                continue
            for json_path in hub_dir.glob(f"registry/*/{component_id}.json"):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    comp = HubComponent.from_dict(data)
                    for ver, rel in list(comp.releases.items()):
                        zip_stem = Path(rel.url).name
                        local_zip = hub_dir / "packages" / zip_stem
                        if local_zip.is_file():
                            rel.url = local_zip.as_uri()
                    return comp
                except Exception:
                    continue

        cache_index = self.data_dir / "cache" / "hub" / "index.json"
        if cache_index.is_file():
            try:
                data = json.loads(cache_index.read_text(encoding="utf-8"))
                for item in data.get("components", []):
                    if item.get("id") == component_id:
                        return HubComponent.from_dict(item)
            except Exception:
                pass

        return None

    def check_installed(self, component: HubComponent) -> Tuple[bool, Optional[str]]:
        """Check if a component is installed and determine its version."""
        target_dir = self.resolve_target_dir(component)
        if not target_dir.exists() or not target_dir.is_dir():
            return False, None

        # 1. Check .theta_component.json manifest
        meta_file = target_dir / ".theta_component.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        # 2. Check plugin.json manifest
        plugin_file = target_dir / "plugin.json"
        if plugin_file.exists():
            try:
                data = json.loads(plugin_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        return True, "unknown"

    def download_and_verify(
        self,
        url: str,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Download an archive and verify its cryptographic SHA-256 checksum."""
        tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="theta_pkg_", suffix=".zip")
        # Writes below reopen the file by path; an open handle would also block unlink() on Windows
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        hasher = hashlib.sha256()
        total_downloaded = 0

        try:
            local_candidates = [
                self.workspace_dir.resolve().parent / "theta-hub" / "packages" / Path(url).name,
                Path(__file__).resolve().parents[3] / "theta-hub" / "packages" / Path(url).name,
            ]
            matching_local = next((p for p in local_candidates if p.is_file()), None)

            if url.startswith("file://"):
                local_source = Path(url[7:])
                content = local_source.read_bytes()
                hasher.update(content)
                tmp_path.write_bytes(content)
                total_downloaded = len(content)
                if progress_callback:
                    progress_callback(total_downloaded, total_downloaded)
            elif matching_local is not None:
                content = matching_local.read_bytes()
                hasher.update(content)
                tmp_path.write_bytes(content)
                total_downloaded = len(content)
                if progress_callback:
                    progress_callback(total_downloaded, total_downloaded)
            else:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ThetaIDE-HubClient/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(tmp_path, "wb") as f:
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else -1

                    chunk_size = 64 * 1024
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        total_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(total_downloaded, total_size)

            computed_sha = hasher.hexdigest().lower()
            if expected_sha256:
                expected_clean = expected_sha256.strip().lower()
                if computed_sha != expected_clean:
                    raise ValueError(
                        f"Checksum verification failed!\n"
                        f"Expected SHA-256: {expected_clean}\n"
                        f"Computed SHA-256: {computed_sha}"
                    )

            return tmp_path

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def install(
        self,
        component: HubComponent,
        version: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """Download, verify, and unpack a component release into its destination directory."""
        ver = version or component.version
        if ver not in component.releases:
            raise ValueError(f"Version {ver} not found in releases for {component.id}")

        release = component.releases[ver]
        archive_path = self.download_and_verify(
            url=release.url,
            expected_sha256=release.sha256,
            progress_callback=progress_callback,
        )

        target_dir = self.resolve_target_dir(component)
        staging_dir = Path(tempfile.mkdtemp(prefix="theta_staging_"))

        try:
            with ZipFile(archive_path, "r") as zf:
                zf.extractall(staging_dir)

            # Handle case where zip contains a single enclosing root directory
            extracted_items = list(staging_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
            else:
                source_dir = staging_dir

            # Validate component manifest if packaged in archive
            pkg_meta_file = source_dir / ".theta_component.json"
            if pkg_meta_file.is_file():
                from src.app.core.manifest_schema import validate_manifest_file
                is_valid, err_msg = validate_manifest_file(pkg_meta_file)
                if not is_valid:
                    raise ValueError(f"Component archive contains invalid manifest: {err_msg}")

            # Check for packaged configuration YAML(s) and deploy to in/config/
            installed_config_paths = []
            is_root_relative = (source_dir / "in").is_dir() or component.kind == "experiment"

            if is_root_relative:
                # Deploy root-relative assets directly to workspace_dir (in/, src/, results/)
                for item in sorted(source_dir.rglob("*")):
                    if not item.is_file():
                        continue
                    rel = item.relative_to(source_dir)
                    if rel.parts and rel.parts[0] in ("in", "src", "results"):
                        dest_file = self.workspace_dir / rel
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(item), str(dest_file))
                        if len(rel.parts) >= 2 and rel.parts[0] == "in" and rel.parts[1] == "config":
                            installed_config_paths.append(str(dest_file.relative_to(self.workspace_dir)))

            pkg_config_dir = source_dir / "config"
            if pkg_config_dir.is_dir():
                target_in_config = self.workspace_dir / "in" / "config"
                for yaml_file in sorted(pkg_config_dir.rglob("*.yaml")):
                    rel = yaml_file.relative_to(pkg_config_dir)
                    dest_file = target_in_config / rel
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(yaml_file), str(dest_file))
                    installed_config_paths.append(str(dest_file.relative_to(self.workspace_dir)))

            config_dest = self.resolve_config_path(component)
            installed_config_rel = installed_config_paths[0] if installed_config_paths else None
            if not installed_config_paths and config_dest:
                config_candidates = [
                    source_dir / f"{component.id}.yaml",
                    source_dir / f"{component.id}.yml",
                    source_dir / "config.yaml",
                    source_dir / "config.yml",
                    source_dir / "default.yaml",
                    source_dir / "default.yml",
                    source_dir / "default_config.yaml",
                ]
                for yaml_file in sorted(list(source_dir.glob("*.yaml")) + list(source_dir.glob("*.yml"))):
                    if yaml_file not in config_candidates:
                        config_candidates.append(yaml_file)

                for cand in config_candidates:
                    if cand.is_file():
                        config_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(cand), str(config_dest))
                        installed_config_rel = str(config_dest.relative_to(self.workspace_dir))
                        installed_config_paths.append(installed_config_rel)
                        break

            # Write component tracking metadata
            existing_pkg_meta = {}
            if (source_dir / ".theta_component.json").is_file():
                try:
                    existing_pkg_meta = json.loads((source_dir / ".theta_component.json").read_text(encoding="utf-8"))
                except Exception:
                    pass

            meta = dict(existing_pkg_meta)
            meta.update({
                "id": component.id,
                "name": component.name,
                "kind": component.kind,
                "version": ver,
                "installed_from": release.url,
                "sha256": release.sha256,
                "config_path": installed_config_rel,
                "config_paths": installed_config_paths,
            })
            if hasattr(component, "extra") and isinstance(component.extra, dict) and "paradigms" in component.extra:
                meta["paradigms"] = component.extra["paradigms"]

            if is_root_relative:
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / ".theta_component.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                readme_src = source_dir / "README.md"
                if readme_src.is_file():
                    shutil.copy2(str(readme_src), str(target_dir / "README.md"))
            else:
                (source_dir / ".theta_component.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )

                # Atomic swap into target_dir
                if target_dir.exists():
                    backup_dir = target_dir.with_name(f"{target_dir.name}.backup")
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    target_dir.rename(backup_dir)
                    try:
                        shutil.move(str(source_dir), str(target_dir))
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    except Exception:
                        # Rollback
                        if target_dir.exists():
                            shutil.rmtree(target_dir, ignore_errors=True)
                        backup_dir.rename(target_dir)
                else:
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_dir), str(target_dir))

            # If this component is a suite declaring child components, install each
            suite_components = []
            target_meta_file = target_dir / ".theta_component.json"
            if target_meta_file.is_file():
                try:
                    pkg_data = json.loads(target_meta_file.read_text(encoding="utf-8"))
                    suite_components = pkg_data.get("dependencies", {}).get("components", [])
                except Exception:
                    pass
            if not suite_components and hasattr(component, "extra") and isinstance(component.extra, dict):
                suite_components = component.extra.get("dependencies", {}).get("components", [])

            installed_children = []
            if suite_components:
                for child_id in suite_components:
                    child_comp = self.lookup_component(child_id)
                    if child_comp:
                        self.install(child_comp)
                        installed_children.append(child_id)

            if installed_children:
                meta["installed_components"] = installed_children
                target_meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            # Inspect and install declared Python dependencies
            declared_deps = []
            if hasattr(component, "extra") and isinstance(component.extra, dict):
                declared_deps.extend(component.extra.get("dependencies", {}).get("pip", []))

            plugin_file = target_dir / "plugin.json"
            if plugin_file.exists():
                try:
                    pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                    declared_deps.extend(pdata.get("dependencies", {}).get("pip", []))
                except Exception:
                    pass

            req_file = target_dir / "requirements.txt"
            if req_file.exists():
                declared_deps.extend(load_requirements_from_file(req_file))

            # Deduplicate
            declared_deps = list(dict.fromkeys(declared_deps))
            if declared_deps:
                missing = check_missing_dependencies(declared_deps)
                if missing:
                    install_packages(missing)

            # Update component object state
            component.is_installed = True
            component.installed_version = ver

            if self.on_change_callback:
                self.on_change_callback(component.id, "install")
                plugin_file = target_dir / "plugin.json"
                if plugin_file.exists():
                    try:
                        pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                        pid = pdata.get("id")
                        if pid and pid != component.id:
                            self.on_change_callback(pid, "install")
                    except Exception:
                        pass
                if component.target_path:
                    tname = Path(component.target_path).name
                    if tname != component.id:
                        self.on_change_callback(tname, "install")

            return True

        finally:
            if archive_path.exists():
                archive_path.unlink()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def has_configs(self, component: HubComponent) -> bool:
        """Check if any configuration files for this component or its suite children exist on disk."""
        target_dir = self.resolve_target_dir(component)
        meta_file = target_dir / ".theta_component.json"
        children = []
        if meta_file.exists():
            try:
                mdata = json.loads(meta_file.read_text(encoding="utf-8"))
                for cfg_rel in mdata.get("config_paths", []):
                    if (self.workspace_dir / cfg_rel).is_file():
                        return True
                if mdata.get("config_path") and (self.workspace_dir / mdata["config_path"]).is_file():
                    return True
                children.extend(mdata.get("installed_components", []))
                children.extend(mdata.get("dependencies", {}).get("components", []))
            except Exception:
                pass

        if hasattr(component, "extra") and isinstance(component.extra, dict):
            children.extend(component.extra.get("dependencies", {}).get("components", []))

        for child_id in set(children):
            child_comp = self.lookup_component(child_id)
            if child_comp and self.has_configs(child_comp):
                return True

        config_path = self.resolve_config_path(component)
        if config_path and config_path.is_file():
            return True
        return False

    def find_component_for_path(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Given a file or directory path, determine if it belongs to an installed Hub component.

        Returns a dictionary with metadata ('id', 'name', 'kind', 'version', 'target_dir', 'config_paths')
        or None if the path does not belong to an installed component.
        """
        target_path = Path(path)
        if not target_path.is_absolute():
            target_path = (self.workspace_dir / target_path).resolve()
        else:
            target_path = target_path.resolve()

        scan_roots = [
            self.workspace_dir / "in" / "envs",
            self.workspace_dir / "src" / "usr" / "methods",
            self.workspace_dir / "src" / "usr" / "models",
            self.workspace_dir / ".thetaide" / "suites",
            self.workspace_dir / "plugins",
            self.data_dir / "plugins",
        ]
        for root in scan_roots:
            if not root.is_dir():
                continue
            for meta_file in root.rglob(".theta_component.json"):
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    comp_dir = meta_file.parent.resolve()
                    # Check 1: path is the component directory or inside it
                    if target_path == comp_dir or comp_dir in target_path.parents:
                        data["target_dir"] = comp_dir
                        return data
                    # Check 2: path is one of the recorded configs or their parent folder
                    for cfg_rel in data.get("config_paths", []):
                        cfg_full = (self.workspace_dir / cfg_rel).resolve()
                        if target_path == cfg_full or target_path in cfg_full.parents:
                            data["target_dir"] = comp_dir
                            return data
                    if data.get("config_path"):
                        cfg_full = (self.workspace_dir / data["config_path"]).resolve()
                        if target_path == cfg_full or target_path in cfg_full.parents:
                            data["target_dir"] = comp_dir
                            return data
                except Exception:
                    continue
        return None

    def uninstall_by_metadata(self, meta: Dict[str, Any], remove_configs: bool = True) -> bool:
        """Uninstall an installed component given its metadata dict."""
        target_dir = Path(meta["target_dir"])
        recorded_configs = meta.get("config_paths") or ([meta["config_path"]] if meta.get("config_path") else [])

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        if remove_configs and recorded_configs:
            for cfg_rel in recorded_configs:
                p = self.workspace_dir / cfg_rel
                if p.is_file():
                    try:
                        p.unlink()
                        parent = p.parent.resolve()
                        in_cfg = (self.workspace_dir / "in" / "config").resolve()
                        while parent != in_cfg and parent.parent != in_cfg and in_cfg in parent.parents and parent.is_dir() and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                    except Exception:
                        pass

        if self.on_change_callback:
            comp_id = meta.get("id") or target_dir.name
            self.on_change_callback(comp_id, "uninstall")

        children_to_uninstall = meta.get("installed_components") or meta.get("dependencies", {}).get("components", [])
        for child_id in children_to_uninstall:
            child_comp = self.lookup_component(child_id)
            if child_comp:
                try:
                    self.uninstall(child_comp, remove_configs=remove_configs)
                except Exception:
                    pass

        return True

    def uninstall(self, component: HubComponent, remove_configs: bool = True) -> bool:
        """Remove an installed component and optionally its packaged configuration from disk."""
        target_dir = self.resolve_target_dir(component)
        config_path = self.resolve_config_path(component)

        # Check .theta_component.json for recorded config paths before removing target_dir
        recorded_configs = []
        recorded_installed_children = []
        meta_file = target_dir / ".theta_component.json"
        if meta_file.exists():
            try:
                mdata = json.loads(meta_file.read_text(encoding="utf-8"))
                if mdata.get("config_paths"):
                    recorded_configs.extend(mdata["config_paths"])
                elif mdata.get("config_path"):
                    recorded_configs.append(mdata["config_path"])
                if mdata.get("installed_components"):
                    recorded_installed_children.extend(mdata["installed_components"])
                if not recorded_installed_children and mdata.get("dependencies", {}).get("components"):
                    recorded_installed_children.extend(mdata["dependencies"]["components"])
            except Exception:
                pass

        if not recorded_installed_children and hasattr(component, "extra") and isinstance(component.extra, dict):
            recorded_installed_children.extend(component.extra.get("dependencies", {}).get("components", []))

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        # If this was a suite with recorded child components, uninstall them as well
        for child_id in recorded_installed_children:
            child_comp = self.lookup_component(child_id)
            if child_comp:
                try:
                    self.uninstall(child_comp, remove_configs=remove_configs)
                except Exception:
                    pass

        # Remove deployed config YAML(s) in in/config/ if remove_configs is True
        if remove_configs:
            if recorded_configs:
                for cfg_rel in recorded_configs:
                    p = self.workspace_dir / cfg_rel
                    if p.is_file():
                        try:
                            p.unlink()
                            parent = p.parent.resolve()
                            in_cfg = (self.workspace_dir / "in" / "config").resolve()
                            while parent != in_cfg and parent.parent != in_cfg and in_cfg in parent.parents and parent.is_dir() and not any(parent.iterdir()):
                                parent.rmdir()
                                parent = parent.parent
                        except Exception:
                            pass
            elif config_path and config_path.is_file():
                try:
                    config_path.unlink()
                    parent = config_path.parent.resolve()
                    in_cfg = (self.workspace_dir / "in" / "config").resolve()
                    while parent != in_cfg and parent.parent != in_cfg and in_cfg in parent.parents and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                except Exception:
                    pass
            elif config_path and config_path.is_file():
                try:
                    config_path.unlink()
                except Exception:
                    pass

        component.is_installed = False
        component.installed_version = None

        if self.on_change_callback:
            self.on_change_callback(component.id, "uninstall")
            if component.target_path:
                tname = Path(component.target_path).name
                if tname != component.id:
                    self.on_change_callback(tname, "uninstall")

        return True
