"""Actions and dialogs for managing components: opening source code in editor and uninstalling/deleting components."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


def find_workspace_root(start_path: Optional[Path] = None) -> Path:
    """Find the workspace root by looking for 'in' and 'src' directories upwards."""
    curr = (start_path or Path.cwd()).resolve()
    if curr.is_file():
        curr = curr.parent

    for p in [curr, *curr.parents]:
        if (p / "in").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd().resolve()


def find_component_assets(
    config_path: Path,
    rel_path: Optional[str] = None,
    workspace_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve all associated assets for a component given its configuration YAML path.

    Returns a dict with:
        comp_id: Component identifier stem
        comp_name: Display name
        kind: Component category (model, agent, env, etc.)
        config_path: Path to the active config YAML
        code_dir: Path to directory containing source code (if any)
        primary_code_file: Path to the most relevant .py file to open in an editor
        files_to_delete: List of Path objects (config files, source directories, rules)
        is_component: True if this is an installed/managed component, False if a loose config
        action_type: 'uninstall' or 'delete_config'
        meta: Parsed .theta_component.json dict if managed by Hub / scaffolder
    """
    ws = workspace_dir or find_workspace_root(config_path)
    cfg = Path(config_path).resolve()
    comp_id = cfg.stem

    # 1. Attempt lookup via HubInstaller / .theta_component.json
    meta: Optional[Dict[str, Any]] = None
    try:
        from frontend.hub.installer import HubInstaller
        inst = HubInstaller(workspace_dir=ws, data_dir=ws / ".thetaide")
        meta = inst.find_component_for_path(cfg)
    except Exception:
        meta = None

    code_dir: Optional[Path] = None
    primary_code_file: Optional[Path] = None
    files_to_delete: List[Path] = [cfg]
    kind = "component"
    comp_name = comp_id
    is_component = False

    if meta:
        is_component = True
        comp_id = meta.get("id") or comp_id
        comp_name = meta.get("name") or comp_id
        kind = meta.get("kind") or kind
        if meta.get("target_dir"):
            code_dir = Path(meta["target_dir"]).resolve()
            if code_dir.exists() and code_dir not in files_to_delete:
                files_to_delete.append(code_dir)

        # Include all declared config paths
        for c_rel in meta.get("config_paths", []):
            cp = (ws / c_rel).resolve()
            if cp.exists() and cp not in files_to_delete:
                files_to_delete.append(cp)
        if meta.get("config_path"):
            cp = (ws / meta["config_path"]).resolve()
            if cp.exists() and cp not in files_to_delete:
                files_to_delete.append(cp)

    # 2. Heuristic fallback resolution if no metadata found
    rel = rel_path or str(cfg.relative_to(ws / "in" / "config") if (ws / "in" / "config") in cfg.parents else cfg.name)
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.startswith("in/config/"):
        rel = rel[len("in/config/"):]

    if not code_dir:
        candidate_dir: Optional[Path] = None
        candidate_file: Optional[Path] = None

        if rel.startswith("model/"):
            kind = "model"
            candidate_dir = ws / "src" / "usr" / "models" / comp_id
            candidate_file = ws / "src" / "usr" / "models" / f"{comp_id}.py"
        elif rel.startswith("agent/"):
            kind = "agent"
            candidate_dir = ws / "src" / "usr" / "methods" / comp_id
            candidate_file = ws / "src" / "usr" / "methods" / f"{comp_id}.py"
        elif rel.startswith("env/"):
            kind = "env"
            candidate_dir = ws / "in" / "envs" / comp_id
            rules_dir = ws / "in" / "rules" / comp_id
            if rules_dir.is_dir() and rules_dir not in files_to_delete:
                files_to_delete.append(rules_dir)
        elif rel.startswith("data/") or rel.startswith("datamodule/"):
            kind = "datamodule"
            candidate_dir = ws / "src" / "usr" / "data" / comp_id
        elif rel.startswith("eval/") or rel.startswith("evaluator/"):
            kind = "evaluator"
            candidate_dir = ws / "src" / "usr" / "eval" / comp_id

        if candidate_dir and candidate_dir.is_dir():
            code_dir = candidate_dir
            is_component = True
            if code_dir not in files_to_delete:
                files_to_delete.append(code_dir)
        elif candidate_file and candidate_file.is_file():
            primary_code_file = candidate_file

    # 3. If the configs are located in a dedicated subfolder (e.g. in/config/env/atari),
    # ensure any sibling configs belonging to the component suite are included.
    cfg_parent = cfg.parent
    in_config = (ws / "in" / "config").resolve()
    if is_component and cfg_parent != in_config and in_config in cfg_parent.parents:
        # Check if parent folder name matches component or target directory name
        folder_matches = (
            cfg_parent.name == comp_id or
            (code_dir and cfg_parent.name == code_dir.name) or
            (meta and meta.get("id") and cfg_parent.name in meta["id"])
        )
        if folder_matches:
            for sibling_yaml in cfg_parent.glob("*.yaml"):
                if sibling_yaml not in files_to_delete:
                    files_to_delete.append(sibling_yaml)
            for sibling_yml in cfg_parent.glob("*.yml"):
                if sibling_yml not in files_to_delete:
                    files_to_delete.append(sibling_yml)

    # 4. Determine the primary code file to open in the editor
    if not primary_code_file and code_dir and code_dir.is_dir():
        # Check for specific naming patterns: <comp_id>_model.py, env_vectorized.py, env.py, etc.
        named_candidates = [
            code_dir / f"{comp_id}_model.py",
            code_dir / f"{comp_id}_agent.py",
            code_dir / "env_vectorized.py",
            code_dir / "env.py",
            code_dir / f"{comp_id}_evaluator.py",
            code_dir / f"{comp_id}_datamodule.py",
            code_dir / f"{comp_id}.py",
        ]
        for c in named_candidates:
            if c.is_file():
                primary_code_file = c
                break

        # If none matched, check for any non-__init__.py python file
        if not primary_code_file:
            py_files = sorted([f for f in code_dir.glob("*.py") if f.name != "__init__.py"])
            if py_files:
                primary_code_file = py_files[0]
            elif (code_dir / "__init__.py").is_file():
                primary_code_file = code_dir / "__init__.py"

    is_hub_installed = bool(meta and meta.get("installed_from"))
    action_type = "uninstall" if is_hub_installed else ("delete_component" if is_component else "delete_config")

    return {
        "comp_id": comp_id,
        "comp_name": comp_name,
        "kind": kind,
        "config_path": cfg,
        "code_dir": code_dir,
        "primary_code_file": primary_code_file or code_dir or cfg,
        "files_to_delete": files_to_delete,
        "is_component": is_component,
        "is_hub_installed": is_hub_installed,
        "action_type": action_type,
        "meta": meta,
    }


def open_in_editor(path: Path) -> bool:
    """Open a file or directory in the user's preferred or default code editor."""
    resolved = path.resolve()
    path_str = str(resolved)

    # 1. Check $VISUAL or $EDITOR environment variables
    env_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if env_editor:
        try:
            import shlex
            cmd = shlex.split(env_editor) + [path_str]
            subprocess.Popen(cmd)
            return True
        except Exception:
            pass

    # 2. Check for common code editors in PATH
    for editor_cmd in ("code", "cursor", "subl"):
        if shutil.which(editor_cmd):
            try:
                subprocess.Popen([editor_cmd, path_str])
                return True
            except Exception:
                pass

    # 3. macOS: use native 'open'
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", path_str])
            return True
        except Exception:
            pass

    # 4. Windows: use 'os.startfile'
    if sys.platform == "win32":
        try:
            os.startfile(path_str)
            return True
        except Exception:
            pass

    # 5. Linux: use xdg-open
    if shutil.which("xdg-open"):
        try:
            subprocess.Popen(["xdg-open", path_str])
            return True
        except Exception:
            pass

    # 6. Fallback to Qt QDesktopServices
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))


class DeleteComponentDialog(QDialog):
    """Confirmation dialog when uninstalling a component or deleting a config from the components panel.

    Prompts to confirm, lists all files/folders that will be permanently removed,
    and includes a 'Don't ask again' checkbox.
    """

    def __init__(
        self,
        comp_name: str,
        files_to_delete: List[Path],
        is_component: bool = True,
        is_hub_installed: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.is_component = is_component
        self.is_hub_installed = is_hub_installed
        self.dont_ask_again = False

        if is_hub_installed:
            self.setWindowTitle("Uninstall Component")
            action_verb = "Uninstall"
        elif is_component:
            self.setWindowTitle("Delete Component")
            action_verb = "Delete"
        else:
            self.setWindowTitle("Delete Configuration")
            action_verb = "Delete"

        self.setFixedWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(22, 20, 22, 20)

        title = QLabel(f"{action_verb} {comp_name}?")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ebdbb2;")
        layout.addWidget(title)

        configs_count = len([f for f in files_to_delete if f.is_file() and f.suffix in (".yaml", ".yml")])
        if is_hub_installed:
            if configs_count > 1:
                desc_text = (
                    f"This file is part of the Hub-installed component suite <b>{comp_name}</b>, which provides "
                    f"<b>{configs_count} configurations</b>.<br><br>"
                    f"Uninstalling will remove the entire component suite, including its source code and all configurations:"
                )
            else:
                desc_text = (
                    f"Are you sure you want to uninstall Hub component <b>{comp_name}</b>?<br>"
                    "This action cannot be undone. Its source code directory and configuration will be permanently removed:"
                )
        elif is_component:
            if configs_count > 1:
                desc_text = (
                    f"This file is part of local component <b>{comp_name}</b>, which provides "
                    f"<b>{configs_count} configurations</b>.<br><br>"
                    f"Are you sure you want to delete this component, including its source code directory and all configurations?"
                )
            else:
                desc_text = (
                    f"Are you sure you want to delete component <b>{comp_name}</b>?<br>"
                    "This action cannot be undone. Its source code directory and configuration file will be permanently deleted:"
                )
        else:
            desc_text = (
                f"Are you sure you want to delete configuration <b>'{comp_name}'</b>?<br>"
                "This action cannot be undone. The configuration file will be permanently removed from disk:"
            )

        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a89984; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        # File box listing all items to delete
        files_frame = QFrame()
        files_frame.setStyleSheet(
            "background-color: #1d2021; border: 1px solid #3c3836; border-radius: 4px; padding: 6px;"
        )
        fb_layout = QVBoxLayout(files_frame)
        fb_layout.setSpacing(4)
        fb_layout.setContentsMargins(8, 6, 8, 6)

        ws = find_workspace_root()
        for f in files_to_delete:
            try:
                rel = f.relative_to(ws)
            except ValueError:
                rel = f
            suffix = "/" if f.is_dir() else ""
            lbl_f = QLabel(f"• {rel}{suffix}")
            lbl_f.setStyleSheet("color: #ebdbb2; font-family: monospace; font-size: 11px;")
            lbl_f.setWordWrap(True)
            fb_layout.addWidget(lbl_f)

        layout.addWidget(files_frame)

        self.chk_dont_ask = QCheckBox("Don't ask again")
        self.chk_dont_ask.setStyleSheet("color: #a89984; font-size: 12px; margin-top: 4px;")
        layout.addWidget(self.chk_dont_ask)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        btn_action = QPushButton(f"{action_verb} Permanently")
        btn_action.setStyleSheet(
            "background-color: #cc241d; color: #ebdbb2; font-weight: bold; padding: 6px 14px; border-radius: 4px;"
        )
        btn_action.clicked.connect(self._on_confirm)
        btn_row.addWidget(btn_action)

        layout.addLayout(btn_row)

    def _on_confirm(self):
        self.dont_ask_again = self.chk_dont_ask.isChecked()
        self.accept()
