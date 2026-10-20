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


# Re-export headless asset discovery functions from core services
from src.app.services.component_exporter import find_component_assets, find_workspace_root


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
                    f"Uninstalling removes the source code and unchanged bundled YAMLs. Edited YAMLs are preserved, "
                    f"and any files that existed before installation are restored:"
                )
            else:
                desc_text = (
                    f"Are you sure you want to uninstall Hub component <b>{comp_name}</b>?<br>"
                    "The source code and unchanged bundled YAML will be removed. Edited YAML is preserved, and any "
                    "file that existed before installation is restored:"
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
