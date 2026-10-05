"""Unified settings management for Theta-IDE.

A single ``settings.toml`` file is the canonical source of truth for all user
preferences (theme, sidebar layout, enabled plugins, backend URL, etc.).

Two locations are merged in priority order:
  1. User global: ``~/.config/thetaide/settings.toml``
  2. Workspace:   ``<data_dir>/settings.toml``

Workspace values override user-global values for every key they define.

The file is watched with ``QFileSystemWatcher`` so that editing it in an
external editor (Neovim, nano, VS Code …) applies changes live without a
restart.

GUI interactions that mutate a preference call :meth:`SettingsManager.set`,
which writes the change back into the workspace ``settings.toml`` and emits
:attr:`changed`, which in turn the rest of the app listens to.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QFileSystemWatcher, QObject, pyqtSignal

from .panels import DEFAULT_HOTKEY_PANES, DEFAULT_VISIBLE, PANEL_IDS, panel_title, toml_list


# ---------------------------------------------------------------------------
# Default settings — these are written verbatim into a fresh settings.toml
# (with the comments intact) on first run.
# ---------------------------------------------------------------------------

_DEFAULT_TOML = """\
# =============================================================================
# Theta-IDE Settings  (settings.toml)
# =============================================================================
# Edit this file directly in your favourite text editor; Theta-IDE will pick up
# changes automatically without restarting.
#
# Priority order (workspace overrides user-global):
#   ~/.config/thetaide/settings.toml   — user-global defaults
#   <workspace>/.thetaide/settings.toml — this file (workspace overrides)

[appearance]
# Active color theme.  Built-in choices:
#   "Gruvbox Dark", "Gruvbox Light", "Nord", "Dracula", "Catppuccin",
#   "Catppuccin Macchiato", "Catppuccin Latte", "Paper",
#   "Apollo", "Athena", "Ares", "Dionysus", "Poseidon"
theme = "Catppuccin"

# Enable the rotating 3-D ASCII sculpture on the Settings & About panel.
ascii_animation = true

[backend]
# FastAPI backend that manages training runs and pipelines.
url = "http://127.0.0.1:8000"

# Seconds before a connection-test attempt times out.
timeout = 5

[sidebar]
# Canonical panel IDs, in left-to-right display order.
order = @@SIDEBAR_ORDER@@

# Subset of order that is shown by default.  Hidden panels are still available
# via View → <Panel Name>.
visible = @@SIDEBAR_VISIBLE@@

[plugins]
# List of plugin IDs that are currently enabled.
enabled = []

[hotkeys]
# Enable custom keyboard shortcuts.
enabled = true

# Primary action key (leader / chord modifier).
# Press or hold this key, then press a number key to switch panes.
# Default: "ctrl+b" (tmux-style prefix). Also supports: "caps_lock", "alt", "ctrl", "meta", "ctrl+tab", etc.
action_key = "ctrl+b"

# Leader timeout in seconds for sequential presses (e.g. tap Action key, then press number).
leader_timeout = 1.5

# When true, the Terminal pane takes precedence over IDE action hotkeys so tmux
# and shell keybindings (e.g. tmux prefix, ctrl/caps shortcuts) work uninterrupted.
terminal_precedence = true

# Pane navigation shortcuts (Action key + number):
# Maps numbers [0, 1, 2, ...] to pane IDs.
@@HOTKEY_PANES_COMMENT@@panes = @@HOTKEY_PANES@@
"""
# Panel lists come from frontend/panels.py, the single definition of the sidebar panels.
_DEFAULT_TOML = (
    _DEFAULT_TOML.replace("@@SIDEBAR_ORDER@@", toml_list(PANEL_IDS))
    .replace("@@SIDEBAR_VISIBLE@@", toml_list(DEFAULT_VISIBLE))
    .replace("@@HOTKEY_PANES_COMMENT@@",
             "".join(f'#   {n} -> "{pane}" ({panel_title(pane)})\n' for n, pane in enumerate(DEFAULT_HOTKEY_PANES)))
    .replace("@@HOTKEY_PANES@@", toml_list(DEFAULT_HOTKEY_PANES))
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *override* merged on top of *base* (deep)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_toml_file(path: Path) -> dict:
    """Read and parse a TOML file; return an empty dict on any error."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _write_toml_file(path: Path, data: dict) -> None:
    """Serialise *data* as minimal TOML and write it atomically.

    This is a best-effort writer that handles the flat-to-two-level structure
    used by Theta-IDE settings.  It does **not** preserve user comments (that
    would require ``tomlkit``), but it never corrupts the file.
    """
    lines: list[str] = []
    for section, value in data.items():
        if not isinstance(value, dict):
            # Top-level scalar (unusual but tolerated)
            lines.append(f"{section} = {_toml_value(value)}")
        else:
            lines.append(f"\n[{section}]")
            for k, v in value.items():
                lines.append(f"{k} = {_toml_value(v)}")
    content = "\n".join(lines).lstrip("\n") + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _toml_value(v: Any) -> str:
    """Convert a Python value to its TOML literal representation."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return repr(v)  # fallback


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------

class SettingsManager(QObject):
    """Single source of truth for all Theta-IDE user preferences.

    Signals
    -------
    changed
        Emitted whenever any preference changes, whether caused by an in-app
        GUI action or a live file-system edit.
    """

    changed = pyqtSignal()

    def __init__(self, data_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # Canonical workspace settings file sits one level above data_dir
        # (e.g. data_dir=.thetaide/runs  →  .thetaide/settings.toml),
        # but stays inside data_dir for custom or temporary directories.
        p = Path(data_dir)
        if p.name == "runs" or p.parent.name == ".thetaide":
            self._workspace_path = p.parent / "settings.toml"
        else:
            self._workspace_path = p / "settings.toml"
        # User-global settings file (lower priority)
        self._user_path = Path.home() / ".config" / "thetaide" / "settings.toml"

        # Merged, in-memory view of all settings
        self._data: dict = {}

        # Initialise defaults then load from disk
        self._bootstrap_workspace_file()
        self._reload()

        # Watch both files for live edits
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watch_files()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """Return a (possibly nested) value by dot-path keys.

        Example::

            mgr.get("appearance", "theme")
        """
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, None)
            if node is None:
                return default
        return node

    def section(self, key: str) -> dict:
        """Return a whole top-level section as a dict (or empty dict)."""
        return dict(self._data.get(key, {}))

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def set(self, *keys_and_value: Any) -> None:
        """Set a (possibly nested) preference and persist to the workspace file.

        The last positional argument is the value; all preceding arguments are
        the key path.

        Example::

            mgr.set("appearance", "theme", "Nord")
            mgr.set("sidebar", "visible", ["components", "config"])
        """
        if len(keys_and_value) < 2:  # noqa: PLR2004
            raise ValueError("set() requires at least one key and a value.")
        *keys, value = keys_and_value

        # Update the in-memory merged view
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

        # Persist to the workspace file (load → patch → write)
        on_disk = _parse_toml_file(self._workspace_path)
        disk_node = on_disk
        for k in keys[:-1]:
            disk_node = disk_node.setdefault(k, {})
        disk_node[keys[-1]] = value

        # Temporarily pause the watcher so our own write doesn't re-trigger reload
        self._watcher.removePath(str(self._workspace_path))
        try:
            _write_toml_file(self._workspace_path, on_disk)
        finally:
            self._watcher.addPath(str(self._workspace_path))

        self.changed.emit()

    # ------------------------------------------------------------------
    # Convenience helpers for the most-touched sections
    # ------------------------------------------------------------------

    @property
    def theme(self) -> str:
        return self.get("appearance", "theme", default="Catppuccin")

    @property
    def ascii_animation(self) -> bool:
        return bool(self.get("appearance", "ascii_animation", default=True))

    @property
    def backend_url(self) -> str:
        return self.get("backend", "url", default="http://127.0.0.1:8000")

    @property
    def sidebar_order(self) -> list[str]:
        v = self.get("sidebar", "order", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def sidebar_visible(self) -> list[str]:
        v = self.get("sidebar", "visible", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def plugins_enabled(self) -> list[str]:
        v = self.get("plugins", "enabled", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def hotkeys_enabled(self) -> bool:
        return bool(self.get("hotkeys", "enabled", default=True))

    @property
    def action_key(self) -> str:
        return str(self.get("hotkeys", "action_key", default="ctrl+b"))

    @property
    def hotkeys_leader_timeout(self) -> float:
        try:
            return float(self.get("hotkeys", "leader_timeout", default=1.5))
        except (ValueError, TypeError):
            return 1.5

    @property
    def hotkeys_terminal_precedence(self) -> bool:
        return bool(self.get("hotkeys", "terminal_precedence", default=True))

    @property
    def hotkey_panes(self) -> list[str]:
        v = self.get("hotkeys", "panes", default=None)
        if isinstance(v, list) and v:
            return list(v)
        # Default order: settings, followed by sidebar panels in order
        return list(DEFAULT_HOTKEY_PANES)

    @property
    def workspace_settings_path(self) -> Path:
        """Absolute path to the workspace ``settings.toml``."""
        return self._workspace_path


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_workspace_file(self) -> None:
        """Write the default settings.toml if it doesn't exist yet."""
        if not self._workspace_path.exists():
            self._workspace_path.parent.mkdir(parents=True, exist_ok=True)
            self._workspace_path.write_text(_DEFAULT_TOML, encoding="utf-8")

    def _reload(self) -> None:
        """Merge user-global and workspace files into the in-memory view."""
        user_data = _parse_toml_file(self._user_path)
        workspace_data = _parse_toml_file(self._workspace_path)
        self._data = _deep_merge(user_data, workspace_data)

    def _watch_files(self) -> None:
        for p in (self._user_path, self._workspace_path):
            if p.exists():
                self._watcher.addPath(str(p))

    def _on_file_changed(self, _path: str) -> None:
        """Called by QFileSystemWatcher when any watched settings file changes."""
        # Some editors (Vim, Neovim) replace the file via rename; re-watch.
        self._watcher.removePath(str(self._workspace_path))
        self._watcher.removePath(str(self._user_path))
        self._reload()
        self._watch_files()
        self.changed.emit()
