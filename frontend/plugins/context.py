"""PluginContext host API facade providing controlled IDE access to plugins."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from ..app import Window


class PluginContext:
    """The host execution context passed to a Plugin when it is activated."""

    def __init__(self, plugin_id: str, window: Window, storage_file: Path):
        self.plugin_id = plugin_id
        self._window = window
        self._storage_file = storage_file
        self._registered_tabs: List[str] = []
        self._experiment_listeners: List[Callable[[Optional[str], Optional[str]], None]] = []
        self._storage_cache: Optional[Dict[str, Any]] = None  # lazy-loaded on first access

    # ── Sidebar & UI Contributions ──────────────────────────────────────────

    def add_sidebar_tab(
        self,
        tab_id: str,
        widget: QWidget,
        title: str,
        icon_name: Optional[str] = None,
        short_label: Optional[str] = None,
    ) -> None:
        """Add a panel to the IDE's primary vertical sidebar.
        
        Args:
            tab_id: Unique string identifier for this tab.
            widget: QWidget to display when tab is selected.
            title: Full display title (and tooltip).
            icon_name: SVG icon name in frontend/icons/ or absolute path to an .svg.
            short_label: Short text label under the icon in the sidebar (defaults to title).
        """
        resolved_icon = icon_name
        if resolved_icon and hasattr(self._window, "plugin_manager"):
            manifest = self._window.plugin_manager.manifests.get(self.plugin_id)
            if manifest and manifest.plugin_dir:
                p_cand = Path(resolved_icon)
                builtin_icon_dir = Path(__file__).resolve().parent.parent / "icons"
                if not (p_cand.is_file() or (builtin_icon_dir / f"{resolved_icon}.svg").is_file()):
                    for cand in [
                        manifest.plugin_dir / resolved_icon,
                        manifest.plugin_dir / f"{resolved_icon}.svg",
                        manifest.plugin_dir / f"{self.plugin_id}.svg",
                        manifest.plugin_dir / f"{manifest.id}.svg",
                    ]:
                        if cand.is_file():
                            resolved_icon = str(cand)
                            break

        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            self._window.tabs.addTab(
                widget=widget,
                text=title,
                icon=resolved_icon,
                short=short_label or title,
                tab_id=tab_id,
            )
            if tab_id not in self._registered_tabs:
                self._registered_tabs.append(tab_id)

    def remove_sidebar_tab(self, tab_id: str) -> None:
        """Remove a contributed panel from the sidebar."""
        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            self._window.tabs.removeTab(tab_id)
        if tab_id in self._registered_tabs:
            self._registered_tabs.remove(tab_id)

    def select_sidebar_tab(self, tab_id: str) -> None:
        """Switch the active view to the specified sidebar tab."""
        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            entry = self._window.tabs.tabs.get(tab_id)
            if entry and "widget" in entry:
                self._window.tabs.setCurrentWidget(entry["widget"])

    # ── Workspace & Environment ──────────────────────────────────────────────

    def get_workspace_dir(self) -> Path:
        """Return the root workspace directory."""
        if hasattr(self._window, "store"):
            root = getattr(self._window.store, "root", getattr(self._window.store, "data_dir", None))
            if root:
                return Path(root).resolve().parent
        return Path.cwd()

    def get_data_dir(self) -> Path:
        """Return the ThetaIDE internal data/storage directory."""
        if hasattr(self._window, "store"):
            root = getattr(self._window.store, "root", getattr(self._window.store, "data_dir", None))
            if root:
                return Path(root)
        return Path.cwd() / ".thetaide"

    def log(self, message: str) -> None:
        """Log a message to the IDE console / status log."""
        prefix = f"[{self.plugin_id}]"
        if hasattr(self._window, "log") and callable(self._window.log):
            self._window.log(f"{prefix} {message}")
        else:
            print(f"{prefix} {message}")

    def show_status_message(self, message: str, timeout_ms: int = 4000) -> None:
        """Display a transient message on the main window status bar."""
        if hasattr(self._window, "statusBar"):
            status_bar = self._window.statusBar()
            if status_bar:
                status_bar.showMessage(message, timeout_ms)

    # ── Experiment State & Event Subscriptions ───────────────────────────────

    def get_current_experiment(self) -> Optional[str]:
        """Return the currently selected experiment name, e.g. 'cartpole/quick_test'."""
        return getattr(self._window, "current_experiment", None)

    def get_active_config_path(self) -> Optional[Path]:
        """Return the Path to the currently active config file, if any."""
        path = getattr(self._window, "active_config_path", None)
        return Path(path) if path else None

    def add_experiment_change_listener(
        self, callback: Callable[[Optional[str], Optional[str]], None]
    ) -> None:
        """Register a callback for when the active experiment config changes.
        
        Callback signature: callback(experiment_name: str | None, file_path: str | None)
        """
        if callback not in self._experiment_listeners:
            self._experiment_listeners.append(callback)

    def remove_experiment_change_listener(
        self, callback: Callable[[Optional[str], Optional[str]], None]
    ) -> None:
        """Unregister an experiment change listener."""
        if callback in self._experiment_listeners:
            self._experiment_listeners.remove(callback)

    def notify_experiment_changed(
        self, exp_name: Optional[str], file_path: Optional[str]
    ) -> None:
        """Internal dispatch called by Window when user selects a new config."""
        for listener in list(self._experiment_listeners):
            try:
                listener(exp_name, file_path)
            except Exception as exc:
                self.log(f"Error in experiment change listener: {exc}")

    # ── Plugin-Scoped Persistence ────────────────────────────────────────────

    def _read_storage(self) -> Dict[str, Any]:
        if self._storage_cache is not None:
            return self._storage_cache
        if not self._storage_file.exists():
            self._storage_cache = {}
            return self._storage_cache
        try:
            self._storage_cache = json.loads(self._storage_file.read_text(encoding="utf-8"))
        except Exception:
            self._storage_cache = {}
        return self._storage_cache

    def _write_storage(self, data: Dict[str, Any]) -> None:
        self._storage_cache = data
        try:
            self._storage_file.parent.mkdir(parents=True, exist_ok=True)
            self._storage_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.log(f"Failed to save plugin storage: {exc}")

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieve a saved setting value for this plugin."""
        data = self._read_storage()
        return data.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Save a setting value for this plugin."""
        data = self._read_storage()
        data[key] = value
        self._write_storage(data)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove any remaining tabs and listeners registered by this context."""
        for tab_id in list(self._registered_tabs):
            self.remove_sidebar_tab(tab_id)
        self._experiment_listeners.clear()
        self._storage_cache = None
