"""Command registry for plugins to contribute commands, hotkeys, and launcher items."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class Command:
    """Specification of an executable command contributed by a plugin or core module."""
    command_id: str
    title: str
    callback: Callable[[], None]
    default_hotkey: Optional[str] = None
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    category: str = "Plugin"
    icon: Optional[str] = None
    plugin_id: Optional[str] = None
    detail: str = ""


class CommandRegistry(QObject):
    """Central registry of executable commands reachable via launcher, hotkeys, or menus."""
    changed = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._commands: Dict[str, Command] = {}

    def register_command(self, cmd: Command) -> None:
        """Register a command in the registry."""
        self._commands[cmd.command_id] = cmd
        self.changed.emit()

    def unregister_command(self, command_id: str) -> None:
        """Remove a command from the registry."""
        if command_id in self._commands:
            del self._commands[command_id]
            self.changed.emit()

    def get_command(self, command_id: str) -> Optional[Command]:
        """Find a command by its unique ID."""
        return self._commands.get(command_id)

    def get_commands(self) -> List[Command]:
        """Return all registered commands."""
        return list(self._commands.values())

    def clear(self) -> None:
        """Remove all registered commands."""
        self._commands.clear()
        self.changed.emit()
