"""ThetaIDE Plugin Framework."""
from .base import Plugin, PluginManifest
from .context import PluginContext
from .manager import PluginManager
from .slots import ExtensionSlot, ExtensionSlotRegistry, ExtensionSlotWidget
from .commands import Command, CommandRegistry

__all__ = [
    "Plugin",
    "PluginManifest",
    "PluginContext",
    "PluginManager",
    "ExtensionSlot",
    "ExtensionSlotRegistry",
    "ExtensionSlotWidget",
    "Command",
    "CommandRegistry",
]
