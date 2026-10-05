"""The sidebar panels, defined once.

The sidebar tabs, the Settings & About visibility toggles, the View menu, the "Restore default sidebar"
order, the default settings.toml (sidebar order/visibility and hotkey panes) and the hotkey help all read
from PANELS. Adding a panel means adding a row here and building its widget in app.py.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Panel:
    id: str  # identifier used in settings.toml, hotkeys and SideTabs
    title: str  # full name: sidebar tooltip, Settings toggle and View menu
    short: str  # label under the sidebar icon
    icon: str  # SVG in frontend/icons/
    description: str  # subtitle of the Settings toggle
    visible: bool  # shown in the sidebar on a fresh install


PANELS = (
    Panel("components", "Components", "Components", "components",
          "Modular configs (agent, env, model, paradigms, site, hydra)", True),
    Panel("config", "Experiment builder", "Experiment", "config", "Hydra configurations & hyperparameter tuner", True),
    Panel("monitor", "Training monitor", "Monitor", "monitor", "Overview charts & live training curves", True),
    Panel("results", "Results browser", "Results", "results", "Experiment runs, comparison, & metrics", False),
    Panel("plots", "Plot viewer", "Plots", "plots", "Saved figure plots & multi-seed comparisons", False),
    Panel("tensorboard", "TensorBoard", "TensorBoard", "tensorboard", "Interactive TensorBoard event visualizer", False),
    Panel("queue", "Job queue", "Queue", "queue", "Local sequential run scheduler & manager", False),
    Panel("terminal", "Terminal", "Terminal", "terminal", "Embedded terminal shell", True),
    Panel("console", "Console", "Console", "console", "Live Theta IDE system log & command line", True),
)
PANELS_BY_ID = {panel.id: panel for panel in PANELS}
PANEL_IDS = tuple(panel.id for panel in PANELS)  # default sidebar order
DEFAULT_VISIBLE = tuple(panel.id for panel in PANELS if panel.visible)

# The Settings & About page is reached from the sidebar's logo, not a tab, but has a hotkey (Action + 0).
SETTINGS_ID, SETTINGS_TITLE = "settings", "Settings & About"
DEFAULT_HOTKEY_PANES = (SETTINGS_ID, *PANEL_IDS)  # Action + 0..9


def panel_title(panel_id):
    """Display name of a panel id (including the Settings page); unknown ids are returned unchanged."""
    if panel_id == SETTINGS_ID:
        return SETTINGS_TITLE
    panel = PANELS_BY_ID.get(panel_id)
    return panel.title if panel else panel_id


def toml_list(items, indent="    "):
    """A TOML array with one quoted item per line, as written in the default settings.toml."""
    return "[\n" + "".join(f'{indent}"{item}",\n' for item in items) + "]"
