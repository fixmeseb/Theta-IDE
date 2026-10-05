"""The sidebar panel registry (frontend/panels.py) and the UI built from it."""
import os
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.hotkeys import PANE_ALIASES
    from frontend.panels import (DEFAULT_HOTKEY_PANES, DEFAULT_VISIBLE, PANEL_IDS, PANELS, SETTINGS_TITLE,
                                 panel_title)
    from frontend.settings import _DEFAULT_TOML
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestPanelRegistry(unittest.TestCase):
    def test_registry_is_consistent(self):
        self.assertEqual(len(set(PANEL_IDS)), len(PANELS))
        self.assertEqual(PANEL_IDS, ("components", "config", "monitor", "results", "plots", "tensorboard", "queue",
                                     "terminal", "console"))
        self.assertEqual(DEFAULT_HOTKEY_PANES, ("settings", *PANEL_IDS))
        for panel in PANELS:
            self.assertTrue((Path("frontend/icons") / f"{panel.icon}.svg").is_file(), panel.icon)

    def test_default_settings_come_from_the_registry(self):
        settings = tomllib.loads(_DEFAULT_TOML)
        self.assertEqual(settings["sidebar"]["order"], list(PANEL_IDS))
        self.assertEqual(settings["sidebar"]["visible"], list(DEFAULT_VISIBLE))
        self.assertEqual(settings["hotkeys"]["panes"], list(DEFAULT_HOTKEY_PANES))
        self.assertIn('#   2 -> "config" (Experiment builder)', _DEFAULT_TOML)

    def test_every_pane_id_is_a_hotkey_alias_of_itself(self):
        for pane in DEFAULT_HOTKEY_PANES:
            self.assertEqual(PANE_ALIASES[pane], pane)
        self.assertEqual(PANE_ALIASES["tb"], "tensorboard")


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestWindowPanels(unittest.TestCase):
    def make_window(self, settings_text=None):
        self.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_dir.name) / "runs"  # settings.toml lands in the private temp dir
        if settings_text:
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            (data_dir.parent / "settings.toml").write_text(settings_text, encoding="utf-8")
        self.window = Window(data_dir, api_url="http://127.0.0.1:1")
        return self.window

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def hotkey_help(self, window):
        text = next(label.text() for label in window.settings_panel.findChildren(QLabel) if "Action + 0" in label.text())
        return re.findall(r"Action \+ (\d)</code> ➔ ([^<]+)", text)

    def test_tabs_toggles_menu_and_reset_follow_the_registry(self):
        window = self.make_window()
        self.assertEqual(list(window.tabs.tab_order), list(PANEL_IDS))
        for panel in PANELS:
            entry = window.tabs.tabs[panel.id]
            self.assertEqual((entry["button"].toolTip(), entry["button"].text()), (panel.title, panel.short))
        self.assertEqual(list(window.pane_sliders), list(PANEL_IDS))
        view = next(a.menu() for a in window.menuBar().actions() if a.text() == "View")
        titles = [a.text() for a in view.actions()]
        start = titles.index(PANELS[0].title)
        self.assertEqual(titles[start:start + len(PANELS) + 1], [p.title for p in PANELS] + [SETTINGS_TITLE])
        next(a for a in view.actions() if a.text() == "Plot viewer").trigger()
        self.assertIs(window.tabs.currentWidget(), window.plot_viewer)
        window.tabs.move_tab_to_visible_slot("terminal", 0)
        window.reset_sidebar_layout()
        self.assertEqual(list(window.tabs.tab_order), list(PANEL_IDS))

    def test_hotkey_help_shows_default_mapping(self):
        window = self.make_window()
        self.assertEqual(self.hotkey_help(window),
                         [(str(n), panel_title(p)) for n, p in sorted(enumerate(DEFAULT_HOTKEY_PANES),
                                                                      key=lambda item: (item[0] % 5, item[0]))])

    def test_hotkey_help_follows_custom_hotkey_panes(self):
        custom = _DEFAULT_TOML.replace(
            'panes = [\n    "settings",\n    "components",', 'panes = [\n    "settings",\n    "terminal",\n    "components",')
        window = self.make_window(custom)
        help_map = dict(self.hotkey_help(window))
        self.assertEqual(help_map["1"], "Terminal")
        self.assertEqual(help_map["2"], "Components")


if __name__ == "__main__":
    unittest.main()
