"""Switching themes restyles the whole window immediately, not only icons and Settings.

Zoom puts the theme stylesheet on the main window itself, and a widget's own stylesheet
overrides the application's, so a theme change used to leave most of the window in the old
colors until restart.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.theme import stylesheet
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestThemeSwitchRestylesWindow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.resize(1200, 800)
        self.window.show()
        app.processEvents()
        self.manager = self.window.theme_manager

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def switch_to_another_theme(self):
        current = self.manager.active["name"]
        name = next(n for n in self.manager.themes() if n != current
                    and self.manager.themes()[n]["colors"]["panel"] != self.manager.active["colors"]["panel"])
        self.window.select_theme(self.manager.themes()[name])
        app.processEvents()
        return self.manager.themes()[name]

    def test_window_stylesheet_follows_the_new_theme(self):
        theme = self.switch_to_another_theme()
        self.assertEqual(self.window.styleSheet(), stylesheet(theme, zoom=self.manager.zoom))

    def test_panel_background_is_repainted_in_the_new_color(self):
        theme = self.switch_to_another_theme()
        panel = self.window.config_panel
        self.window.tabs.setCurrentWidget(panel)
        app.processEvents()
        corner = panel.grab().toImage().pixelColor(2, panel.height() - 3)
        self.assertEqual(corner.name(), QColor(theme["colors"]["panel"]).name())

    def test_zoom_after_a_theme_change_keeps_the_new_theme(self):
        theme = self.switch_to_another_theme()
        self.window.set_zoom(1.2, persist=False)
        app.processEvents()
        self.assertEqual(self.window.styleSheet(), stylesheet(theme, zoom=1.2))


if __name__ == "__main__":
    unittest.main()
