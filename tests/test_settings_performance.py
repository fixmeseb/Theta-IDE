"""Settings view stays smooth: the ASCII sculpture behind an open page is paused and cached."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.about import AsciiTheta
    from frontend.app import Window
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSculptureHeldBehindSettingsPages(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.show()
        self.window.tabs.setCurrentWidget(self.window.settings_view)
        app.processEvents()
        self.view = self.window.settings_view
        self.sculpture = self.view.ascii_splash.ascii_sculpture

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_animates_on_the_landing_view(self):
        self.assertTrue(self.sculpture.timer.isActive())

    def test_paused_while_a_settings_page_is_open(self):
        self.view.tab_buttons["hotkeys"].click()
        self.assertFalse(self.sculpture.timer.isActive())

    def test_keeps_running_on_animation_settings_for_the_live_preview(self):
        self.view.show_tab("animation")
        self.assertTrue(self.sculpture.timer.isActive())

    def test_resumes_when_settings_close(self):
        self.view.tab_buttons["hotkeys"].click()
        self.view.hide_overlay()
        self.assertTrue(self.sculpture.timer.isActive())

    def test_hold_does_not_change_the_saved_animation_setting(self):
        self.view.tab_buttons["hotkeys"].click()
        self.assertTrue(self.window.settings_manager.get("appearance", "ascii_animation", default=True))

    def test_user_pause_survives_closing_settings(self):
        self.sculpture.set_paused(True)
        self.view.tab_buttons["hotkeys"].click()
        self.view.hide_overlay()
        self.assertFalse(self.sculpture.timer.isActive())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSculptureFrameCache(unittest.TestCase):
    def setUp(self):
        self.sculpture = AsciiTheta()
        self.sculpture.resize(600, 400)
        self.sculpture.show()
        app.processEvents()
        self.addCleanup(self.sculpture.deleteLater)

    def test_repaints_reuse_the_cached_frame(self):
        self.sculpture.repaint()  # builds the cache
        with mock.patch.object(self.sculpture, "frame", wraps=self.sculpture.frame) as frame:
            self.sculpture.repaint()
            self.sculpture.repaint()
        frame.assert_not_called()

    def test_update_rebuilds_the_frame(self):
        self.sculpture.repaint()
        with mock.patch.object(self.sculpture, "frame", wraps=self.sculpture.frame) as frame:
            self.sculpture.advance()
            self.sculpture.repaint()
        frame.assert_called_once()


if __name__ == "__main__":
    unittest.main()
