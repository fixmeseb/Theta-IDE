"""Settings and About stay smooth: the About page's ASCII sculpture animates only while it is
shown and repaints from a cached frame, and each settings page scrolls only as far as its content."""
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
class TestAboutLogoAnimatesOnlyWhenShown(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.show()
        self.window.show_about()
        app.processEvents()
        self.sculpture = self.window.about_view.logo

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_animates_on_the_about_page(self):
        self.assertTrue(self.sculpture.timer.isActive())

    def test_stops_when_another_page_is_shown(self):
        self.window.show_settings()
        app.processEvents()
        self.assertFalse(self.sculpture.timer.isActive())
        self.window.show_about()
        app.processEvents()
        self.assertTrue(self.sculpture.timer.isActive())

    def test_user_pause_survives_leaving_and_returning(self):
        self.sculpture.set_paused(True)
        self.window.show_settings()
        self.window.show_about()
        app.processEvents()
        self.assertFalse(self.sculpture.timer.isActive())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSettingsPageScrolling(unittest.TestCase):
    """Each page scrolls only as far as its own content, and opens at its top."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.resize(1500, 900)
        self.window.show()
        self.window.tabs.setCurrentWidget(self.window.settings_view)
        app.processEvents()
        self.view = self.window.settings_view
        self.area = self.view.settings_window.scroll_area
        self.bar = self.area.verticalScrollBar()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def open(self, tab_id):
        self.view.show_tab(tab_id)
        for _ in range(3):
            app.processEvents()

    def test_short_page_is_not_padded_to_the_tallest_page(self):
        self.open("hotkeys")      # the tallest page
        self.open("backend")      # a short one
        content = self.view.page_widgets["backend"].sizeHint().height()
        self.assertLessEqual(self.bar.maximum(), max(0, content - self.area.viewport().height()))

    def test_scroll_range_matches_the_current_page(self):
        self.open("backend")
        short_range = self.bar.maximum()
        self.open("hotkeys")
        self.assertGreater(self.bar.maximum(), short_range)
        self.open("backend")
        self.assertEqual(self.bar.maximum(), short_range)

    def test_each_page_opens_at_its_top(self):
        self.open("hotkeys")
        self.bar.setValue(self.bar.maximum())
        self.assertGreater(self.bar.value(), 0)
        self.open("appearance")
        self.assertEqual(self.bar.value(), 0)


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
