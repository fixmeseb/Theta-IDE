"""Settings stays smooth: each settings page scrolls only as far as its content."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


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
        page = self.view.page_widgets["backend"]
        # Its height at the width it is shown at: wrapped text makes that taller than sizeHint()
        content = page.heightForWidth(page.width()) if page.hasHeightForWidth() else page.sizeHint().height()
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


if __name__ == "__main__":
    unittest.main()
