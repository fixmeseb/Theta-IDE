"""Themed native title bar (frontend/titlebar.py)."""
import os
import sys
import unittest
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QMainWindow
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend import titlebar
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

COLORS = {"base": "#1d2021", "text": "#ebdbb2", "border": "#504945"}


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTitleBar(unittest.TestCase):
    def test_colorref_is_bgr(self):
        self.assertEqual(titlebar.colorref("#1d2021"), 0x21201D)
        self.assertEqual(titlebar.colorref("#ff0000"), 0x0000FF)

    def test_other_platforms_leave_the_title_bar_alone(self):
        with mock.patch.object(titlebar.sys, "platform", "linux"):
            self.assertFalse(titlebar.apply_title_bar(QMainWindow(), COLORS))

    @unittest.skipIf(sys.platform != "win32", "Windows only")
    def test_sets_caption_text_border_and_dark_mode(self):
        calls = []

        def fake_set(hwnd, attribute, data, size):
            calls.append((attribute, data._obj.value))
            return 0  # S_OK

        with mock.patch("ctypes.windll.dwmapi.DwmSetWindowAttribute", side_effect=fake_set):
            self.assertTrue(titlebar.apply_title_bar(QMainWindow(), COLORS))
        self.assertEqual(dict(calls), {
            titlebar.DWMWA_USE_IMMERSIVE_DARK_MODE: 1,
            titlebar.DWMWA_CAPTION_COLOR: titlebar.colorref("#1d2021"),
            titlebar.DWMWA_TEXT_COLOR: titlebar.colorref("#ebdbb2"),
            titlebar.DWMWA_BORDER_COLOR: titlebar.colorref("#504945"),
        })

    @unittest.skipIf(sys.platform != "win32", "Windows only")
    def test_light_theme_uses_light_caption_buttons(self):
        calls = {}
        with mock.patch("ctypes.windll.dwmapi.DwmSetWindowAttribute",
                        side_effect=lambda h, a, d, s: calls.__setitem__(a, d._obj.value) or 0):
            titlebar.apply_title_bar(QMainWindow(), {"base": "#ffffff", "text": "#202b3b", "border": "#c1cad8"})
        self.assertEqual(calls[titlebar.DWMWA_USE_IMMERSIVE_DARK_MODE], 0)

    @unittest.skipIf(sys.platform != "win32", "Windows only")
    def test_refused_caption_colour_reports_false(self):
        with mock.patch("ctypes.windll.dwmapi.DwmSetWindowAttribute", return_value=-2147024809):  # E_INVALIDARG
            self.assertFalse(titlebar.apply_title_bar(QMainWindow(), COLORS))


if __name__ == "__main__":
    unittest.main()
