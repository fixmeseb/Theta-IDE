"""Colour the native Windows title bar with the active theme so it blends into the menu bar below it.

Windows 11 (build 22000+) lets an app set the caption, caption-text and border colours. Windows 10
(build 18985+) only offers a dark caption, which is used for dark themes. On other platforms, or if
Windows refuses, the system title bar is left as it is.
"""
import sys

from PyQt6.QtGui import QColor

# DwmSetWindowAttribute attributes (dwmapi.h)
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36


def colorref(hex_color):
    """A "#rrggbb" colour as a Windows COLORREF (0x00bbggrr)."""
    color = QColor(hex_color)
    return color.red() | color.green() << 8 | color.blue() << 16


def apply_title_bar(window, colors):
    """Colour `window`'s title bar from a theme's role -> "#rrggbb" map.

    The caption takes the menu bar's colour ("base"), its text the theme's "text" colour and the
    window outline the "border" colour. Returns True if Windows accepted the caption colour.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    try:
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
    except (AttributeError, OSError):
        return False
    hwnd = wintypes.HWND(int(window.winId()))

    def set_value(attribute, value):
        data = wintypes.DWORD(value)
        return set_attribute(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data)) == 0  # S_OK

    # Light or dark caption buttons and fallback caption (the only option on Windows 10).
    set_value(DWMWA_USE_IMMERSIVE_DARK_MODE, int(QColor(colors["base"]).lightness() < 128))
    accepted = set_value(DWMWA_CAPTION_COLOR, colorref(colors["base"]))
    set_value(DWMWA_TEXT_COLOR, colorref(colors["text"]))
    set_value(DWMWA_BORDER_COLOR, colorref(colors["border"]))
    return accepted
