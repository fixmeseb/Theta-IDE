"""Splash screen shown while the main window is built (settings, plugins, workspace, layout, runs).

The window is built on the GUI thread, so the event loop isn't running while the splash is up. The splash
therefore repaints itself synchronously on every progress step (QWidget.repaint) rather than animating.
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from .app_icon import ICON_PATH
from .theme import BUILTINS, DEFAULT, validate_theme

WIDTH, HEIGHT = 440, 250


def splash_theme(data_dir) -> dict:
    """The theme the window will open with, so the splash doesn't flash a different palette."""
    try:
        from .settings import SettingsManager

        name = SettingsManager(Path(data_dir)).theme
    except Exception:
        return DEFAULT
    if name in BUILTINS:
        return BUILTINS[name]
    try:  # a custom theme from the theme builder
        data = json.loads((Path(data_dir) / ".appearance.json").read_text(encoding="utf-8"))
        for item in data.get("themes", []):
            if item.get("name") == name:
                return validate_theme(item)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return DEFAULT


class SplashScreen(QSplashScreen):
    def __init__(self, theme: dict | None = None):
        screen = QApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen else 1.0
        canvas = QPixmap(round(WIDTH * ratio), round(HEIGHT * ratio))
        canvas.setDevicePixelRatio(ratio)
        canvas.fill(Qt.GlobalColor.transparent)
        super().__init__(canvas, Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.colors = (theme or DEFAULT)["colors"]
        self.logo = QIcon(str(ICON_PATH))
        self.message = "Starting…"
        self.progress = 0.0

    def set_progress(self, message: str, fraction: float):
        """Show a startup step. Progress never moves backwards."""
        self.message = message
        self.progress = max(self.progress, min(1.0, max(0.0, fraction)))
        if self.isVisible():
            self.repaint()

    def drawContents(self, painter: QPainter):
        c = {key: QColor(value) for key, value in self.colors.items()}
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRectF(0.5, 0.5, WIDTH - 1, HEIGHT - 1)
        path = QPainterPath()
        path.addRoundedRect(card, 12, 12)
        painter.fillPath(path, c["base"])
        painter.setPen(QPen(c["border"], 1))
        painter.drawPath(path)

        self.logo.paint(painter, QRect(36, 50, 76, 76))
        base_font = QApplication.font()
        title = QFont(base_font)
        title.setPixelSize(30)
        title.setWeight(QFont.Weight.Bold)
        painter.setFont(title)
        painter.setPen(c["text"])
        painter.drawText(QRectF(132, 58, WIDTH - 160, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "ThetaIDE")
        subtitle = QFont(base_font)
        subtitle.setPixelSize(13)
        painter.setFont(subtitle)
        painter.setPen(c["muted"])
        painter.drawText(QRectF(133, 96, WIDTH - 160, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "Research workspace for NeSyRL experiments")

        # status line and progress bar along the bottom
        status = QFont(base_font)
        status.setPixelSize(12)
        painter.setFont(status)
        painter.setPen(c["muted"])
        painter.drawText(QRectF(36, HEIGHT - 66, WIDTH - 72, 20),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.message)
        track = QRectF(36, HEIGHT - 40, WIDTH - 72, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(c["raised"])
        painter.drawRoundedRect(track, 2, 2)
        if self.progress > 0:
            painter.setBrush(c["accent"])
            painter.drawRoundedRect(QRectF(track.x(), track.y(), track.width() * self.progress, track.height()), 2, 2)

    def mousePressEvent(self, event):
        # QSplashScreen hides on click; keep it up until the window is ready.
        event.accept()
