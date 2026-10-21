"""Splash screen shown while the main window is built (settings, plugins, workspace, layout, runs).

The window is built on the GUI thread, so the event loop isn't running while the splash is up. The splash
therefore repaints itself synchronously on every progress step (QWidget.repaint) rather than animating.

Background: a public-domain classical landscape painting (images/splash_landscape.jpg), faded into the
theme's base colour at the bottom so the title and the phase circles read in light and dark themes.
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from .app_icon import ICON_PATH
from .theme import BUILTINS, DEFAULT, validate_theme

WIDTH, HEIGHT = 560, 340
BACKGROUND_PATH = Path(__file__).resolve().parent / "images" / "splash_landscape.jpg"
CREDIT = "Pierre-Henri de Valenciennes, Landscape of Ancient Greece, 1786"

# The startup phases, in order: one connected circle each. These are the messages Window.__init__ and
# main() pass to set_progress; a message not listed here is still shown but doesn't move the circles.
PHASES = (
    "Loading settings…",
    "Discovering plugins…",
    "Loading run history…",
    "Building the workspace…",
    "Starting plugins…",
    "Loading the default experiment…",
    "Restoring layout…",
    "Applying theme…",
    "Loading runs…",
    "Finishing up…",
    "Opening window…",
)
PHASE_RADIUS = 5.5


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
        self.background = QPixmap(str(BACKGROUND_PATH))  # null if missing: the card is then plain
        self.message = "Starting…"
        self.progress = 0.0
        self.phase = -1  # index into PHASES of the step under way; -1 before the first

    def set_progress(self, message: str, fraction: float):
        """Show a startup step: the phases before it are complete. Progress never moves backwards."""
        self.message = message
        self.progress = max(self.progress, min(1.0, max(0.0, fraction)))
        if message in PHASES:
            self.phase = max(self.phase, PHASES.index(message))
        if self.isVisible():
            self.repaint()

    def drawContents(self, painter: QPainter):
        c = {key: QColor(value) for key, value in self.colors.items()}
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        card = QRectF(0.5, 0.5, WIDTH - 1, HEIGHT - 1)
        path = QPainterPath()
        path.addRoundedRect(card, 12, 12)
        painter.fillPath(path, c["base"])
        if not self.background.isNull():
            painter.save()
            painter.setClipPath(path)
            painter.drawPixmap(QRectF(0, 0, WIDTH, HEIGHT), self.background, self._cover_source())
            # fade the painting into the theme's base colour behind the text
            fade = QLinearGradient(QPointF(0, HEIGHT * 0.30), QPointF(0, HEIGHT))
            for stop, alpha in ((0.0, 0), (0.45, 200), (0.7, 238), (1.0, 245)):
                color = QColor(c["base"])
                color.setAlpha(alpha)
                fade.setColorAt(stop, color)
            painter.fillRect(QRectF(0, 0, WIDTH, HEIGHT), fade)
            painter.restore()
        painter.setPen(QPen(c["border"], 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        self.logo.paint(painter, QRect(32, 178, 60, 60))
        base_font = QApplication.font()
        title = QFont(base_font)
        title.setPixelSize(30)
        title.setWeight(QFont.Weight.Bold)
        painter.setFont(title)
        painter.setPen(c["text"])
        painter.drawText(QRectF(108, 178, WIDTH - 140, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "ThetaIDE")
        subtitle = QFont(base_font)
        subtitle.setPixelSize(13)
        painter.setFont(subtitle)
        painter.setPen(c["muted"])
        painter.drawText(QRectF(109, 214, WIDTH - 140, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "A general-purpose workspace for machine-learning research")

        # status line and the phase circles along the bottom
        status = QFont(base_font)
        status.setPixelSize(12)
        painter.setFont(status)
        painter.setPen(c["muted"])
        painter.drawText(QRectF(32, HEIGHT - 62, WIDTH - 64, 20),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.message)
        self._draw_phases(painter, c)

        if not self.background.isNull():
            caption = QFont(base_font)
            caption.setPixelSize(10)
            caption.setItalic(True)
            painter.setFont(caption)
            painter.setPen(c["comment"])
            painter.drawText(QRectF(32, HEIGHT - 26, WIDTH - 64, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, CREDIT)

    def phase_centers(self) -> list[QPointF]:
        """Where each phase's circle sits: evenly spaced along the bottom, inside the text margins."""
        left, right, y = 32 + PHASE_RADIUS, WIDTH - 32 - PHASE_RADIUS, HEIGHT - 34
        step = (right - left) / (len(PHASES) - 1)
        return [QPointF(left + i * step, y) for i in range(len(PHASES))]

    def _draw_phases(self, painter: QPainter, c: dict[str, QColor]):
        """Connected circles: completed phases filled, the current one ringed with a dot, the rest hollow."""
        centers = self.phase_centers()
        line = 2.0
        for i in range(len(centers) - 1):  # connectors first, so the circles sit on top
            reached = i < self.phase  # the segment after a completed phase leads to one that has started
            painter.setPen(QPen(c["accent"] if reached else c["disabled"], line, cap=Qt.PenCapStyle.FlatCap))
            start, end = centers[i], centers[i + 1]
            painter.drawLine(QPointF(start.x() + PHASE_RADIUS, start.y()), QPointF(end.x() - PHASE_RADIUS, end.y()))
        for i, center in enumerate(centers):
            if i < self.phase:  # done
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(c["accent"])
                painter.drawEllipse(center, PHASE_RADIUS, PHASE_RADIUS)
            elif i == self.phase:  # under way
                painter.setPen(QPen(c["accent"], line))
                painter.setBrush(c["base"])
                painter.drawEllipse(center, PHASE_RADIUS - line / 2, PHASE_RADIUS - line / 2)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(c["accent"])
                painter.drawEllipse(center, PHASE_RADIUS / 2.5, PHASE_RADIUS / 2.5)
            else:  # still to come
                painter.setPen(QPen(c["disabled"], line))
                painter.setBrush(c["base"])
                painter.drawEllipse(center, PHASE_RADIUS - line / 2, PHASE_RADIUS - line / 2)

    def _cover_source(self) -> QRectF:
        """The centred part of the painting with the card's aspect ratio (like CSS background-size: cover)."""
        w, h = self.background.width(), self.background.height()
        if w / h > WIDTH / HEIGHT:
            crop = h * WIDTH / HEIGHT
            return QRectF((w - crop) / 2, 0, crop, h)
        crop = w * HEIGHT / WIDTH
        return QRectF(0, (h - crop) / 2, w, crop)

    def mousePressEvent(self, event):
        # QSplashScreen hides on click; keep it up until the window is ready.
        event.accept()
