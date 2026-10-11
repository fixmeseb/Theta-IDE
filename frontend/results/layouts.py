"""The Results panel's four layouts, and the widgets only some of them use.

Quiet toolbar: one row of controls, charts in folding sections, the inspector beside them.
Control rail: every control in a left rail with the runs, three columns of plain charts, inspector below.
Overview first: a card per series with its headline numbers, one large chart, then the rest.
Focus + filmstrip: one large chart at a time, picked from a strip of small previews of every metric.
"""

from __future__ import annotations

import html

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme_color
from ..widgets import label
from .data import group_metrics

LAYOUTS = {
    "overview": "Overview first",
    "quiet": "Quiet toolbar",
    "rail": "Control rail",
    "focus": "Focus + filmstrip",
}
DEFAULT_LAYOUT = "overview"
# The metric the Overview's large chart and the Focus layout open on, if a run logged it.
HEADLINE_METRICS = ("eval/reward", "eval/mean_reward", "reward", "train/reward", "episode_reward")


def headline_metric(metrics: list[str]) -> str | None:
    """The metric most worth a large chart: the reward if logged, else the first one."""
    preferred = next((m for m in HEADLINE_METRICS if m in metrics), None)
    if preferred is not None:
        return preferred
    return next((m for m in metrics if "reward" in m or "return" in m), metrics[0] if metrics else None)


def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()


class KpiCard(QFrame):
    """One series' headline numbers; clicking it makes the series primary."""

    clicked = pyqtSignal(str)

    def __init__(self, name: str, color: str, stats: list[tuple[str, str]], primary: bool, parent=None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("resultsKpi")
        self.setProperty("primary", primary)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "Primary series: hovering reads its steps. Click to go back to Nearest."
            if primary
            else "Click to make this the primary series (hovering reads its steps)"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        title = label(f'<span style="color:{color}">●</span>  {html.escape(name)}')
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setObjectName("resultsKpiName")
        layout.addWidget(title)
        row = QHBoxLayout()
        row.setSpacing(16)
        for caption, value in stats:
            cell = QVBoxLayout()
            cell.setSpacing(1)
            cell.addWidget(label(caption, "resultsKpiCaption"))
            cell.addWidget(label(value, "resultsKpiValue"))
            row.addLayout(cell)
        row.addStretch()
        layout.addLayout(row)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.name)
        super().mousePressEvent(event)


class KpiStrip(QWidget):
    """A row of KpiCards, one per plotted series (Overview layout)."""

    clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsKpiStrip")
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 4)
        self.row.setSpacing(10)
        self.cards: dict[str, KpiCard] = {}

    def set_cards(self, cards: list[tuple[str, str, list[tuple[str, str]]]], primary: str | None):
        """`cards`: (series name, color, [(caption, value)]) per series."""
        _clear(self.row)
        self.cards = {}
        for name, color, stats in cards:
            card = self.cards[name] = KpiCard(name, color, stats, name == primary)
            card.clicked.connect(self.clicked)
            self.row.addWidget(card, 1)
        self.setVisible(bool(cards))


class SeriesChips(QWidget):
    """One toggle per plotted series with its latest value; checking one makes it primary (Focus layout)."""

    clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)
        self.buttons: dict[str, QPushButton] = {}

    def set_chips(self, chips: list[tuple[str, str, str]], primary: str | None):
        """`chips`: (series name, color, latest value text) per series."""
        _clear(self.row)
        self.buttons = {}
        for name, color, value in chips:
            button = self.buttons[name] = QPushButton(f"●  {name}   {value}".rstrip())
            button.setObjectName("resultsChip")
            button.setCheckable(True)
            button.setChecked(name == primary)
            button.setStyleSheet(f"QPushButton#resultsChip {{ border-left: 3px solid {color}; }}")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip("Make this the primary series (click again for Nearest)")
            button.clicked.connect(lambda _=False, n=name: self.clicked.emit(n))
            self.row.addWidget(button)
        self.row.addStretch()


class PlotsList(QListWidget):
    """Every metric as a checkbox under its group, ticked when its plot is shown (Control rail layout)."""

    visibility_changed = pyqtSignal(str, bool)  # metric, visible

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsPlotsList")
        self.itemChanged.connect(self._changed)

    def set_metrics(self, metrics: list[str], hidden: set[str]):
        self.blockSignals(True)
        self.clear()
        for group, names in group_metrics(metrics).items():
            heading = QListWidgetItem(group.upper())
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addItem(heading)
            for metric in names:
                item = QListWidgetItem(metric.split("/", 1)[-1] if "/" in metric else metric)
                item.setToolTip(metric)
                item.setData(Qt.ItemDataRole.UserRole, metric)
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked if metric in hidden else Qt.CheckState.Checked)
                self.addItem(item)
        if not metrics:
            empty = QListWidgetItem("Tick runs to list their plots")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addItem(empty)
        self.blockSignals(False)

    def _changed(self, item):
        metric = item.data(Qt.ItemDataRole.UserRole)
        if metric:
            self.visibility_changed.emit(metric, item.checkState() == Qt.CheckState.Checked)


class Sparkline(QFrame):
    """A small preview of one metric: its name, the latest value and every series' curve."""

    clicked = pyqtSignal(str)

    def __init__(self, metric: str, parent=None):
        super().__init__(parent)
        self.metric = metric
        self.title = metric
        self.value = ""
        self.series: list[tuple[str, list[dict], str]] = []
        self.primary: str | None = None
        self.setObjectName("resultsSparkline")
        self.setFixedSize(172, 92)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, title: str, series, value: str, focused: bool, primary: str | None):
        self.title, self.series, self.value, self.primary = title, series, value, primary
        self.setProperty("focused", focused)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setToolTip(f"{self.metric}: click to show it large")
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.metric)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(theme_color("muted")))
        name = painter.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideMiddle, self.width() - 64)
        painter.drawText(QRectF(10, 6, self.width() - 64, 16), Qt.AlignmentFlag.AlignLeft, name)
        painter.setPen(QColor(theme_color("text")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(self.width() - 60, 6, 50, 16), Qt.AlignmentFlag.AlignRight, self.value)
        area = QRectF(10, 28, self.width() - 20, self.height() - 38)
        points = [p for _, metrics, _ in self.series for p in metrics if p.get(self.metric) is not None]
        if not points:
            painter.setPen(QColor(theme_color("comment")))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "no data")
            return
        xs = [p["step"] for p in points]
        ys = [p[self.metric] for p in points]
        xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
        for name, metrics, color in sorted(self.series, key=lambda s: s[0] == self.primary):
            metrics = [p for p in metrics if p.get(self.metric) is not None]
            if len(metrics) < 2:
                continue
            path = QPainterPath()
            for i, p in enumerate(metrics):
                x = area.left() + area.width() * (p["step"] - xlo) / max(1e-12, xhi - xlo)
                y = area.bottom() - area.height() * (p[self.metric] - ylo) / max(1e-12, yhi - ylo)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(theme_color(color)), 2 if name == self.primary else 1.3))
            painter.drawPath(path)


class Filmstrip(QScrollArea):
    """A horizontal strip with a Sparkline per visible metric; clicking one focuses it (Focus layout)."""

    chosen = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(112)
        body = QWidget()
        self.row = QHBoxLayout(body)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)
        self.row.addStretch()
        self.setWidget(body)
        self.tiles: dict[str, Sparkline] = {}

    def set_tiles(self, tiles, focused: str | None, primary: str | None):
        """`tiles`: (metric, title, series, latest value text) per visible metric, in order."""
        wanted = [metric for metric, *_ in tiles]
        for metric in [m for m in self.tiles if m not in wanted]:
            self.tiles.pop(metric).deleteLater()
        while self.row.count():
            self.row.takeAt(0)
        for metric, title, series, value in tiles:
            tile = self.tiles.get(metric)
            if tile is None:
                tile = self.tiles[metric] = Sparkline(metric)
                tile.clicked.connect(self.chosen)
            tile.set_data(title, series, value, metric == focused, primary)
            self.row.addWidget(tile)
        self.row.addStretch()
