"""Plot grid: one chart card per metric, grouped by prefix, each one hideable."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QScrollArea, QToolButton, QVBoxLayout, QWidget

from ..charts import Chart
from ..widgets import label
from .data import group_metrics
from .metrics_catalog import describe

COLUMNS = 2


def chart_title(metric: str) -> str:
    """Catalog label plus the raw column name, so every chart says exactly which data it shows."""
    info = describe(metric)
    return metric if info.label == metric else f"{info.label}  ·  {metric}"


class ChartCard(QFrame):
    hide_requested = pyqtSignal(str)

    def __init__(self, metric: str, parent=None):
        super().__init__(parent)
        self.metric = metric
        self.setObjectName("resultsChartCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chart = Chart(metric, chart_title(metric), zero_based=False)
        self.chart.empty_text = "No data for the selected runs"
        self.chart.setMinimumHeight(220)
        layout.addWidget(self.chart)
        self.hide_button = QToolButton()
        self.hide_button.setText("Hide")
        self.hide_button.setToolTip(f"Hide {metric} (bring it back from the Plots menu)")
        self.hide_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_button.clicked.connect(lambda: self.hide_requested.emit(self.metric))
        self.chart.set_corner_widget(self.hide_button)  # top-right of the chart, like the Monitor's selector


class PlotGrid(QScrollArea):
    """Holds a ChartCard for every visible metric. Cards are reused across redraws."""

    hide_requested = pyqtSignal(str)
    point_hovered = pyqtSignal(float)  # x of the point under the mouse, on any chart

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 8, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(10)
        self.setWidget(self.body)
        self.cards: dict[str, ChartCard] = {}
        self.empty = label("", "muted")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_metrics(self, metrics: list[str], empty_text: str = ""):
        """Lay out cards for `metrics` (already filtered to visible ones), grouped by prefix."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        for metric in [m for m in self.cards if m not in metrics]:
            self.cards.pop(metric).deleteLater()
        if not metrics:
            self.empty.setText(empty_text)
            self.grid.addWidget(self.empty, 0, 0, 1, COLUMNS)
            return
        row = 0
        for group, names in group_metrics(metrics).items():
            heading = label(group.upper(), "eyebrow")
            self.grid.addWidget(heading, row, 0, 1, COLUMNS)
            row += 1
            for i, metric in enumerate(names):
                card = self.cards.get(metric)
                if card is None:
                    card = self.cards[metric] = ChartCard(metric)
                    card.hide_requested.connect(self.hide_requested)
                    card.chart.point_hovered.connect(self.point_hovered)
                self.grid.addWidget(card, row + i // COLUMNS, i % COLUMNS)
            row += (len(names) + COLUMNS - 1) // COLUMNS
        self.grid.setRowStretch(row, 1)

    def charts(self) -> dict[str, Chart]:
        return {metric: card.chart for metric, card in self.cards.items()}
