"""Plot grid: one chart card per metric, grouped by prefix, each one hideable."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QScrollArea, QToolButton, QVBoxLayout, QWidget

from ..charts import Chart
from ..widgets import label
from .data import group_metrics
from .metrics_catalog import describe

COLUMNS = 2
HERO_HEIGHT = 340  # minimum height of a hero (Overview) or focused (Focus) chart


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
        corner = QWidget()
        corner.setObjectName("resultsCardCorner")
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(6)
        self.value = label("", "resultsCardValue")
        self.value.setToolTip(
            "Latest value of the primary series (or the first series), as drawn: smoothed when Smoothing is on"
        )
        self.value.hide()
        corner_layout.addWidget(self.value)
        self.hide_button = QToolButton()
        self.hide_button.setText("Hide")
        self.hide_button.setToolTip(f"Hide {metric} (bring it back from the Plots menu)")
        self.hide_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_button.clicked.connect(lambda: self.hide_requested.emit(self.metric))
        corner_layout.addWidget(self.hide_button)
        self.chart.set_corner_widget(corner)  # top-right of the chart, like the Monitor's selector

    def set_value(self, text: str, color: str | None = None):
        """Show `text` (e.g. the latest value) beside the Hide button, or nothing when empty."""
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};" if color else "")
        self.value.setVisible(bool(text))
        self.chart.position_corner()


class PlotGrid(QScrollArea):
    """Holds a ChartCard for every visible metric. Cards are reused across redraws.

    How they are laid out is set by `configure`: the number of columns, collapsible group headings,
    a `hero` metric drawn full width above the rest, or a single `focus` metric filling the grid.
    """

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
        self.columns = COLUMNS
        self.collapsible = False
        self.collapsed: set[str] = set()  # groups whose cards are folded away under their heading
        self.hero: str | None = None
        self.focus: str | None = None
        self.header: QWidget | None = None
        self.flat = False
        self._metrics: list[str] = []
        self._empty_text = ""

    def configure(self, columns=COLUMNS, collapsible=False, hero=None, focus=None, header=None, flat=False):
        """Change how cards are laid out; takes effect on the next `set_metrics`."""
        self.columns, self.collapsible, self.hero, self.focus = columns, collapsible, hero, focus
        self.header, self.flat = header, flat
        for card in self.cards.values():
            if card.property("flat") != flat:
                card.setProperty("flat", flat)
                card.style().unpolish(card)  # Qt only re-reads property selectors when re-polished
                card.style().polish(card)

    def relayout(self):
        self.set_metrics(self._metrics, self._empty_text)

    def toggle_group(self, group: str):
        (self.collapsed.discard if group in self.collapsed else self.collapsed.add)(group)
        self.relayout()

    def card(self, metric: str) -> ChartCard:
        card = self.cards.get(metric)
        if card is None:
            card = self.cards[metric] = ChartCard(metric)
            card.setProperty("flat", self.flat)
            card.hide_requested.connect(self.hide_requested)
            card.chart.point_hovered.connect(self.point_hovered)
        return card

    def set_metrics(self, metrics: list[str], empty_text: str = ""):
        """Lay out cards for `metrics` (already filtered to visible ones), grouped by prefix."""
        self._metrics, self._empty_text = list(metrics), empty_text
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 0)
        for metric in [m for m in self.cards if m not in metrics]:
            self.cards.pop(metric).deleteLater()
        columns = self.columns
        for c in range(max(columns, self.grid.columnCount())):
            self.grid.setColumnStretch(c, 1 if c < columns else 0)
        row = 0
        if self.header is not None:
            self.grid.addWidget(self.header, row, 0, 1, columns)
            row += 1
        if not metrics:
            self.empty.setText(empty_text)
            self.grid.addWidget(self.empty, row, 0, 1, columns)
            self.grid.setRowStretch(row, 1)
            return
        for metric in metrics:
            self.card(metric)  # every visible metric has a card, even when only one is on screen
        if self.focus is not None:
            focused = self.focus if self.focus in metrics else metrics[0]
            card = self.cards[focused]
            card.chart.setMinimumHeight(HERO_HEIGHT)
            self.grid.addWidget(card, row, 0, 1, columns)
            self.grid.setRowStretch(row, 1)
            return
        rest = list(metrics)
        if self.hero in metrics:
            card = self.cards[self.hero]
            card.chart.setMinimumHeight(HERO_HEIGHT)
            self.grid.addWidget(card, row, 0, 1, columns)
            row += 1
            rest.remove(self.hero)
        for group, names in group_metrics(rest).items():
            folded = self.collapsible and group in self.collapsed
            if self.collapsible:
                heading = QToolButton()
                heading.setObjectName("resultsSection")
                heading.setText(f"{'▸' if folded else '▾'}  {group.upper()}   {len(names)}")
                heading.setToolTip(f"{'Show' if folded else 'Fold away'} the {group} plots")
                heading.setCursor(Qt.CursorShape.PointingHandCursor)
                heading.clicked.connect(lambda _=False, g=group: self.toggle_group(g))
            else:
                heading = label(group.upper(), "eyebrow")
            self.grid.addWidget(heading, row, 0, 1, columns)
            row += 1
            if folded:
                continue
            for i, metric in enumerate(names):
                card = self.cards[metric]
                card.chart.setMinimumHeight(180 if columns > 2 else 220)
                self.grid.addWidget(card, row + i // columns, i % columns)
            row += (len(names) + columns - 1) // columns
        self.grid.setRowStretch(row, 1)

    def charts(self) -> dict[str, Chart]:
        return {metric: card.chart for metric, card in self.cards.items()}
