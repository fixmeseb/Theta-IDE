"""Summary tab: headline numbers for every plotted series and metric, sortable and exportable."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets import SortableItem, label
from .data import SUMMARY_COLUMNS, to_float
from .metrics_catalog import describe

ALL_METRICS = ""
THRESHOLD_COLUMN = "Reaches threshold at"


def _text(value, fmt: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


class SummaryView(QWidget):
    """Final, best, best-at, range, mean and area under the curve of each series, per metric.

    Choosing one metric enables a threshold: the table then also says when each series first reached it.
    """

    changed = pyqtSignal()  # metric filter or threshold edited: the panel recomputes the rows

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(label("Metric", "muted"))
        self.metric_select = QComboBox()
        self.metric_select.addItem("All metrics", ALL_METRICS)
        self.metric_select.currentIndexChanged.connect(self._metric_changed)
        controls.addWidget(self.metric_select)
        controls.addWidget(label("Threshold", "muted"))
        self.threshold = QLineEdit()
        self.threshold.setPlaceholderText("e.g. 195")
        self.threshold.setToolTip("Pick one metric, then a value: the table shows when each series first reached it")
        self.threshold.setFixedWidth(110)
        self.threshold.setEnabled(False)
        self.threshold.textChanged.connect(lambda _: self.changed.emit())
        controls.addWidget(self.threshold)
        controls.addStretch()
        layout.addLayout(controls)
        self.caption = label("", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self.headers: list[str] = list(SUMMARY_COLUMNS)
        self.rows: list[list] = []

    def selected_metric(self) -> str:
        return self.metric_select.currentData() or ALL_METRICS

    def threshold_value(self) -> float | None:
        """The threshold, when one metric is chosen and a number is entered."""
        return to_float(self.threshold.text()) if self.selected_metric() else None

    def set_metrics(self, metrics: list[str]):
        current = self.selected_metric()
        self.metric_select.blockSignals(True)
        self.metric_select.clear()
        self.metric_select.addItem("All metrics", ALL_METRICS)
        for metric in metrics:
            self.metric_select.addItem(metric, metric)
        index = self.metric_select.findData(current)
        self.metric_select.setCurrentIndex(max(index, 0))
        self.metric_select.blockSignals(False)
        self.threshold.setEnabled(bool(self.selected_metric()))

    def _metric_changed(self, _index):
        self.threshold.setEnabled(bool(self.selected_metric()))
        self.changed.emit()

    def show_rows(self, rows: list[list], axis: str | None, note: str = ""):
        """Rows as summary_rows() returns them; a threshold column is present when they have one more value."""
        self.headers = list(SUMMARY_COLUMNS) + (
            [THRESHOLD_COLUMN] if rows and len(rows[0]) > len(SUMMARY_COLUMNS) else []
        )
        self.rows = rows
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(self.headers))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(self.headers)
        for r, row in enumerate(rows):
            fmt = describe(row[1]).fmt
            for c, value in enumerate(row):
                cell_fmt = "g" if self.headers[c] in ("Best at", THRESHOLD_COLUMN) else fmt
                key = value if not isinstance(value, str) else value.lower()
                self.table.setItem(r, c, SortableItem(_text(value, cell_fmt), key))
        self.table.setSortingEnabled(True)
        where = f"x = {axis}" if axis else "no x-axis"
        self.caption.setText(
            f"{len(rows)} series × metric rows · computed on raw (unsmoothed) data, {where}"
            + (f" · {note}" if note else "")
        )
