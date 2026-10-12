"""Point inspector: every metric of every ticked run at the step under the mouse."""

from __future__ import annotations

from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..widgets import label
from .data import MetricTable, Source
from .metrics_catalog import AXIS_COLUMNS, describe, group_sort_key, metric_group


def format_value(metric: str, value: float) -> str:
    fmt = describe(metric).fmt
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


class PointInspector(QWidget):
    """Rows are metrics, columns are runs (or the other way round when `transposed`).

    Each run shows its logged point nearest the hovered x.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.caption = label("Hover over any chart to see every metric of every ticked run at that point.", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)
        self.values: dict[Source, dict[str, float]] = {}
        self.transposed = False  # runs as rows and metrics as columns (a wide strip under the charts)
        self._last = None

    def set_transposed(self, transposed: bool):
        self.transposed = transposed
        if self._last is not None:
            self.show_point(*self._last)

    def show_point(self, axis: str, x: float, tables: list[tuple[Source, MetricTable]], primary: str | None = None):
        """Fill the table with every value each run logged at its point nearest `x` on `axis`.

        `primary` names the series whose logged step `x` is (its runs come first in `tables`).
        """
        self._last = (axis, x, tables, primary)
        self.values, notes = {}, []
        for source, table in tables:
            nearest = table.nearest_x(axis, x)
            if nearest is None:
                continue
            self.values[source] = table.value_at(axis, nearest)
            if nearest != x:
                notes.append(f"{source.label} at {nearest:g}")
        metrics = sorted(
            {m for values in self.values.values() for m in values},
            # x-axis columns first, in axis preference order; then metrics by group and name
            key=lambda m: (
                (0, AXIS_COLUMNS.index(m), ()) if m in AXIS_COLUMNS else (1, 0, (group_sort_key(metric_group(m)), m))
            ),
        )
        sources = list(self.values)
        self.table.clear()
        rows, columns = (sources, metrics) if self.transposed else (metrics, sources)
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(columns))
        run_labels = [s.label for s in sources]
        self.table.setHorizontalHeaderLabels(metrics if self.transposed else run_labels)
        self.table.setVerticalHeaderLabels(run_labels if self.transposed else metrics)
        for i, metric in enumerate(metrics):
            header = self.table.horizontalHeaderItem(i) if self.transposed else self.table.verticalHeaderItem(i)
            header.setToolTip(describe(metric).label)
            for j, source in enumerate(sources):
                value = self.values[source].get(metric)
                item = QTableWidgetItem("—" if value is None else format_value(metric, value))
                self.table.setItem(*((j, i) if self.transposed else (i, j)), item)
        nearest_note = f"  ·  nearest logged point: {'; '.join(notes)}" if notes else ""
        primary_note = f" (a step of {primary})" if primary else ""
        self.caption.setText(
            f"{axis} = {x:g}{primary_note}  ·  {len(metrics)} values across {len(sources)} run(s){nearest_note}"
        )
