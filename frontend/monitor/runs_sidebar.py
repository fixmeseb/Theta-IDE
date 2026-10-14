"""Runs sidebar: every run at a glance; click one to show it, tick others to overlay them.

Live and queued runs come first, then the rest in the app's order (newest first). Each row has a status
mark, the run's name and seed, a small reward curve with the latest reward, and a checkbox. The selected
run is always drawn; ticked runs are overlaid on its Charts and System views in their dot's color.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QPushButton,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..model import LIVE_STATUSES
from ..theme import theme_color
from ..widgets import label

PALETTE = ("#b8bb26", "#83a598", "#fabd2f", "#d3869b", "#8ec07c", "#fe8019", "#fb4934", "#689d6a")
RUN_ID = Qt.ItemDataRole.UserRole
TREND = Qt.ItemDataRole.UserRole + 1  # list of reward values, oldest first
COLOR = Qt.ItemDataRole.UserRole + 2
TREND_POINTS = 60  # at most this many points per sparkline
STATUS_MARKS = {
    "completed": "✓",
    "failed": "✗",
    "interrupted": "✗",
    "stopped": "✗",
    "cancelled": "✗",
    "error": "✗",
    "queued": "◷",
}


def status_mark(status: str) -> str:
    return "●" if status in LIVE_STATUSES else STATUS_MARKS.get(status, "○")


def run_label(run) -> str:
    name = run["config"]["name"] if run.get("simulated") else (run.get("backend") or {}).get("experiment_id")
    return str(name or run["config"].get("name") or run["id"])


def reward_trend(run) -> list[float]:
    """The run's evaluation rewards, thinned to at most TREND_POINTS for drawing."""
    values = [m["reward"] for m in run.get("metrics") or [] if isinstance(m.get("reward"), (int, float))]
    if len(values) > TREND_POINTS:
        stride = len(values) / TREND_POINTS
        values = [values[int(i * stride)] for i in range(TREND_POINTS - 1)] + [values[-1]]
    return values


class TrendDelegate(QStyledItemDelegate):
    """Draws a run's reward sparkline and latest value in the second column."""

    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)
        values = index.data(TREND) or []
        if not values:
            return
        rect = QRectF(option.rect).adjusted(4, 5, -4, -5)
        text_width = 44
        area = QRectF(rect.left(), rect.top(), max(10.0, rect.width() - text_width - 6), rect.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(index.data(COLOR) or theme_color("primary"))
        if len(values) > 1:
            lo, hi = min(values), max(values)
            path = QPainterPath()
            for i, value in enumerate(values):
                x = area.left() + area.width() * i / (len(values) - 1)
                y = area.bottom() - area.height() * ((value - lo) / (hi - lo) if hi > lo else 0.5)
                if i == 0:
                    path.moveTo(QPointF(x, y))
                else:
                    path.lineTo(QPointF(x, y))
            painter.setPen(QPen(color, 1.4))
            painter.drawPath(path)
        painter.setPen(QColor(theme_color("muted")))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(rect.right() - text_width, rect.top(), text_width, rect.height()),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{values[-1]:.1f}",
        )
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(120, 26)


class RunsSidebar(QWidget):
    """Lists runs; `run_chosen` when one is clicked, `compare_changed` when the ticked set changes."""

    run_chosen = pyqtSignal(str)  # run id
    compare_changed = pyqtSignal(list)  # ids of the ticked runs, in the order they were ticked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("monitorRunsSidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(label("RUNS", "eyebrow"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter runs…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        self.tree.setItemDelegateForColumn(1, TrendDelegate(self.tree))
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 120)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemChanged.connect(self._item_changed)
        layout.addWidget(self.tree, 1)
        self.clear_button = QPushButton("Clear comparison")
        self.clear_button.setToolTip("Untick every run overlaid on the selected one")
        self.clear_button.clicked.connect(self.clear_comparison)
        self.clear_button.hide()
        layout.addWidget(self.clear_button)
        self.items: dict[str, QTreeWidgetItem] = {}
        self.compared: list[str] = []  # ticked run ids, in tick order
        self.selected_id: str | None = None
        self._filling = False

    # ── Filling ──────────────────────────────────────────────────────────────

    def set_runs(self, runs, selected=None):
        """Rebuild the list: live and queued runs first, then the rest. Ticks and selection are kept."""
        self._filling = True
        self.tree.clear()
        self.items = {}
        ids = {run["id"] for run in runs}
        self.compared = [run_id for run_id in self.compared if run_id in ids]
        live = [run for run in runs if run.get("status") in (*LIVE_STATUSES, "queued")]
        rest = [run for run in runs if run not in live]
        for title, group in (("LIVE AND QUEUED", live), ("RECENT", rest)):
            if not group:
                continue
            heading = QTreeWidgetItem(self.tree, [title])
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            heading.setForeground(0, QColor(theme_color("secondary")))
            for run in group:
                item = QTreeWidgetItem(self.tree)
                item.setData(0, RUN_ID, run["id"])
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(0, Qt.CheckState.Checked if run["id"] in self.compared else Qt.CheckState.Unchecked)
                self.items[run["id"]] = item
                self._fill(item, run)
        self.selected_id = selected["id"] if selected else self.selected_id
        self._paint_colors()
        self._apply_filter(self.filter.text())
        self._filling = False
        self.select(self.selected_id)
        self.clear_button.setVisible(bool(self.compared))

    def _fill(self, item, run):
        seed = run["config"].get("seed")
        seed_text = f"  (seed {seed})" if seed is not None else ""
        item.setText(0, f"{status_mark(run.get('status', ''))}  {run_label(run)}{seed_text}")
        item.setToolTip(
            0, f"{run_label(run)} · {run.get('status', '')}\nClick to show it; tick to overlay it on the shown run."
        )
        item.setData(1, TREND, reward_trend(run))

    def update_run(self, run):
        """Refresh one run's row (new metrics or status) without rebuilding the list."""
        item = self.items.get(run["id"])
        if item is not None:
            self._filling = True
            self._fill(item, run)
            self._filling = False

    def colors(self) -> dict[str, str]:
        """run id -> color: the selected run first, then the ticked runs in tick order."""
        order = [self.selected_id] if self.selected_id else []
        order += [run_id for run_id in self.compared if run_id != self.selected_id]
        return {run_id: PALETTE[i % len(PALETTE)] for i, run_id in enumerate(order)}

    def _paint_colors(self):
        colors = self.colors()
        filling, self._filling = self._filling, True  # recoloring fires itemChanged: not a tick
        for run_id, item in self.items.items():
            color = colors.get(run_id)
            item.setData(1, COLOR, color or theme_color("comment"))
            item.setForeground(0, QColor(color) if color else QColor(theme_color("text")))
        self._filling = filling

    def _apply_filter(self, text):
        needle = text.strip().lower()
        for item in self.items.values():
            item.setHidden(bool(needle) and needle not in item.text(0).lower())

    # ── Selection and comparison ─────────────────────────────────────────────

    def select(self, run_id):
        """Highlight `run_id` as the shown run, without signalling."""
        self.selected_id = run_id
        item = self.items.get(run_id)
        self._filling = True
        if item is not None:
            self.tree.setCurrentItem(item)
        self._filling = False
        self._paint_colors()

    def _current_changed(self, current, _previous):
        if self._filling or current is None:
            return
        run_id = current.data(0, RUN_ID)
        if run_id and run_id != self.selected_id:
            self.selected_id = run_id
            self._paint_colors()
            self.run_chosen.emit(run_id)

    def _item_changed(self, item, column):
        if self._filling or column != 0:
            return
        run_id = item.data(0, RUN_ID)
        if not run_id:
            return
        ticked = item.checkState(0) == Qt.CheckState.Checked
        if ticked and run_id not in self.compared:
            self.compared.append(run_id)
        elif not ticked and run_id in self.compared:
            self.compared.remove(run_id)
        self._paint_colors()
        self.clear_button.setVisible(bool(self.compared))
        self.compare_changed.emit(list(self.compared))

    def clear_comparison(self):
        self._filling = True
        for run_id in self.compared:
            item = self.items.get(run_id)
            if item is not None:
                item.setCheckState(0, Qt.CheckState.Unchecked)
        self._filling = False
        self.compared = []
        self._paint_colors()
        self.clear_button.hide()
        self.compare_changed.emit([])
