"""The monitor's switcher: one chip per experiment training now, shown when there is more than one.

The monitor shows one run at a time; the chips pick which of the active runs it is. Each reads like the
runs list's rows (status mark, name, how far along), and the shown run's chip is checked.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from ..widgets import label
from .runs_sidebar import run_detail, run_label, status_mark


class RunSwitcher(QWidget):
    """`chosen(run_id)` when a chip is clicked."""

    chosen = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(6)
        self.row.addWidget(label("TRAINING NOW", "eyebrow"))
        self.row.addStretch(1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.chips: dict[str, QPushButton] = {}
        self.hide()

    def show_runs(self, runs: list[dict], shown_id: str | None):
        """Chips for `runs` (the active runs, in start order); the one for `shown_id` is checked."""
        ids = [run["id"] for run in runs]
        if ids != list(self.chips):
            for chip in self.chips.values():  # take them out now: deleteLater alone leaves them on screen
                self.group.removeButton(chip)
                self.row.removeWidget(chip)
                chip.hide()
                chip.setParent(None)
                chip.deleteLater()
            self.chips = {}
            for index, run in enumerate(runs):
                chip = QPushButton()
                chip.setObjectName("monitorViewButton")
                chip.setCheckable(True)
                chip.clicked.connect(lambda _=False, run_id=run["id"]: self.chosen.emit(run_id))
                self.group.addButton(chip)
                self.row.insertWidget(1 + index, chip)
                self.chips[run["id"]] = chip
        for run in runs:
            chip = self.chips[run["id"]]
            chip.setText(f"{status_mark(run.get('status', ''))}  {run_label(run)}  ·  {run_detail(run)}")
            chip.setToolTip(f"Show {run_label(run)} in the monitor")
            chip.setChecked(run["id"] == shown_id)
        self.setVisible(len(runs) > 1)
