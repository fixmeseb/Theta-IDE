"""Config comparison: every setting that differs between the ticked runs' saved configs."""

from __future__ import annotations

import yaml
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..widgets import label
from .data import config_differences, decode_text


def run_config_path(run_key: str) -> str:
    """Where the pipeline saves a run's resolved Hydra config, relative to results/."""
    return f"logs/{run_key}/config.yaml"


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


class ConfigDiff(QWidget):
    """Rows are dotted settings (env.name, methods.ppo.lr, …); columns are runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.caption = label("", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self.rows: list[tuple[str, list]] = []
        self.show_configs({}, [])

    def show_configs(self, configs: dict[str, bytes], missing: list[str]):
        """`configs`: run key -> config.yaml bytes. `missing`: ticked runs without a saved config."""
        parsed, unreadable = {}, []
        for run_key, data in configs.items():
            try:
                value = yaml.safe_load(decode_text(data))
            except yaml.YAMLError:
                unreadable.append(run_key)
                continue
            parsed[run_key] = value if isinstance(value, dict) else {}
        self.rows = config_differences(parsed)
        names = list(parsed)
        self.table.clear()
        self.table.setColumnCount(len(names))
        self.table.setRowCount(len(self.rows))
        self.table.setHorizontalHeaderLabels(names)
        self.table.setVerticalHeaderLabels([key for key, _ in self.rows])
        for r, (_, values) in enumerate(self.rows):
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(_cell(value)))
        notes = [f"no saved config for {', '.join(missing)}"] if missing else []
        notes += [f"could not read {', '.join(unreadable)}"] if unreadable else []
        if not names:
            text = "Tick runs to compare their saved configs."
        elif len(names) == 1:
            text = f"All {len(self.rows)} settings of {names[0]}. Tick more runs to see only what differs."
        else:
            text = f"{len(self.rows)} settings differ across {len(names)} runs (identical ones are hidden)."
        self.caption.setText(text + (f"  ·  {'; '.join(notes)}" if notes else ""))
