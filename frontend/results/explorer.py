"""Run explorer: a checkable tree of group → experiment → agent → version."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from .data import Manifest, Source

_KIND, _DATA = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE | Qt.ItemFlag.ItemIsAutoTristate


class RunExplorer(QWidget):
    """Tick experiments, agents or versions; every ticked leaf is a series source.

    An agent is a leaf (its newest version) until its manifest shows more than one version;
    then each version becomes a child the user can tick.
    """

    selection_changed = pyqtSignal(list)  # list[Source]
    manifest_wanted = pyqtSignal(str, str)  # group, experiment_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter runs…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.apply_filter)
        layout.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        layout.addWidget(self.tree, 1)
        self._emitting = True

    # ── Building ─────────────────────────────────────────────────────────────

    def set_runs(self, entries: list[dict]):
        """(Re)build from GET /api/runs items, keeping what was ticked and expanded."""
        checked = set(self.checked_sources())
        expanded = {item.data(0, _DATA) for item in self._items() if item.isExpanded()}
        self._emitting = False
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for entry in sorted(entries, key=lambda e: (e["group"], e["experiment_id"])):
            group, exp = entry["group"], entry["experiment_id"]
            agents = [a["name"] for a in entry.get("agents") or [] if a.get("name")]
            if not agents:
                continue
            if group not in groups:
                groups[group] = self._item(self.tree.invisibleRootItem(), group, "group", group, _PARENT)
            exp_item = self._item(groups[group], exp, "experiment", f"{group}/{exp}", _PARENT)
            exp_item.setToolTip(0, f"results/logs/{group}/{exp}")
            for agent in agents:
                self._item(exp_item, agent, "agent", Source(group, exp, agent), _CHECKABLE)
        for item in self._items():
            if item.data(0, _DATA) in expanded:
                item.setExpanded(True)
        for item in self._leaves():
            if item.data(0, _DATA) in checked:
                item.setCheckState(0, Qt.CheckState.Checked)
        self.apply_filter(self.filter.text())
        self._emitting = True
        self._emit()

    def add_manifest(self, manifest: Manifest):
        """Turn each multi-version agent of this run into a parent with one child per version."""
        self._emitting = False
        for item in self._items():
            source = item.data(0, _DATA)
            if item.data(0, _KIND) != "agent" or source.run_key != manifest.run_key or item.childCount():
                continue
            versions = manifest.agents.get(source.agent) or []
            if len(versions) < 2:
                continue
            was_checked = item.checkState(0) == Qt.CheckState.Checked
            item.setFlags(_PARENT)
            for version in versions:
                child = self._item(
                    item,
                    version,
                    "version",
                    Source(source.group, source.experiment_id, source.agent, version),
                    _CHECKABLE,
                )
                # A ticked agent meant "its newest version": keep exactly that ticked.
                if was_checked and version == versions[-1]:
                    child.setCheckState(0, Qt.CheckState.Checked)
        self._emitting = True
        self._emit()

    def _item(self, parent, text, kind, data, flags):
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, _KIND, kind)
        item.setData(0, _DATA, data)
        item.setFlags(flags)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        return item

    # ── Reading ──────────────────────────────────────────────────────────────

    def _items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _leaves(self):
        return [item for item in self._items() if item.data(0, _KIND) in ("agent", "version") and not item.childCount()]

    def checked_sources(self) -> list[Source]:
        """Ticked series sources, in tree order."""
        leaves = [item for item in self._leaves() if item.checkState(0) == Qt.CheckState.Checked]
        order = {id(item): i for i, item in enumerate(self._items_in_order())}
        return [item.data(0, _DATA) for item in sorted(leaves, key=lambda it: order[id(it)])]

    def _items_in_order(self):
        def walk(item):
            yield item
            for i in range(item.childCount()):
                yield from walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            yield from walk(self.tree.topLevelItem(i))

    # ── Events ───────────────────────────────────────────────────────────────

    def _item_changed(self, item, _column):
        if item.data(0, _KIND) == "agent" and item.checkState(0) != Qt.CheckState.Unchecked:
            source = item.data(0, _DATA)
            self.manifest_wanted.emit(source.group, source.experiment_id)
        self._emit()

    def _item_expanded(self, item):
        if item.data(0, _KIND) == "experiment":
            group, exp = item.data(0, _DATA).split("/", 1)
            self.manifest_wanted.emit(group, exp)

    def _emit(self):
        if self._emitting:
            self.selection_changed.emit(self.checked_sources())

    def apply_filter(self, text):
        """Show experiments whose group/name contains `text`, and the groups that hold them."""
        needle = text.strip().lower()
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            any_visible = False
            for e in range(group.childCount()):
                exp = group.child(e)
                visible = not needle or needle in exp.data(0, _DATA).lower()
                exp.setHidden(not visible)
                any_visible |= visible
            group.setHidden(not any_visible)
