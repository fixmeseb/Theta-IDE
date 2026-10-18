"""Run explorer: a checkable tree of group → experiment → agent → version.

It looks like the Experiment tab's tree (frontend/config_tree.py): the same square header buttons
(reload, collapse all, expand all, search), bold group headings with a rule under them, and folders for
experiments. Groups are headings only; experiments, agents and versions are ticked.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..config_tree import TOOL_ICON_COLORS, GroupItemDelegate, square_tool_button
from ..sidetabs import svg_icon
from .data import Manifest, Source

_KIND, _DATA = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE | Qt.ItemFlag.ItemIsAutoTristate
_HEADING = Qt.ItemFlag.ItemIsEnabled  # a group: a heading, not ticked itself


class _RunGroupDelegate(GroupItemDelegate):
    """The experiment tree's group heading look, for this tree's group rows."""

    def is_group(self, index) -> bool:
        return index.data(_KIND) == "group"


class RunExplorer(QWidget):
    """Tick experiments, agents or versions; every ticked leaf is a series source.

    An agent is a leaf (its newest version) until its manifest shows more than one version;
    then each version becomes a child the user can tick.
    """

    selection_changed = pyqtSignal(list)  # list[Source]
    manifest_wanted = pyqtSignal(str, str)  # group, experiment_id
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.btn_refresh = square_tool_button("reload", "Re-read results/ from disk")
        self.btn_refresh.clicked.connect(self.refresh_requested)
        self.btn_collapse = square_tool_button("collapse_all", "Collapse all groups")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_expand = square_tool_button("expand_all", "Expand all groups and experiments")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_search = square_tool_button("search", "Filter runs")
        self.btn_search.setCheckable(True)
        self.btn_search.toggled.connect(self.toggle_search)
        for button in (self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search):
            header.addWidget(button)
        header.addStretch(1)
        layout.addLayout(header)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter runs…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.apply_filter)
        self.filter.hide()  # the search button shows it, as in the Experiment tab
        layout.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setItemDelegate(_RunGroupDelegate(self.tree))
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        self.tree.itemCollapsed.connect(self._item_collapsed)
        layout.addWidget(self.tree, 1)
        self._collapsed_groups: set[str] = set()  # groups start expanded, as in the Experiment tab
        self._emitting = True

    # ── Header buttons ───────────────────────────────────────────────────────

    def toggle_search(self, on: bool):
        self.filter.setVisible(on)
        if on:
            self.filter.setFocus()
        else:
            self.filter.clear()

    def collapse_all(self):
        self.tree.collapseAll()

    def expand_all(self):
        self.tree.expandAll()

    def refresh_icons(self):
        """Re-render the header and folder icons in the current theme's colors."""
        for button in self.findChildren(QToolButton):
            name = button.property("svg_icon")
            if name:
                button.setIcon(svg_icon(name, TOOL_ICON_COLORS))
        for item in self._items():
            if item.data(0, _KIND) == "experiment":
                item.setIcon(0, svg_icon("folder", TOOL_ICON_COLORS))

    # ── Building ─────────────────────────────────────────────────────────────

    def set_runs(self, entries: list[dict]):
        """(Re)build from catalog.list_runs items, keeping what was ticked and expanded."""
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
                groups[group] = self._item(self.tree.invisibleRootItem(), group, "group", group, _HEADING)
            exp_item = self._item(groups[group], exp, "experiment", f"{group}/{exp}", _PARENT)
            exp_item.setToolTip(0, f"results/logs/{group}/{exp}")
            exp_item.setIcon(0, svg_icon("folder", TOOL_ICON_COLORS))
            for agent in agents:
                self._item(exp_item, agent, "agent", Source(group, exp, agent), _CHECKABLE)
        for item in self._items():
            group = item.data(0, _KIND) == "group"
            if (group and item.data(0, _DATA) not in self._collapsed_groups) or item.data(0, _DATA) in expanded:
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
        if flags & Qt.ItemFlag.ItemIsUserCheckable:  # a group heading has no checkbox
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

    def _item_collapsed(self, item):
        if item.data(0, _KIND) == "group" and self._emitting:
            self._collapsed_groups.add(item.data(0, _DATA))

    def _item_expanded(self, item):
        if item.data(0, _KIND) == "group":
            self._collapsed_groups.discard(item.data(0, _DATA))
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
