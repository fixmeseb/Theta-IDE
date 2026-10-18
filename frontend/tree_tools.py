"""What the Experiment tab's tree (config_tree.py) and the Results run selector (results/explorer.py) share.

One source for their look and their header behaviour: the themed 30 × 30 header buttons, the bold group
headings with a rule under them, the folder icon, and the search / collapse / expand actions.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPen
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QToolButton

from .sidetabs import svg_icon

TOOL_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "text",
    (QIcon.Mode.Active, QIcon.State.Off): "accent",
    (QIcon.Mode.Selected, QIcon.State.Off): "accent",
    (QIcon.Mode.Disabled, QIcon.State.Off): "disabled",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
    (QIcon.Mode.Selected, QIcon.State.On): "accent",
    (QIcon.Mode.Disabled, QIcon.State.On): "disabled",
}


def group_heading_font(base: QFont) -> QFont:
    """The bold, larger font of a group heading (cartpole, mimic, ...)."""
    font = QFont(base)
    font.setPointSize(max(base.pointSize() + 1, 13) if base.pointSize() > 0 else 13)
    font.setBold(True)
    return font


def folder_icon() -> QIcon:
    return svg_icon("folder", TOOL_ICON_COLORS)


def style_tool_button(button: QToolButton, icon_name: str, tooltip: str = "") -> QToolButton:
    """Make `button` a 30 × 30 header button with a themed SVG icon (refresh_tool_icons recolors it)."""
    button.setProperty("svg_icon", icon_name)
    button.setIcon(svg_icon(icon_name, TOOL_ICON_COLORS))
    button.setIconSize(QSize(16, 16))
    button.setFixedSize(30, 30)
    button.setStyleSheet("padding: 0;")
    if tooltip:
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
    return button


def square_tool_button(icon_name: str, tooltip: str = "") -> QToolButton:
    return style_tool_button(QToolButton(), icon_name, tooltip)


def refresh_tool_icons(widget) -> None:
    """Re-render every header button under `widget` in the current theme's colors."""
    for button in widget.findChildren(QToolButton):
        name = button.property("svg_icon")
        if name:
            button.setIcon(svg_icon(name, TOOL_ICON_COLORS))


class GroupItemDelegate(QStyledItemDelegate):
    """Draws group rows as headings: larger bold text, a subtle rule under them, more room above."""

    def is_group(self, index) -> bool:
        """Whether `index` is a group heading (the experiment tree marks them in its item data)."""
        data = index.data(Qt.ItemDataRole.UserRole)
        return isinstance(data, dict) and bool(data.get("is_group"))

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self.is_group(index):
            option.font = group_heading_font(option.font)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if self.is_group(index):
            painter.save()
            painter.setPen(QPen(QColor("#504945"), 1))
            y = option.rect.bottom() - 1
            painter.drawLine(option.rect.left() + 4, y, option.rect.right() - 4, y)
            painter.restore()

    def sizeHint(self, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        hint = super().sizeHint(opt, index)
        if self.is_group(index):
            return QSize(hint.width(), hint.height() + 10)
        return hint


class TreeHeaderActions:
    """Search, collapse all and expand all, for a widget with `tree`, `search` (a line edit) and
    `btn_search` (a checkable button). Call `watch_search()` once both exist: Esc then closes the search."""

    def watch_search(self):
        self.search.hide()
        self.search.installEventFilter(self)
        self.btn_search.setCheckable(True)
        self.btn_search.setChecked(False)
        self.btn_search.clicked.connect(self.toggle_search)

    def toggle_search(self, checked: bool | None = None):
        """Show or hide the filter; hiding it clears the filter."""
        if checked is None:
            checked = self.search.isHidden()
        self.btn_search.setChecked(checked)
        self.search.setVisible(checked)
        if checked:
            self.search.setFocus()
            self.search.selectAll()
        else:
            self.search.clear()

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == event.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.toggle_search(False)
            return True
        return super().eventFilter(obj, event)

    def collapse_all(self):
        """Collapse every node of the tree."""
        self.tree.collapseAll()

    def expand_all(self):
        """Expand every node of the tree."""
        self.tree.expandAll()
