"""Tree widget mirroring in/config/ directory for Theta-IDE."""
import json
import shutil
from pathlib import Path

import yaml
from PyQt6.QtCore import QByteArray, QMimeData, QPoint, QRect, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from .config_model import (
    ConfigTree,
    default_method_model,
    deletion_blocked_reason,
    experiment_yaml,
    group_base_yaml,
    rename_experiment_text,
    unique_method_name,
)
from .sidetabs import svg_icon
from .tree_tools import (  # noqa: F401  (TOOL_ICON_COLORS and GroupItemDelegate are re-exported)
    TOOL_ICON_COLORS,
    GroupItemDelegate,
    TreeHeaderActions,
    folder_icon,
    group_heading_font,
    refresh_tool_icons,
    style_tool_button,
)
from .theme import theme_color
from .widgets import label

TREE_GEAR_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "secondary",
    (QIcon.Mode.Active, QIcon.State.Off): "accent",
    (QIcon.Mode.Selected, QIcon.State.Off): "text",
    (QIcon.Mode.Disabled, QIcon.State.Off): "disabled",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
    (QIcon.Mode.Selected, QIcon.State.On): "accent",
}

CATEGORY_ICONS = {
    "agent": "backend",
    "model": "components",
    "env": "monitor",
    "data": "storage",
    "eval": "results",
    "plot": "plots",
    "plots": "plots",
    "datamodule": "storage",
    "evaluator": "results",
    "plotter": "plots",
    "paradigms": "hotkeys",
    "site": "storage",
    "hydra": "gear",
    "sweeper": "gear",
    "hydra/sweeper": "gear",
    "methods": "components",
    "workflow": "workflow",
}


class InlineGroupEditor(QLineEdit):
    escape_pressed = pyqtSignal()
    lost_focus_empty = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if not self.text().strip():
            self.lost_focus_empty.emit()


class NewExperimentDialog(QDialog):
    """Group, name, and - only for a group that does not exist yet - the paradigm
    and environment its _base.yaml will bind.

    One form rather than a chain of prompts, so the paradigm and environment
    rows can appear and re-filter as the group name is typed.
    """

    def __init__(self, tree, groups, group=None, parent=None):
        super().__init__(parent)
        self.tree = tree
        self.setWindowTitle("New Experiment")
        self.setMinimumWidth(420)

        form = QFormLayout()
        form.setVerticalSpacing(10)

        self.group = QComboBox()
        self.group.setEditable(True)
        self.group.addItems(groups)
        if group:
            self.group.setCurrentText(group)
            self.group.setEnabled(False)
        form.addRow("Group", self.group)

        self.name = QLineEdit("new_experiment")
        form.addRow("Experiment name", self.name)

        self.paradigm = QComboBox()
        self.paradigm.addItems(sorted(tree.paradigms) if tree else [])
        form.addRow("Paradigm", self.paradigm)

        self.env = QComboBox()
        form.addRow("Environment", self.env)

        self.hint = label("", "muted")
        self.hint.setWordWrap(True)
        form.addRow("", self.hint)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.group.currentTextChanged.connect(self._group_changed)
        self.paradigm.currentTextChanged.connect(self._paradigm_changed)
        self._group_changed(self.group.currentText())

    def _group_changed(self, name):
        """A known group already fixes paradigm and env; a new one must choose."""
        known = bool(self.tree) and self.tree.group(name.strip()).has_base
        self.paradigm.setEnabled(not known)
        self.env.setEnabled(not known)
        if known:
            group = self.tree.group(name.strip())
            self.hint.setText(f"Inherits {group.paradigm or 'defaults'} / {group.env or 'env'} from {name}/_base.yaml")
            if group.paradigm:
                self.paradigm.setCurrentText(group.paradigm)
            self._paradigm_changed(self.paradigm.currentText())
            if group.env:
                self.env.setCurrentText(group.env)
        else:
            self.hint.setText(f"'{name}' is new — a _base.yaml will be written binding these.")
            self._paradigm_changed(self.paradigm.currentText())

    def _paradigm_changed(self, paradigm):
        if not self.tree or paradigm not in self.tree.paradigms:
            return
        current = self.env.currentText()
        self.env.clear()
        self.env.addItems([e.name for e in self.tree.environments_for(paradigm)])
        if current:
            self.env.setCurrentText(current)

    def values(self):
        return (self.group.currentText().strip(), self.name.text().strip(),
                self.paradigm.currentText(), self.env.currentText())


class ComponentDeleteDialog(QDialog):
    """Dialog displayed when a user deletes a folder or file belonging to a Hub component."""

    def __init__(self, rel_path: str, comp_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Uninstall Component?")
        self.setFixedWidth(460)
        self.choice = "cancel"

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Managed Hub Component: {comp_name}")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ebdbb2;")
        layout.addWidget(title)

        desc = QLabel(
            f"<b>'{rel_path}'</b> is managed by Theta Hub (<b>{comp_name}</b>).<br><br>"
            f"Would you like to completely uninstall this component from your workspace, "
            f"or only delete this folder?"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a89984; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        btn_delete_only = QPushButton("Delete Folder Only")
        btn_delete_only.setToolTip("Deletes only this folder from disk without uninstalling the component source code")
        btn_delete_only.clicked.connect(self._on_delete_only)
        btn_row.addWidget(btn_delete_only)

        btn_uninstall = QPushButton("Uninstall Completely")
        btn_uninstall.setStyleSheet("background-color: #cc241d; color: #ebdbb2; font-weight: bold;")
        btn_uninstall.setToolTip("Completely uninstalls the component and removes both its source code and configurations")
        btn_uninstall.clicked.connect(self._on_uninstall)
        btn_row.addWidget(btn_uninstall)

        layout.addLayout(btn_row)

    def _on_cancel(self):
        self.choice = "cancel"
        self.reject()

    def _on_delete_only(self):
        self.choice = "delete_only"
        self.accept()

    def _on_uninstall(self):
        self.choice = "uninstall"
        self.accept()


def find_config_root():
    """Locate in/config directory from cwd or parent directories."""
    candidates = [
        Path.cwd() / "in" / "config",
        Path.cwd() / "in" / "configs",
        Path(__file__).resolve().parent.parent / "in" / "config",
        Path(__file__).resolve().parent.parent / "in" / "configs",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    return (Path(__file__).resolve().parent.parent / "in" / "config").resolve()


TREE_BUTTON_MIME = "application/x-theta-tree-button"


class DraggableToolButton(QToolButton):
    """Square header tool button with drag-and-drop reordering support."""

    def __init__(self, button_id: str, parent=None):
        super().__init__(parent)
        self.button_id = button_id
        self._drag_start_pos = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            dist = (event.pos() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._start_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        p = self.parentWidget()
        if p is not None and hasattr(p, "contextMenuEvent"):
            p.contextMenuEvent(event)
        else:
            super().contextMenuEvent(event)

    def _start_drag(self):
        self.setDown(False)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(TREE_BUTTON_MIME, QByteArray(self.button_id.encode("utf-8")))
        drag.setMimeData(mime)

        pixmap = self.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.75)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)


class TreeHeaderLayout(QLayout):
    """Layout managing two rows of toolbar buttons with left and right groups per row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self.groups: dict[str, list[DraggableToolButton]] = {
            "row1_left": [],
            "row1_right": [],
            "row2_left": [],
            "row2_right": [],
        }

    def addItem(self, item):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def sizeHint(self) -> QSize:
        return QSize(200, 64)

    def minimumSize(self) -> QSize:
        return QSize(120, 64)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        btn_w, btn_h, spacing = 30, 30, 4
        y1 = rect.top()
        y2 = rect.top() + btn_h + spacing

        # row1_left: left-aligned
        x = rect.left()
        for w in self.groups.get("row1_left", []):
            w.setGeometry(QRect(x, y1, btn_w, btn_h))
            x += btn_w + spacing

        # row1_right: right-aligned, rendered left-to-right
        r1_list = self.groups.get("row1_right", [])
        if r1_list:
            total_r1_w = len(r1_list) * btn_w + (len(r1_list) - 1) * spacing
            rx = rect.right() - total_r1_w + 1
            for w in r1_list:
                w.setGeometry(QRect(rx, y1, btn_w, btn_h))
                rx += btn_w + spacing

        # row2_left: left-aligned
        x = rect.left()
        for w in self.groups.get("row2_left", []):
            w.setGeometry(QRect(x, y2, btn_w, btn_h))
            x += btn_w + spacing

        # row2_right: right-aligned, rendered left-to-right
        r2_list = self.groups.get("row2_right", [])
        if r2_list:
            total_r2_w = len(r2_list) * btn_w + (len(r2_list) - 1) * spacing
            rx = rect.right() - total_r2_w + 1
            for w in r2_list:
                w.setGeometry(QRect(rx, y2, btn_w, btn_h))
                rx += btn_w + spacing


class TreeHeaderWidget(QWidget):
    """Header toolbar widget with two rows and 4 draggable/reorderable button groups."""

    layout_changed = pyqtSignal()

    def __init__(self, tree_widget: "ConfigTreeWidget", parent=None):
        super().__init__(parent)
        self.tree_widget = tree_widget
        self.setAcceptDrops(True)
        self.header_layout = TreeHeaderLayout(self)
        self._drop_info = None  # (group_key, target_index, (kind, rect))

    def set_groups(self, groups: dict[str, list[DraggableToolButton]]):
        self.header_layout.groups = {
            "row1_left": list(groups.get("row1_left", [])),
            "row1_right": list(groups.get("row1_right", [])),
            "row2_left": list(groups.get("row2_left", [])),
            "row2_right": list(groups.get("row2_right", [])),
        }
        self.header_layout._items.clear()
        for g in ("row1_left", "row1_right", "row2_left", "row2_right"):
            for b in self.header_layout.groups[g]:
                self.header_layout.addWidget(b)
        self.header_layout.setGeometry(self.contentsRect())
        self.updateGeometry()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.header_layout.setGeometry(self.contentsRect())

    def sizeHint(self) -> QSize:
        return QSize(200, 64)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, 64)

    def _calc_drop_info(self, pos: QPoint):
        w = self.width()
        mid_x = w / 2.0
        row_key = "row1" if pos.y() < 32 else "row2"
        side_key = "left" if pos.x() <= mid_x else "right"
        group_key = f"{row_key}_{side_key}"
        buttons = self.header_layout.groups.get(group_key, [])

        y = 0 if row_key == "row1" else 34
        btn_w, btn_h = 30, 30

        if not buttons:
            if side_key == "left":
                rect = QRect(0, y, btn_w, btn_h)
            else:
                rect = QRect(w - btn_w, y, btn_w, btn_h)
            return (group_key, 0, ("box", rect))

        for idx, btn in enumerate(buttons):
            geom = btn.geometry()
            if pos.x() < geom.center().x():
                lx = max(0, geom.left() - 2)
                return (group_key, idx, ("line", QRect(lx, geom.top() + 2, 2, 26)))

        last_geom = buttons[-1].geometry()
        lx = min(w - 2, last_geom.right() + 2)
        return (group_key, len(buttons), ("line", QRect(lx, last_geom.top() + 2, 2, 26)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            self._drop_info = self._calc_drop_info(event.position().toPoint())
            self.update()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            self._drop_info = self._calc_drop_info(event.position().toPoint())
            self.update()

    def dragLeaveEvent(self, event):
        self._drop_info = None
        self.update()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            button_id = bytes(event.mimeData().data(TREE_BUTTON_MIME)).decode("utf-8")
            if self._drop_info:
                target_group, target_idx, _ = self._drop_info
                self.move_button(button_id, target_group, target_idx)
            self._drop_info = None
            self.update()

    def move_button(self, button_id: str, target_group: str, target_idx: int):
        src_group = None
        src_idx = None
        target_btn = None
        for g_name, b_list in self.header_layout.groups.items():
            for idx, b in enumerate(b_list):
                if b.button_id == button_id:
                    src_group = g_name
                    src_idx = idx
                    target_btn = b
                    break
            if src_group:
                break

        if not target_btn or target_group not in self.header_layout.groups:
            return

        if src_group == target_group:
            if src_idx == target_idx or target_idx == src_idx + 1:
                return
            self.header_layout.groups[src_group].pop(src_idx)
            if target_idx > src_idx:
                target_idx -= 1
            self.header_layout.groups[target_group].insert(target_idx, target_btn)
        else:
            self.header_layout.groups[src_group].pop(src_idx)
            self.header_layout.groups[target_group].insert(target_idx, target_btn)

        self.set_groups(self.header_layout.groups)
        self.layout_changed.emit()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        reset_act = menu.addAction("Reset Toolbar Layout")
        reset_act.triggered.connect(self.tree_widget.reset_toolbar_layout)
        menu.exec(event.globalPos())

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_info:
            painter = QPainter(self)
            try:
                acc_color = QColor(theme_color("accent"))
            except Exception:
                acc_color = QColor("#fabd2f")
            pen = QPen(acc_color, 2)
            painter.setPen(pen)
            kind, geom = self._drop_info[2]
            if kind == "line":
                painter.fillRect(geom, acc_color)
            elif kind == "box":
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(geom.adjusted(1, 1, -1, -1))
            painter.end()


class ConfigTreeWidget(TreeHeaderActions, QWidget):
    """File tree that mirrors in/config/ and allows selecting, duplicating,

    and creating experiments in groups.
    """
    file_selected = pyqtSignal(object, str)  # (Path, rel_path) - preview
    file_opened = pyqtSignal(object, str)    # (Path, rel_path) - pinned / double-click
    file_deleted = pyqtSignal(str)           # (rel_path)
    dir_deleted = pyqtSignal(str)            # (rel_path)
    dir_renamed = pyqtSignal(str, str)       # (old_rel, new_rel)
    duplicate_requested = pyqtSignal(str)     # (rel_path)
    new_in_group_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, root_dir=None, parent=None, mode="all", settings_manager=None):
        super().__init__(parent)
        self.root_dir = Path(root_dir) if root_dir else find_config_root()
        self.mode = mode  # "all", "experiments", "components"
        self.settings_manager = settings_manager
        self.current_rel_path = None
        self._dirty_paths = set()
        self.last_focused_group = None
        self._active_inline_item = None
        self._committing_group = False
        self._init_ui()
        self.populate()

    def _default_toolbar_groups(self) -> dict[str, list[DraggableToolButton]]:
        if self.mode == "components":
            return {
                "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
                "row1_right": [self.btn_plus, self.btn_hub, self.btn_toggle_raw],
                "row2_left": [self.btn_duplicate],
                "row2_right": [],
            }
        elif self.mode in ("experiments", "experiment"):
            return {
                "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
                "row1_right": [self.btn_launch, self.btn_export, self.btn_toggle_yaml],
                "row2_left": [self.btn_new_group, self.btn_new_subfolder, self.btn_new_experiment, self.btn_duplicate],
                "row2_right": [],
            }
        else:
            return {
                "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
                "row1_right": [],
                "row2_left": [self.btn_new_group, self.btn_new_subfolder, self.btn_new_experiment, self.btn_duplicate],
                "row2_right": [],
            }

    def _load_toolbar_groups(self) -> dict[str, list[DraggableToolButton]]:
        defaults = self._default_toolbar_groups()
        all_buttons = {b.button_id: b for g in defaults.values() for b in g}

        saved_state = None
        if self.settings_manager:
            try:
                saved_state = self.settings_manager.get("filetree", f"buttons_{self.mode}", default=None)
            except Exception:
                saved_state = None
        if not saved_state:
            settings = QSettings("ThetaIDE", "ThetaIDE")
            raw = settings.value(f"filetree_buttons_{self.mode}", None)
            if raw and isinstance(raw, str):
                try:
                    saved_state = json.loads(raw)
                except Exception:
                    saved_state = None

        if not isinstance(saved_state, dict):
            return defaults

        loaded_groups = {
            "row1_left": [],
            "row1_right": [],
            "row2_left": [],
            "row2_right": [],
        }
        placed = set()
        for g_name in ("row1_left", "row1_right", "row2_left", "row2_right"):
            b_ids = saved_state.get(g_name, [])
            if isinstance(b_ids, list):
                for bid in b_ids:
                    if bid in all_buttons and bid not in placed:
                        loaded_groups[g_name].append(all_buttons[bid])
                        placed.add(bid)

        # Append any newly added / missing buttons to their default location
        for g_name, b_list in defaults.items():
            for b in b_list:
                if b.button_id not in placed:
                    loaded_groups[g_name].append(b)
                    placed.add(b.button_id)

        return loaded_groups

    def _save_toolbar_layout(self):
        state = {
            g_name: [b.button_id for b in btn_list]
            for g_name, btn_list in self.header_widget.header_layout.groups.items()
        }
        if self.settings_manager:
            try:
                self.settings_manager.set("filetree", f"buttons_{self.mode}", state)
            except Exception:
                pass
        try:
            settings = QSettings("ThetaIDE", "ThetaIDE")
            settings.setValue(f"filetree_buttons_{self.mode}", json.dumps(state))
        except Exception:
            pass

    def reset_toolbar_layout(self):
        """Reset the header toolbar buttons to their default 2-row layout."""
        self.header_widget.set_groups(self._default_toolbar_groups())
        self._save_toolbar_layout()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_refresh = self._tool_button("reload", button_id="reload")
        self.btn_refresh.setToolTip("Reload configuration files from disk")
        self.btn_refresh.clicked.connect(self.populate)

        self.btn_collapse = self._tool_button("collapse_all", button_id="collapse_all")
        self.btn_collapse.setToolTip("Collapse all folders in the tree")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_collapse_all = self.btn_collapse  # alias

        self.btn_expand = self._tool_button("expand_all", button_id="expand_all")
        self.btn_expand.setToolTip("Expand all folders in the tree")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_expand_all = self.btn_expand  # alias

        self.btn_search = self._tool_button("search", button_id="search")
        self.btn_search.setToolTip("Filter configurations")

        if self.mode == "components":
            self.btn_plus = self._tool_button("plus", "+", button_id="plus")
            self.btn_plus.setToolTip("Scaffold a new component plugin (+)")
            self.btn_plus.clicked.connect(self.prompt_scaffold_component)
            self.btn_new = self.btn_plus
            self.btn_new_group = self.btn_new  # backwards compatibility alias
            self.btn_new_subfolder = None
            self.btn_new_experiment = self.btn_new

            self.btn_duplicate = self._tool_button("copy", "Copy", button_id="copy")
            self.btn_duplicate.setToolTip("Duplicate currently selected component")
            self.btn_duplicate.clicked.connect(self.prompt_duplicate)

            self.btn_hub = self._tool_button("hub", "Component Hub", button_id="hub")
            self.btn_hub.setCheckable(True)
            self.btn_hub.setChecked(False)
            self.btn_hub.setToolTip("Browse and install new RL methods, models, and environments from the Community Hub")

            self.btn_toggle_raw = self._tool_button("raw_yaml", "View Raw YAML", button_id="raw_yaml")
            self.btn_toggle_raw.setCheckable(True)
            self.btn_toggle_raw.setChecked(False)
            self.btn_toggle_raw.setToolTip("Toggle side-by-side view of the raw YAML file content")

            self.btn_launch = None
            self.btn_start = None
            self.start_button = None
            self.btn_export = None
            self.btn_toggle_yaml = None
        else:
            self.btn_new_group = self._tool_button("new_folder", "New Group", button_id="new_folder")
            self.btn_new_group.setToolTip("Create a new experiment group")
            self.btn_new_group.clicked.connect(self.create_new_group_inline)

            self.btn_new_subfolder = self._tool_button("new_subfolder", "New Subfolder", button_id="new_subfolder")
            self.btn_new_subfolder.setToolTip("Create a new subfolder in the active group")
            self.btn_new_subfolder.clicked.connect(self.prompt_new_subfolder_in_active_group)

            self.btn_new_experiment = self._tool_button("new_file", "New Experiment", button_id="new_file")
            self.btn_new_experiment.setToolTip("Create a new default experiment in a group")
            self.btn_new_experiment.clicked.connect(self.prompt_new_experiment_in_active_group)

            self.btn_new = self.btn_new_experiment

            self.btn_duplicate = self._tool_button("copy", "Copy", button_id="copy")
            if self.mode in ("experiments", "experiment"):
                self.btn_duplicate.setToolTip("Duplicate currently selected experiment")
            elif self.mode == "components":
                self.btn_duplicate.setToolTip("Duplicate currently selected component")
            else:
                self.btn_duplicate.setToolTip("Duplicate currently selected configuration")
            self.btn_duplicate.clicked.connect(self.prompt_duplicate)

            self.btn_hub = None
            self.btn_toggle_raw = None

            if self.mode in ("experiments", "experiment"):
                self.btn_launch = self._tool_button("launch", "Launch training", button_id="launch")
                self.btn_launch.setToolTip("Train the loaded experiment config through the backend (F5)")
                self.btn_launch.setEnabled(False)
                self.btn_launch.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
                self.btn_start = self.btn_launch
                self.start_button = self.btn_launch

                self.btn_export = self._tool_button("export", "Export recipe YAML…", button_id="export")
                self.btn_export.setToolTip("Export recipe YAML…")

                self.btn_toggle_yaml = self._tool_button("raw_yaml", "View Hydra YAML", button_id="raw_yaml")
                self.btn_toggle_yaml.setCheckable(True)
                self.btn_toggle_yaml.setChecked(False)
                self.btn_toggle_yaml.setToolTip("Toggle preview of the resolved Hydra YAML configuration")
            else:
                self.btn_launch = None
                self.btn_start = None
                self.start_button = None
                self.btn_export = None
                self.btn_toggle_yaml = None

        self.btn_save = None

        # Two-row header widget with draggable, reorderable button groups
        self.header_widget = TreeHeaderWidget(self)
        groups = self._load_toolbar_groups()
        self.header_widget.set_groups(groups)
        self.header_widget.layout_changed.connect(self._save_toolbar_layout)
        self.header_layout = self.header_widget.header_layout
        layout.addWidget(self.header_widget)

        # Search filter (hidden by default, toggled via magnifying glass button)
        self.search = QLineEdit()
        if self.mode in ("experiments", "experiment"):
            placeholder = "Filter experiments…"
        elif self.mode == "components":
            placeholder = "Filter components…"
        else:
            placeholder = "Filter configs…"
        self.search.setPlaceholderText(placeholder)
        self.search.textChanged.connect(self.filter_tree)
        self.watch_search()
        layout.addWidget(self.search)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setItemDelegate(GroupItemDelegate(self.tree))
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.currentItemChanged.connect(self._on_tree_current_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree, 1)

        self.footer = None

    def get_expanded_paths(self) -> set[str]:
        """Return the set of relative directory paths that are currently expanded."""
        expanded = set()

        def _collect(parent):
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                child = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "dir":
                    if child.isExpanded():
                        rel = str(data.get("rel_path", "")).replace("\\", "/").strip("/")
                        if rel:
                            expanded.add(rel)
                _collect(child)

        _collect(self.tree)
        return expanded

    def populate(self, ensure_expanded: str | None = None):
        """Recursively scan root_dir and populate the tree based on mode."""
        expanded_paths = self.get_expanded_paths()
        if ensure_expanded:
            clean_exp = str(Path(ensure_expanded)).replace("\\", "/").strip("/")
            if clean_exp and clean_exp != ".":
                expanded_paths.add(clean_exp)
                if self.mode in ("experiments", "experiment") and not clean_exp.startswith("experiment/"):
                    expanded_paths.add(f"experiment/{clean_exp}")
                parts = clean_exp.split("/")
                for i in range(1, len(parts)):
                    expanded_paths.add("/".join(parts[:i]))

        has_previous_state = len(expanded_paths) > 0

        self.tree.clear()
        if not self.root_dir.exists():
            item = QTreeWidgetItem(self.tree, ["in/config (not found)"])
            item.setToolTip(0, f"Directory not found: {self.root_dir}")
            return

        existing_dirs = {p.name: p for p in self.root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}
        top_files = sorted([p for p in self.root_dir.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])

        if self.mode in ("experiments", "experiment"):
            # The tree directory is in/config/experiment/ - show groups and files directly
            if self.root_dir.name in ("experiment", "experiments"):
                exp_dir = self.root_dir
            else:
                exp_dir = existing_dirs.get("experiment") or existing_dirs.get("experiments") or (self.root_dir / "experiment")
            if exp_dir.exists():
                subdirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
                files = sorted([p for p in exp_dir.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])
                base_files = [p for p in files if p.stem == "_base"]
                other_files = [p for p in files if p.stem != "_base"]
                for f in base_files:
                    self._add_file_node(self.tree, f)
                for sub in subdirs:
                    self._add_dir_node(self.tree, sub, expand=True, expanded_set=expanded_paths, has_previous_state=has_previous_state)
                for f in other_files:
                    self._add_file_node(self.tree, f)
        elif self.mode == "components":
            # Display all modular configuration directories and files OUTSIDE experiment
            # 1. Component Headers (styled identically to experiment groups: is_group=True, bold large font, underline, no icon)
            # agent -> Algorithms
            # env -> Environments
            # model -> Models
            # data -> Data Loaders
            # eval -> Evaluators
            # plot -> Figures
            component_headers = [
                ("agent", "Algorithms"),
                ("env", "Environments"),
                ("model", "Models"),
                ("data", "Data Loaders"),
                ("eval", "Evaluators"),
                ("plot", "Figures"),
            ]
            standard_create = {"agent", "env", "model", "data", "eval", "plot"}

            for cat, header_title in component_headers:
                cat_dir = self.root_dir / cat
                if cat == "plot" and not cat_dir.is_dir() and (self.root_dir / "plots").is_dir():
                    cat_dir = self.root_dir / "plots"

                if not cat_dir.is_dir() and cat in standard_create:
                    try:
                        cat_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                if cat_dir.is_dir():
                    self._add_dir_node(
                        self.tree,
                        cat_dir,
                        expand=True,
                        expanded_set=expanded_paths,
                        has_previous_state=has_previous_state,
                        is_group=True,
                        custom_display_name=header_title,
                    )

            # 2. System Header at the end
            # "Put a System header at the end, and have the paradigm, site, hydra, and workflow config "groups" stay as they are under this system header."
            system_node = QTreeWidgetItem(self.tree, ["System"])
            f = group_heading_font(self.tree.font())
            system_node.setFont(0, f)
            system_node.setData(0, Qt.ItemDataRole.FontRole, f)
            system_node.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "dir",
                "path": str(self.root_dir / "system"),
                "rel_path": "system",
                "is_group": True,
            })
            system_node.setToolTip(0, "System configurations (paradigms, site, hydra, workflow, and global defaults)")

            # Under System:
            # - Global Pipeline Defaults (config.yaml) pinned at top
            global_config = self.root_dir / "config.yaml"
            if global_config.is_file():
                self._add_file_node(system_node, global_config, custom_display_name="Global Defaults (config)", is_global_config=True)

            # - System config groups (normal folder/category items with icons)
            system_groups = ["paradigms", "site", "hydra", "workflow"]
            for grp in system_groups:
                grp_dir = self.root_dir / grp
                if not grp_dir.is_dir() and grp in ("paradigms", "site", "hydra", "workflow"):
                    try:
                        grp_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                if grp_dir.is_dir():
                    self._add_dir_node(
                        system_node,
                        grp_dir,
                        expand=False,
                        expanded_set=expanded_paths,
                        has_previous_state=has_previous_state,
                        is_group=False,
                    )

            # Any other custom directories outside components, system groups, and experiment
            all_known = set(c[0] for c in component_headers) | {"plots", "experiment", "experiments"} | set(system_groups)
            for name, dir_path in sorted(existing_dirs.items()):
                if name not in all_known:
                    self._add_dir_node(
                        system_node,
                        dir_path,
                        expand=False,
                        expanded_set=expanded_paths,
                        has_previous_state=has_previous_state,
                        is_group=False,
                    )

            for f in top_files:
                if f.name != "config.yaml":
                    self._add_file_node(system_node, f)

            if has_previous_state:
                if "system" in expanded_paths:
                    system_node.setExpanded(True)
            else:
                system_node.setExpanded(True)
        else:
            # "all": Top-level directory ordering with experiment first
            preferred_order = ["experiment", "agent", "env", "model", "data", "eval", "plot", "paradigms", "site", "hydra"]
            standard_create = {"experiment", "agent", "env", "model", "data", "eval", "plot", "paradigms", "site", "hydra"}
            for cat in preferred_order:
                cat_dir = self.root_dir / cat
                if not cat_dir.is_dir() and cat in standard_create:
                    try:
                        cat_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                if cat_dir.is_dir():
                    self._add_dir_node(self.tree, cat_dir, expand=(cat == "experiment"), expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for name, dir_path in sorted(existing_dirs.items()):
                if name not in preferred_order:
                    self._add_dir_node(self.tree, dir_path, expand=False, expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for f in top_files:
                self._add_file_node(self.tree, f)

        # Re-apply current selection if possible without triggering external selection signals
        if self.current_rel_path:
            self.select_file(self.current_rel_path, emit_signal=False)

    def _tool_button(self, icon_name, text=None, button_id=None):
        """Square header button with a themed SVG icon; text, if any, becomes its accessible name."""
        button = style_tool_button(DraggableToolButton(button_id or icon_name), icon_name)
        if text:
            button.setAccessibleName(text)
        return button

    def refresh_icons(self):
        """Re-render the header and tree icons in the current theme's colors."""
        refresh_tool_icons(self)
        if hasattr(self, "header_widget"):
            self.header_widget.update()

        def _update_tree(parent):
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                child = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data:
                    if data.get("is_base"):
                        child.setIcon(0, svg_icon("gear", TREE_GEAR_COLORS))
                    elif data.get("is_global_config"):
                        child.setIcon(0, svg_icon("config", TOOL_ICON_COLORS))
                    elif data.get("type") == "dir":
                        if data.get("is_group"):
                            child.setIcon(0, QIcon())
                        else:
                            rel = data.get("rel_path", "")
                            dname = Path(data.get("path", "")).name
                            c_icon = CATEGORY_ICONS.get(dname) or CATEGORY_ICONS.get(rel)
                            if c_icon:
                                child.setIcon(0, svg_icon(c_icon, TOOL_ICON_COLORS))
                            elif dname == "methods":
                                child.setIcon(0, svg_icon("components", TOOL_ICON_COLORS))
                            else:
                                child.setIcon(0, folder_icon())
                _update_tree(child)

        _update_tree(self.tree)

    def _add_dir_node(
        self,
        parent_widget,
        dir_path: Path,
        expand=False,
        expanded_set=None,
        has_previous_state=False,
        is_group: bool | None = None,
        custom_display_name: str | None = None,
    ):
        name = dir_path.name
        rel_path = str(dir_path.relative_to(self.root_dir)).replace("\\", "/").strip("/")

        display_name = custom_display_name or name
        if rel_path == "hydra/sweeper" or (name == "sweeper" and "hydra" in rel_path):
            display_name = "Hyperparameter Sweepers"

        node = QTreeWidgetItem(parent_widget, [display_name])

        if is_group is None:
            is_group = bool(self.mode in ("experiments", "experiment") and (dir_path.parent == self.root_dir or dir_path.parent.name in ("experiment", "experiments")))

        # Semantic icon based on directory category / purpose
        icon_name = CATEGORY_ICONS.get(name) or CATEGORY_ICONS.get(rel_path)
        if is_group:
            # Top-level group / component header (e.g. cartpole, Algorithms, Figures): no icon, larger text, subtle line
            f = group_heading_font(self.tree.font())
            node.setFont(0, f)
            node.setData(0, Qt.ItemDataRole.FontRole, f)
        elif icon_name:
            node.setIcon(0, svg_icon(icon_name, TOOL_ICON_COLORS))
        elif name == "methods":
            node.setIcon(0, svg_icon("components", TOOL_ICON_COLORS))
        else:
            node.setIcon(0, folder_icon())

        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "dir",
            "path": str(dir_path),
            "rel_path": rel_path,
            "is_group": is_group,
        })
        node.setToolTip(0, rel_path)

        # Subdirectories and files: pin group defaults (_base.yaml) to top above subdirs and other files
        subdirs = sorted([p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")])
        files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])

        base_files = [p for p in files if p.stem == "_base"]
        other_files = [p for p in files if p.stem != "_base"]

        for f in base_files:
            self._add_file_node(node, f)

        for sub in subdirs:
            # Under experiment, expand group folders like cartpole, mimic
            sub_expand = (name == "experiment")
            self._add_dir_node(node, sub, expand=sub_expand, expanded_set=expanded_set, has_previous_state=has_previous_state)

        for f in other_files:
            self._add_file_node(node, f)

        if has_previous_state:
            should_expand = (expanded_set is not None and rel_path in expanded_set)
        else:
            should_expand = (expanded_set is not None and rel_path in expanded_set) or expand

        if should_expand:
            node.setExpanded(True)

        return node

    def _add_file_node(self, parent_node, file_path: Path, custom_display_name: str | None = None, is_global_config: bool = False):
        name = file_path.name
        rel_path = file_path.relative_to(self.root_dir).as_posix()
        is_exp = rel_path.startswith("experiment/") and not name.startswith("_")
        is_base = file_path.stem == "_base"

        display_name = custom_display_name or name
        if not custom_display_name:
            if display_name.endswith(".yaml"):
                display_name = display_name[:-5]
            elif display_name.endswith(".yml"):
                display_name = display_name[:-4]

            if is_base:
                display_name = "group defaults"

        node = QTreeWidgetItem(parent_node, [display_name])
        if is_base:
            node.setIcon(0, svg_icon("gear", TREE_GEAR_COLORS))
        elif is_global_config:
            node.setIcon(0, svg_icon("config", TOOL_ICON_COLORS))
        elif file_path.suffix not in (".yaml", ".yml"):
            node.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "file",
            "path": str(file_path),
            "rel_path": rel_path,
            "is_experiment": is_exp,
            "is_base": is_base,
            "is_global_config": is_global_config,
            "base_name": display_name,
        })
        node.setData(0, Qt.ItemDataRole.ForegroundRole, None)
        node.setToolTip(0, rel_path)
        return node

    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data.get("is_group"):
            self.last_focused_group = Path(data.get("path", "")).name
        elif data.get("rel_path"):
            grp, _ = self._split_group_and_subpath(data["rel_path"])
            if grp:
                self.last_focused_group = grp
        if data.get("type") == "file":
            self.current_rel_path = data["rel_path"]
            self.file_selected.emit(Path(data["path"]), data["rel_path"])

    def _on_tree_current_changed(self, current, previous):
        if not current:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data:
            if data.get("is_group"):
                self.last_focused_group = Path(data.get("path", "")).name
            elif data.get("rel_path"):
                grp, _ = self._split_group_and_subpath(data["rel_path"])
                if grp:
                    self.last_focused_group = grp

    def _on_item_double_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data.get("is_group"):
            self.last_focused_group = Path(data.get("path", "")).name
        elif data.get("rel_path"):
            grp, _ = self._split_group_and_subpath(data["rel_path"])
            if grp:
                self.last_focused_group = grp
        if data.get("type") == "file":
            self.current_rel_path = data["rel_path"]
            self.file_opened.emit(Path(data["path"]), data["rel_path"])

    def filter_tree(self, text):
        query = text.strip().lower()

        def match_and_filter(item):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            item_text = item.text(0).lower()
            rel_path = (data.get("rel_path") or "").lower() if data else ""
            self_match = (query in item_text) or (query in rel_path)

            child_matched = False
            for i in range(item.childCount()):
                if match_and_filter(item.child(i)):
                    child_matched = True

            visible = self_match or child_matched or (not query)
            item.setHidden(not visible)
            if query and (self_match or child_matched):
                item.setExpanded(True)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            match_and_filter(self.tree.topLevelItem(i))

    def find_file_item(self, target_rel_path: str):
        """Find the QTreeWidgetItem corresponding to a relative path."""
        target_norm = str(Path(target_rel_path)).replace("\\", "/")

        def find_item(parent):
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                item = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "file":
                    item_rel = str(Path(data.get("rel_path", ""))).replace("\\", "/")
                    if (
                        item_rel == target_norm
                        or item_rel.endswith(target_norm)
                        or item_rel.removesuffix(".yaml") == target_norm
                        or item_rel.removesuffix(".yaml").endswith(target_norm)
                    ):
                        return item
                found = find_item(item)
                if found:
                    return found
            return None

        return find_item(self.tree)

    def set_file_dirty(self, rel_path: str, is_dirty: bool = False):
        """Keep file names clean; state is auto-saved directly to disk."""
        if not rel_path:
            return
        target_norm = str(Path(rel_path)).replace("\\", "/")
        item = self.find_file_item(target_norm)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        base_name = data.get("base_name") or item.text(0).rstrip(" ●")
        item.setText(0, base_name)
        item.setData(0, Qt.ItemDataRole.ForegroundRole, None)
        item.setToolTip(0, data.get("rel_path", rel_path))

    def select_file(self, target_rel_path: str, emit_signal: bool = True):
        """Find and select an item by its relative path."""
        found = self.find_file_item(target_rel_path)
        if found:
            self.tree.setCurrentItem(found)
            # Ensure all parent items are expanded
            curr = found.parent()
            while curr:
                curr.setExpanded(True)
                curr = curr.parent()
            data = found.data(0, Qt.ItemDataRole.UserRole)
            self.current_rel_path = data["rel_path"]
            if emit_signal:
                self.file_selected.emit(Path(data["path"]), data["rel_path"])
            return True
        return False

    def _model(self):
        """The parsed in/config tree, or None when it cannot be read."""
        try:
            return ConfigTree(self.root_dir)
        except (OSError, ValueError):
            return None

    def _ensure_group_base(self, tree, group_name, paradigm, env):
        """Write the group's _base.yaml if it has none.

        An experiment inherits its environment and paradigm from the group base,
        so creating one in a group without a base produces a config Hydra cannot
        even load ("Could not load 'experiment/<group>/_base'").
        """
        group = tree.group(group_name)
        if group.has_base:
            return group.paradigm
        base_dir = self.root_dir / "experiment" / group_name
        base_dir.mkdir(parents=True, exist_ok=True)
        base_yaml = group_base_yaml(env, paradigm, tree.paradigms.get(paradigm))
        (base_dir / "_base.yaml").write_text(base_yaml, encoding="utf-8")
        return paradigm

    def _split_group_and_subpath(self, rel_path: str) -> tuple[str, str]:
        """Split a relative path under experiment into (group_name, subpath)."""
        clean = rel_path
        if clean.startswith("experiment/"):
            clean = clean[len("experiment/"):]
        parts = clean.strip("/").split("/")
        group = parts[0]
        subpath = "/".join(parts[1:]) if len(parts) > 1 else ""
        return group, subpath

    def _write_experiment(self, group_name, filename, exp_id, paradigm=None, env=None, subpath=""):
        """Create an experiment valid for its group's paradigm. Returns rel path or None."""
        tree = self._model()
        paradigm_obj, agent, model = None, None, None
        if tree is not None:
            paradigm_name = self._ensure_group_base(tree, group_name, paradigm, env)
            paradigm_obj = tree.paradigms.get(paradigm_name) if paradigm_name else None
            permitted = [a.name for a in tree.agents_for(paradigm_name)] if paradigm_name else []
            agent = permitted[0] if permitted else None
            if paradigm_name and not permitted:
                # A paradigm with no agents still requires methods to be
                # non-empty, and supervised satisfies that with model-only
                # methods, so seed one instead of writing an invalid file.
                model = default_method_model(
                    tree.group(group_name), tree.models, paradigm=tree.paradigms.get(paradigm_name)
                )

        if subpath:
            target_dir = (self.root_dir / "experiment" / group_name / subpath) if self.root_dir.name not in ("experiment", "experiments") else (self.root_dir / group_name / subpath)
        else:
            target_dir = (self.root_dir / "experiment" / group_name) if self.root_dir.name not in ("experiment", "experiments") else (self.root_dir / group_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Experiment file already exists:\n{target_file}")
            return None
        try:
            target_file.write_text(
                experiment_yaml(group_name, exp_id, paradigm_obj, agent, model), encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, "Error Creating Experiment", f"Could not create file:\n{exc}")
            return None
        if subpath:
            return f"experiment/{group_name}/{subpath}/{filename}"
        return f"experiment/{group_name}/{filename}"

    def get_available_groups(self):
        """Return sorted list of experiment groups in in/config/experiment/."""
        exp_dir = self.root_dir if self.root_dir.name in ("experiment", "experiments") else (self.root_dir / "experiment")
        if not exp_dir.exists():
            return []
        groups = [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        return sorted(groups)

    def get_active_group(self) -> str | None:
        """Return the last focused or currently selected experiment group name."""
        exp_dir = self.root_dir if self.root_dir.name in ("experiment", "experiments") else (self.root_dir / "experiment")
        if self.last_focused_group and (exp_dir / self.last_focused_group).is_dir():
            return self.last_focused_group

        curr = self.tree.currentItem()
        if curr:
            data = curr.data(0, Qt.ItemDataRole.UserRole)
            if data:
                if data.get("is_group"):
                    grp = Path(data.get("path", "")).name
                    if (exp_dir / grp).is_dir():
                        self.last_focused_group = grp
                        return grp
                elif data.get("rel_path"):
                    grp, _ = self._split_group_and_subpath(data["rel_path"])
                    if grp and (exp_dir / grp).is_dir():
                        self.last_focused_group = grp
                        return grp

        available = self.get_available_groups()
        if available:
            self.last_focused_group = available[0]
            return available[0]
        return None

    def create_new_group_inline(self):
        """Immediately place user cursor on a new group label in the tree with an empty name."""
        if getattr(self, "_active_inline_item", None):
            self._cancel_inline_group(self._active_inline_item)

        item = QTreeWidgetItem([""])
        item.setData(0, Qt.ItemDataRole.UserRole, {"type": "dir", "is_group": True})

        f = group_heading_font(self.tree.font())
        item.setFont(0, f)
        item.setData(0, Qt.ItemDataRole.FontRole, f)

        self.tree.insertTopLevelItem(0, item)
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)

        editor = InlineGroupEditor()
        editor.setPlaceholderText("Group name...")
        editor.setFont(f)
        editor.setStyleSheet("padding: 2px 4px; font-weight: bold;")
        self.tree.setItemWidget(item, 0, editor)
        self._active_inline_item = item
        self._committing_group = False

        def _do_commit():
            if getattr(self, "_committing_group", False):
                return
            self._committing_group = True
            raw_name = editor.text().strip()
            if not raw_name:
                self._cancel_inline_group(item)
                self._committing_group = False
                return

            # Validation: slashes and invalid filesystem chars
            if "/" in raw_name or "\\" in raw_name or any(c in raw_name for c in '<>:"|?*'):
                QMessageBox.warning(self, "Invalid Group Name", f"Group name cannot contain slashes or special characters: '{raw_name}'")
                self._committing_group = False
                editor.setFocus()
                return

            exp_dir = self.root_dir if self.root_dir.name in ("experiment", "experiments") else (self.root_dir / "experiment")
            target_group_dir = exp_dir / raw_name
            if target_group_dir.exists():
                QMessageBox.warning(self, "Group Already Exists", f"A group named '{raw_name}' already exists.")
                self._committing_group = False
                editor.setFocus()
                return

            try:
                target_group_dir.mkdir(parents=True, exist_ok=True)
                base_file = target_group_dir / "_base.yaml"
                base_content = (
                    "# @package _global_\n"
                    "defaults:\n"
                    "  - override /env: null\n\n"
                    "paradigm: null\n\n"
                    "seed: 1\n"
                    "save_dataset: false\n"
                    "recover: false\n"
                )
                base_file.write_text(base_content, encoding="utf-8")
            except OSError as exc:
                QMessageBox.critical(self, "Error Creating Group", f"Could not create group directory:\n{exc}")
                self._cancel_inline_group(item)
                self._committing_group = False
                return

            self._active_inline_item = None
            self._committing_group = False
            self.last_focused_group = raw_name
            self.populate(ensure_expanded=f"experiment/{raw_name}")
            rel_path = f"experiment/{raw_name}/_base.yaml" if self.root_dir.name not in ("experiment", "experiments") else f"{raw_name}/_base.yaml"
            self.select_file(rel_path, emit_signal=True)

        editor.returnPressed.connect(_do_commit)
        editor.escape_pressed.connect(lambda: self._cancel_inline_group(item))
        editor.lost_focus_empty.connect(lambda: self._cancel_inline_group(item))

        editor.setFocus()

    def _cancel_inline_group(self, item):
        """Cancel inline group creation and clean up the placeholder item."""
        if getattr(self, "_active_inline_item", None) is item:
            self._active_inline_item = None
        self.tree.removeItemWidget(item, 0)
        idx = self.tree.indexOfTopLevelItem(item)
        if idx >= 0:
            self.tree.takeTopLevelItem(idx)

    def prompt_new_subfolder_in_active_group(self):
        """Create a new subfolder in the active/last focused group."""
        group_name = self.get_active_group()
        if not group_name:
            QMessageBox.information(self, "No Group Available", "Please create an experiment group first.")
            return

        target_rel = f"experiment/{group_name}" if self.root_dir.name not in ("experiment", "experiments") else group_name
        curr = self.tree.currentItem()
        if curr:
            data = curr.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get("type") == "dir" and not data.get("is_group"):
                rel = data.get("rel_path", "")
                grp, _ = self._split_group_and_subpath(rel)
                if grp == group_name:
                    target_rel = rel

        folder_name, ok = QInputDialog.getText(
            self, "New Subfolder", f"Folder name in '{target_rel}':",
            QLineEdit.EchoMode.Normal, "new_folder"
        )
        if ok and folder_name.strip():
            _, err = self.create_dir(target_rel, folder_name.strip())
            if err:
                QMessageBox.warning(self, "Cannot Create Subfolder", err)

    def prompt_new_experiment_in_active_group(self, group=None, subpath=None):
        """Create a new experiment in the active or specified group, generated from group defaults."""
        group_name = group or self.get_active_group()
        if not group_name:
            QMessageBox.information(self, "No Group Available", "Please create an experiment group first.")
            return

        if subpath is None:
            subpath = ""
            curr = self.tree.currentItem()
            if curr:
                data = curr.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("rel_path"):
                    grp, sub = self._split_group_and_subpath(data["rel_path"])
                    if grp == group_name and data.get("type") == "dir" and not data.get("is_group"):
                        subpath = sub

        display_target = f"{group_name}/{subpath}" if subpath else group_name
        name, ok = QInputDialog.getText(
            self, "New Experiment", f"Experiment name in '{display_target}':",
            QLineEdit.EchoMode.Normal, "new_experiment"
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        exp_id = filename[:-5]

        rel_path = self._write_experiment_from_defaults(group_name, filename, exp_id, subpath=subpath)
        if rel_path:
            self.populate()
            self.select_file(rel_path)

    def _write_experiment_from_defaults(self, group_name: str, filename: str, exp_id: str, subpath: str = "") -> str | None:
        """Create a new experiment generated directly from group defaults.
        Anything required not in group defaults is included with empty value (null).
        """
        tree = self._model()
        group_obj = tree.group(group_name) if tree else None
        group_paradigm = group_obj.paradigm if group_obj else None
        group_env = group_obj.env if group_obj else None

        if subpath:
            target_dir = (self.root_dir / "experiment" / group_name / subpath) if self.root_dir.name not in ("experiment", "experiments") else (self.root_dir / group_name / subpath)
        else:
            target_dir = (self.root_dir / "experiment" / group_name) if self.root_dir.name not in ("experiment", "experiments") else (self.root_dir / group_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Experiment file already exists:\n{target_file}")
            return None

        lines = [
            "# @package _global_",
            "defaults:",
            f"  - {group_name}/_base",
            "",
            f"experiment_id: {exp_id}",
        ]

        # If group defaults do not specify paradigm, prompt it with empty value:
        if not group_paradigm:
            lines.append("paradigm: null")

        # If group defaults do not specify environment, prompt it with empty value:
        if not group_env:
            lines.append("env: null")

        # Methods block (only if group base does not already specify methods)
        if not (group_obj and group_obj.base_methods) and tree and group_paradigm:
            permitted = [a.name for a in tree.agents_for(group_paradigm)]
            agent = permitted[0] if permitted else None
            model = (
                default_method_model(group_obj, tree.models, paradigm=tree.paradigms.get(group_paradigm))
                if not agent
                else "dnn"
            )
            if agent or model:
                m_name = unique_method_name((), agent, model)
                lines += ["", "methods:", f"  {m_name}:"]
                if agent:
                    lines.append(f"    agent: {agent}")
                lines.append(f"    model: {model}")

        content = "\n".join(lines) + "\n"
        try:
            target_file.write_text(content, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Error Creating Experiment", f"Could not create file:\n{exc}")
            return None

        if subpath:
            return f"experiment/{group_name}/{subpath}/{filename}" if self.root_dir.name not in ("experiment", "experiments") else f"{group_name}/{subpath}/{filename}"
        return f"experiment/{group_name}/{filename}" if self.root_dir.name not in ("experiment", "experiments") else f"{group_name}/{filename}"

    def prompt_new_in_folder(self, folder_rel_path: str):
        """Create a new experiment inside a specific subfolder."""
        group_name, subpath = self._split_group_and_subpath(folder_rel_path)
        self.last_focused_group = group_name
        self.prompt_new_experiment_in_active_group(group=group_name, subpath=subpath)

    def prompt_new_in_group(self, group=None):
        """Create a new experiment, in an existing group or a brand new one."""
        if group:
            self.last_focused_group = group
        self.prompt_new_experiment_in_active_group(group=group)

    def prompt_duplicate(self):
        """Prompt to duplicate the currently selected experiment."""
        if not self.current_rel_path:
            QMessageBox.information(self, "No File Selected", "Please select an experiment YAML file first.")
            return

        source_file = self.root_dir / self.current_rel_path
        if not source_file.exists() or not source_file.is_file():
            QMessageBox.warning(self, "Invalid Selection", "Selected item is not a valid file.")
            return

        stem = source_file.stem
        parent_dir = source_file.parent
        new_name, ok = QInputDialog.getText(
            self, "Duplicate Experiment", f"Duplicate '{stem}' as:",
            QLineEdit.EchoMode.Normal, f"{stem}_copy"
        )
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        new_filename = f"{new_name}.yaml" if not new_name.endswith(".yaml") else new_name
        target_file = parent_dir / new_filename

        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Target file already exists:\n{target_file}")
            return

        try:
            raw_text = source_file.read_text(encoding="utf-8")
            # If experiment_id is declared, update it
            try:
                parsed = yaml.safe_load(raw_text)
                if isinstance(parsed, dict) and "experiment_id" in parsed:
                    parsed["experiment_id"] = new_name.replace(".yaml", "")
                    new_text = yaml.safe_dump(parsed, sort_keys=False)
                else:
                    new_text = raw_text
            except Exception:
                new_text = raw_text

            target_file.write_text(new_text, encoding="utf-8")
            self.populate()
            new_rel = target_file.relative_to(self.root_dir).as_posix()
            self.select_file(new_rel)
        except OSError as exc:
            QMessageBox.critical(self, "Duplicate Error", f"Could not duplicate file:\n{exc}")

    def delete_config(self, rel_path):
        """Delete a config file. Returns (removed, error).

        Opens no dialogs: a modal message box blocks forever when nobody is
        there to dismiss it, so the caller reports the error instead.
        """
        path = self.root_dir / rel_path
        blocked = deletion_blocked_reason(path)
        if blocked:
            return False, blocked
        try:
            path.unlink()
        except OSError as exc:
            return False, str(exc)
        if self.current_rel_path == rel_path:
            self.current_rel_path = None
        from .config_viewer import invalidate_config_tree
        invalidate_config_tree()
        self.file_deleted.emit(rel_path)
        self.populate()
        return True, None

    def rename_config(self, rel_path, new_name):
        """Rename a config, keeping experiment_id in step. Returns (new_rel, error)."""
        path = self.root_dir / rel_path
        new_name = (new_name or "").strip()
        if not new_name:
            return None, "Give the file a name."
        if not new_name.endswith(".yaml"):
            new_name += ".yaml"
        target = path.parent / new_name
        if target == path:
            return rel_path, None
        if target.exists():
            return None, f"{new_name} already exists in this folder."
        try:
            text = path.read_text(encoding="utf-8")
            path.rename(target)
            target.write_text(rename_experiment_text(text, target.stem), encoding="utf-8")
        except OSError as exc:
            return None, str(exc)
        from .config_viewer import invalidate_config_tree
        invalidate_config_tree()
        new_rel = str(Path(rel_path).parent / new_name)
        self.populate()
        self.select_file(new_rel)
        return new_rel, None

    def create_dir(self, parent_rel: str, dir_name: str) -> tuple[str | None, str | None]:
        """Create a new subdirectory inside parent_rel. Returns (new_rel, error)."""
        dir_name = (dir_name or "").strip()
        if not dir_name:
            return None, "Folder name cannot be empty."
        if "/" in dir_name or "\\" in dir_name:
            return None, "Folder name cannot contain slashes."
        parent_dir = self.root_dir / parent_rel
        target_dir = parent_dir / dir_name
        if target_dir.exists():
            return None, f"A folder named '{dir_name}' already exists."
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return None, str(exc)
        new_rel = str(target_dir.relative_to(self.root_dir)).replace("\\", "/")
        self.populate(ensure_expanded=new_rel)
        return new_rel, None

    def rename_dir(self, rel_path: str, new_name: str) -> tuple[str | None, str | None]:
        """Rename a directory. Returns (new_rel, error)."""
        new_name = (new_name or "").strip()
        if not new_name:
            return None, "Folder name cannot be empty."
        if "/" in new_name or "\\" in new_name:
            return None, "Folder name cannot contain slashes."
        old_dir = self.root_dir / rel_path
        if not old_dir.exists() or not old_dir.is_dir():
            return None, f"Folder '{rel_path}' does not exist."
        if old_dir.name == new_name:
            return rel_path, None
        new_dir = old_dir.with_name(new_name)
        if new_dir.exists():
            return None, f"A folder named '{new_name}' already exists."
        try:
            old_dir.rename(new_dir)
        except OSError as exc:
            return None, str(exc)
        new_rel = str(new_dir.relative_to(self.root_dir)).replace("\\", "/")
        self.dir_renamed.emit(rel_path, new_rel)
        self.populate(ensure_expanded=new_rel)
        return new_rel, None

    def delete_dir(self, rel_path: str) -> tuple[bool, str | None]:
        """Delete a directory and all contents. Returns (removed, error)."""
        target = self.root_dir / rel_path
        if not target.exists() or not target.is_dir():
            return False, f"Folder '{rel_path}' does not exist."
        blocked = deletion_blocked_reason(target)
        if blocked:
            return False, blocked
        try:
            shutil.rmtree(target)
        except OSError as exc:
            return False, str(exc)
        if self.current_rel_path and (self.current_rel_path == rel_path or self.current_rel_path.startswith(rel_path + "/")):
            self.current_rel_path = None
        self.dir_deleted.emit(rel_path)
        self.populate()
        return True, None

    def prompt_new_folder(self, parent_rel_path: str):
        """Prompt to create a new subfolder inside parent_rel_path."""
        folder_name, ok = QInputDialog.getText(
            self, "New Subfolder", f"Folder name in '{parent_rel_path}':",
            QLineEdit.EchoMode.Normal, "new_folder"
        )
        if ok and folder_name.strip():
            _, err = self.create_dir(parent_rel_path, folder_name.strip())
            if err:
                QMessageBox.warning(self, "Error Creating Folder", err)

    def prompt_rename_dir(self, rel_path: str):
        """Prompt to rename a directory."""
        old_dir = self.root_dir / rel_path
        current_name = old_dir.name
        new_name, ok = QInputDialog.getText(
            self, "Rename Folder", f"New name for folder '{current_name}':",
            QLineEdit.EchoMode.Normal, current_name
        )
        if ok and new_name.strip():
            _, err = self.rename_dir(rel_path, new_name.strip())
            if err:
                QMessageBox.warning(self, "Rename Failed", err)

    def prompt_delete(self, rel_path):
        blocked = deletion_blocked_reason(self.root_dir / rel_path)
        if blocked:
            QMessageBox.warning(self, "Cannot delete", blocked)
            return
        answer = QMessageBox.question(
            self, "Delete config",
            f"Delete {rel_path}?\n\nThis file is tracked in git; deleting it here "
            f"removes it from your working tree.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed, error = self.delete_config(rel_path)
        if not removed:
            QMessageBox.critical(self, "Delete failed", error or "Unknown error")

    def _get_installer(self):
        try:
            from frontend.hub.installer import HubInstaller
            workspace_dir = self.root_dir.parent.parent
            data_dir = workspace_dir / ".thetaide"
            return HubInstaller(workspace_dir=workspace_dir, data_dir=data_dir)
        except Exception:
            return None

    def prompt_delete_dir(self, rel_path):
        full_path = self.root_dir / rel_path
        if not full_path.exists():
            QMessageBox.warning(self, "Cannot delete", f"Folder '{rel_path}' does not exist.")
            return

        installer = self._get_installer()
        comp_meta = installer.find_component_for_path(full_path) if installer else None

        if comp_meta:
            comp_name = comp_meta.get("name") or comp_meta.get("id")
            dlg = ComponentDeleteDialog(rel_path, comp_name, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted or dlg.choice == "cancel":
                return
            if dlg.choice == "uninstall":
                installer.uninstall_by_metadata(comp_meta, remove_configs=True)
                self.dir_deleted.emit(rel_path)
                self.populate()
                return
            elif dlg.choice == "delete_only":
                removed, err = self.delete_dir(rel_path)
                if not removed:
                    QMessageBox.critical(self, "Delete failed", err)
                return

        blocked = deletion_blocked_reason(full_path)
        if blocked:
            QMessageBox.warning(self, "Cannot delete", blocked)
            return

        files_count = len(list(full_path.rglob("*")))
        answer = QMessageBox.question(
            self,
            "Delete folder",
            f"Delete folder '{rel_path}' and all contents ({files_count} items)?\n\n"
            f"This removes the folder from your working tree.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed, err = self.delete_dir(rel_path)
        if not removed:
            QMessageBox.critical(self, "Delete failed", err)

    def prompt_rename(self, rel_path):
        current = Path(rel_path).stem
        name, ok = QInputDialog.getText(
            self, "Rename", f"New name for '{current}':", QLineEdit.EchoMode.Normal, current
        )
        if ok:
            _, error = self.rename_config(rel_path, name)
            if error:
                QMessageBox.warning(self, "Rename failed", error)

    def _show_context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QMenu(self)
        if data["type"] == "file":
            action_select = menu.addAction("Load in Config Viewer")
            action_select.triggered.connect(lambda: self._on_item_clicked(item, 0))
            action_new_tab = menu.addAction("Open in New Tab")
            action_new_tab.triggered.connect(lambda: self._on_item_double_clicked(item, 0))
            label_dup = "Duplicate Experiment…" if self.mode in ("experiments", "experiment") else "Duplicate Config…"
            action_dup = menu.addAction(label_dup)
            action_dup.triggered.connect(self.prompt_duplicate)
            rel = data["rel_path"]
            action_rename = menu.addAction("Rename…")
            action_rename.triggered.connect(lambda: self.prompt_rename(rel))
            action_delete = menu.addAction("Delete…")
            action_delete.triggered.connect(lambda: self.prompt_delete(rel))
        elif data["type"] == "dir":
            rel = data["rel_path"]
            dir_path = Path(data["path"])
            is_exp = self.mode in ("experiments", "experiment") or rel.startswith("experiment")
            is_top_group = is_exp and (dir_path.parent == self.root_dir or dir_path.parent.name in ("experiment", "experiments"))
            if is_top_group:
                group_name = dir_path.name
                action_new = menu.addAction("New Experiment in this group…")
                action_new.triggered.connect(lambda: self._create_in_specific_group(group_name))
                action_new_folder = menu.addAction("New Subfolder…")
                action_new_folder.triggered.connect(lambda: self.prompt_new_folder(rel))
                action_export_plugin = menu.addAction("Export Experiment Plugin…")
                action_export_plugin.triggered.connect(lambda: self.prompt_export_experiment_group(group_name))
                action_refresh = menu.addAction("Refresh Group")
                action_refresh.triggered.connect(self.populate)
                action_delete_dir = menu.addAction("Delete Group…")
                action_delete_dir.triggered.connect(lambda: self.prompt_delete_dir(rel))
            elif is_exp:
                action_new = menu.addAction("New Experiment…")
                action_new.triggered.connect(lambda: self.prompt_new_in_folder(rel))
                action_new_folder = menu.addAction("New Subfolder…")
                action_new_folder.triggered.connect(lambda: self.prompt_new_folder(rel))
                action_rename_dir = menu.addAction("Rename Folder…")
                action_rename_dir.triggered.connect(lambda: self.prompt_rename_dir(rel))
                action_delete_dir = menu.addAction("Delete Folder…")
                action_delete_dir.triggered.connect(lambda: self.prompt_delete_dir(rel))
            else:
                if rel == "system":
                    action_new_folder = menu.addAction("New Subfolder…")
                    action_new_folder.triggered.connect(lambda: self.prompt_new_folder(rel))
                else:
                    action_new = menu.addAction("New Component in this category…")
                    cat_name = Path(data["rel_path"]).name
                    action_new.triggered.connect(lambda: self._create_in_specific_component_category(cat_name))
                    action_new_folder = menu.addAction("New Subfolder…")
                    action_new_folder.triggered.connect(lambda: self.prompt_new_folder(rel))
                    if not data.get("is_group"):
                        action_rename_dir = menu.addAction("Rename Folder…")
                        action_rename_dir.triggered.connect(lambda: self.prompt_rename_dir(rel))
                        action_delete_dir = menu.addAction("Delete Folder…")
                        action_delete_dir.triggered.connect(lambda: self.prompt_delete_dir(rel))

        menu.addSeparator()
        action_collapse = menu.addAction("Collapse All")
        action_collapse.triggered.connect(self.collapse_all)
        action_expand = menu.addAction("Expand All")
        action_expand.triggered.connect(self.expand_all)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _create_in_specific_group(self, group_name):
        self.prompt_new_in_group(group=group_name)

    def prompt_export_experiment_group(self, group_name: str):
        """Prompt to export an entire experiment group as an installable Experiment Plugin."""
        try:
            from .export_dialog import ExportExperimentDialog
            dlg = ExportExperimentDialog(
                experiment_id=group_name,
                workspace_root=self.root_dir.parent.parent,
                group=group_name,
                parent=self,
            )
            dlg.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not launch export dialog:\n{exc}")

    def prompt_new_component(self):
        """Prompt to create a new component configuration (agent, env, model, etc.)."""
        categories = ["agent", "env", "model", "data", "eval", "plot", "paradigms", "site", "hydra", "workflow"]
        existing = [p.name for p in self.root_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in ("experiment", "experiments")]
        for e in sorted(existing):
            if e not in categories:
                categories.append(e)

        cat, ok = QInputDialog.getItem(
            self, "New Component", "Select or type component category:",
            categories, 0, True
        )
        if not ok or not cat.strip():
            return
        cat = cat.strip()

        name, ok2 = QInputDialog.getText(
            self, "New Component Name", f"Component name in '{cat}':",
            QLineEdit.EchoMode.Normal, "custom"
        )
        if not ok2 or not name.strip():
            return

        name = name.strip()
        filename = f"{name}.yaml" if not name.endswith(".yaml") else name
        target_dir = self.root_dir / cat
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename

        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Component file already exists:\n{target_file}")
            return

        content = (
            f"# @package {cat}\n"
            f"# Modular component configuration: {cat}/{filename}\n\n"
            f"name: {name.replace('.yaml', '')}\n"
        )
        try:
            target_file.write_text(content, encoding="utf-8")
            self.populate()
            self.select_file(f"{cat}/{filename}")
        except OSError as exc:
            QMessageBox.critical(self, "Error Creating Component", f"Could not create file:\n{exc}")

    def _create_in_specific_component_category(self, cat_name):
        name, ok = QInputDialog.getText(
            self, "New Component", f"New component name in '{cat_name}':",
            QLineEdit.EchoMode.Normal, "custom"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        filename = f"{name}.yaml" if not name.endswith(".yaml") else name
        target_dir = self.root_dir / cat_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Component file already exists:\n{target_file}")
            return

        content = (
            f"# @package {cat_name}\n"
            f"# Modular component configuration: {cat_name}/{filename}\n\n"
            f"name: {name.replace('.yaml', '')}\n"
        )
        try:
            target_file.write_text(content, encoding="utf-8")
            self.populate()
            self.select_file(f"{cat_name}/{filename}")
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def prompt_scaffold_component(self):
        """Open the component scaffolding wizard dialog to create a new component plugin."""
        window = self.window()
        if hasattr(window, "scaffold_plugin"):
            window.scaffold_plugin()
        else:
            from .scaffold_dialog import ScaffoldPluginDialog
            workspace_root = find_config_root().parent
            dialog = ScaffoldPluginDialog(workspace_root=workspace_root, parent=self)
            if dialog.exec() and dialog.scaffold_result and dialog.scaffold_result.success:
                from .config_viewer import invalidate_config_tree
                invalidate_config_tree()
                self.populate()

    def refresh(self):
        """Alias for populate() to ensure compatibility."""
        self.populate()
