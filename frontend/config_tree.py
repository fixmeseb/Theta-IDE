"""Tree widget mirroring in/config/ directory for Theta-IDE."""
import shutil
from pathlib import Path

import yaml
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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
)

from .config_model import (
    ConfigTree,
    default_method_model,
    deletion_blocked_reason,
    experiment_yaml,
    group_base_yaml,
    rename_experiment_text,
)
from .sidetabs import svg_icon
from .widgets import label

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

TREE_GEAR_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "secondary",
    (QIcon.Mode.Active, QIcon.State.Off): "accent",
    (QIcon.Mode.Selected, QIcon.State.Off): "text",
    (QIcon.Mode.Disabled, QIcon.State.Off): "disabled",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
    (QIcon.Mode.Selected, QIcon.State.On): "accent",
}


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


class ConfigTreeWidget(QWidget):
    """File tree that mirrors in/config/ and allows selecting, duplicating,

    and creating experiments in groups.
    """
    file_selected = pyqtSignal(object, str)  # (Path, rel_path)
    duplicate_requested = pyqtSignal(str)     # (rel_path)
    new_in_group_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, root_dir=None, parent=None, mode="all"):
        super().__init__(parent)
        self.root_dir = Path(root_dir) if root_dir else find_config_root()
        self.mode = mode  # "all", "experiments", "components"
        self.current_rel_path = None
        self._dirty_paths = set()
        self._init_ui()
        self.populate()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(4)

        self.btn_refresh = self._tool_button("reload")
        self.btn_refresh.setToolTip("Reload configuration files from disk")
        self.btn_refresh.clicked.connect(self.populate)
        header.addWidget(self.btn_refresh)

        self.btn_collapse = self._tool_button("collapse_all")
        self.btn_collapse.setToolTip("Collapse all folders in the tree")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_collapse_all = self.btn_collapse  # alias
        header.addWidget(self.btn_collapse)

        self.btn_expand = self._tool_button("expand_all")
        self.btn_expand.setToolTip("Expand all folders in the tree")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_expand_all = self.btn_expand  # alias
        header.addWidget(self.btn_expand)

        self.btn_search = self._tool_button("search")
        self.btn_search.setToolTip("Filter configurations")
        self.btn_search.setCheckable(True)
        self.btn_search.setChecked(False)
        self.btn_search.clicked.connect(self.toggle_search)
        header.addWidget(self.btn_search)

        header.addStretch()

        self.btn_new = self._tool_button("new_file", "New")
        if self.mode == "components":
            self.btn_new.setToolTip("Create a new modular component (agent, env, model, etc.)")
            self.btn_new.clicked.connect(self.prompt_new_component)
        else:
            self.btn_new.setToolTip("Create a new default experiment in a group")
            self.btn_new.clicked.connect(self.prompt_new_in_group)
        self.btn_new_group = self.btn_new  # backwards compatibility alias
        header.addWidget(self.btn_new)

        self.btn_duplicate = self._tool_button("copy", "Copy")
        if self.mode in ("experiments", "experiment"):
            self.btn_duplicate.setToolTip("Duplicate currently selected experiment")
        elif self.mode == "components":
            self.btn_duplicate.setToolTip("Duplicate currently selected component")
        else:
            self.btn_duplicate.setToolTip("Duplicate currently selected configuration")
        self.btn_duplicate.clicked.connect(self.prompt_duplicate)
        header.addWidget(self.btn_duplicate)

        self.btn_save = None

        if self.mode == "components":
            self.btn_hub = self._tool_button("hub", "Component Hub")
            self.btn_hub.setCheckable(True)
            self.btn_hub.setChecked(False)
            self.btn_hub.setToolTip("Browse and install new RL methods, models, and environments from the Community Hub")
            header.addWidget(self.btn_hub)

            self.btn_toggle_raw = self._tool_button("raw_yaml", "View Raw YAML")
            self.btn_toggle_raw.setCheckable(True)
            self.btn_toggle_raw.setChecked(False)
            self.btn_toggle_raw.setToolTip("Toggle side-by-side view of the raw YAML file content")
            header.addWidget(self.btn_toggle_raw)

            self.btn_launch = None
            self.btn_start = None
            self.start_button = None
            self.btn_export = None
            self.btn_toggle_yaml = None
        elif self.mode in ("experiments", "experiment"):
            self.btn_hub = None
            self.btn_toggle_raw = None

            self.btn_launch = self._tool_button("launch", "Launch training")
            self.btn_launch.setToolTip("Train the loaded experiment config through the backend (F5)")
            self.btn_launch.setEnabled(False)
            self.btn_start = self.btn_launch
            self.start_button = self.btn_launch
            header.addWidget(self.btn_launch)

            self.btn_export = self._tool_button("export", "Export recipe YAML…")
            self.btn_export.setToolTip("Export recipe YAML…")
            header.addWidget(self.btn_export)

            self.btn_toggle_yaml = self._tool_button("raw_yaml", "View Hydra YAML")
            self.btn_toggle_yaml.setCheckable(True)
            self.btn_toggle_yaml.setChecked(False)
            self.btn_toggle_yaml.setToolTip("Toggle preview of the resolved Hydra YAML configuration")
            header.addWidget(self.btn_toggle_yaml)
        else:
            self.btn_hub = None
            self.btn_toggle_raw = None
            self.btn_launch = None
            self.btn_start = None
            self.start_button = None
            self.btn_export = None
            self.btn_toggle_yaml = None

        self.header_layout = header
        layout.addLayout(header)

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
        self.search.hide()
        self.search.installEventFilter(self)
        layout.addWidget(self.search)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.itemClicked.connect(self._on_item_clicked)
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
                for sub in subdirs:
                    self._add_dir_node(self.tree, sub, expand=True, expanded_set=expanded_paths, has_previous_state=has_previous_state)
                for f in files:
                    self._add_file_node(self.tree, f)
        elif self.mode == "components":
            # Display all modular configuration directories and files OUTSIDE experiment
            preferred_comp_order = ["agent", "env", "model", "paradigms", "site", "hydra"]
            for cat in preferred_comp_order:
                if cat in existing_dirs:
                    self._add_dir_node(self.tree, existing_dirs[cat], expand=False, expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for name, dir_path in sorted(existing_dirs.items()):
                if name not in preferred_comp_order and name not in ("experiment", "experiments"):
                    self._add_dir_node(self.tree, dir_path, expand=False, expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for f in top_files:
                self._add_file_node(self.tree, f)
        else:
            # "all": Top-level directory ordering with experiment first
            preferred_order = ["experiment", "agent", "env", "model", "paradigms", "site", "hydra"]
            for cat in preferred_order:
                if cat in existing_dirs:
                    self._add_dir_node(self.tree, existing_dirs[cat], expand=(cat == "experiment"), expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for name, dir_path in sorted(existing_dirs.items()):
                if name not in preferred_order:
                    self._add_dir_node(self.tree, dir_path, expand=False, expanded_set=expanded_paths, has_previous_state=has_previous_state)
            for f in top_files:
                self._add_file_node(self.tree, f)

        # Re-apply current selection if possible without triggering external selection signals
        if self.current_rel_path:
            self.select_file(self.current_rel_path, emit_signal=False)

    def _tool_button(self, icon_name, text=None):
        """Square header button with a themed SVG icon; text, if any, becomes its accessible name."""
        button = QToolButton()
        button.setProperty("svg_icon", icon_name)
        button.setIcon(svg_icon(icon_name, TOOL_ICON_COLORS))
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(30, 30)
        button.setStyleSheet("padding: 0;")
        if text:
            button.setAccessibleName(text)
        return button

    def refresh_icons(self):
        """Re-render the header and tree icons in the current theme's colors."""
        for button in self.findChildren(QToolButton):
            name = button.property("svg_icon")
            if name:
                button.setIcon(svg_icon(name, TOOL_ICON_COLORS))

        def _update_tree(parent):
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                child = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("is_base"):
                    child.setIcon(0, svg_icon("gear", TREE_GEAR_COLORS))
                _update_tree(child)

        _update_tree(self.tree)

    def _add_dir_node(self, parent_widget, dir_path: Path, expand=False, expanded_set=None, has_previous_state=False):
        name = dir_path.name
        node = QTreeWidgetItem(parent_widget, [name])
        node.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        rel_path = str(dir_path.relative_to(self.root_dir)).replace("\\", "/").strip("/")
        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "dir",
            "path": str(dir_path),
            "rel_path": rel_path,
        })
        node.setToolTip(0, rel_path)

        # Subdirectories first, then files
        subdirs = sorted([p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")])
        files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])

        for sub in subdirs:
            # Under experiment, expand group folders like cartpole, mimic
            sub_expand = (name == "experiment")
            self._add_dir_node(node, sub, expand=sub_expand, expanded_set=expanded_set, has_previous_state=has_previous_state)

        for f in files:
            self._add_file_node(node, f)

        if has_previous_state:
            should_expand = (expanded_set is not None and rel_path in expanded_set)
        else:
            should_expand = (expanded_set is not None and rel_path in expanded_set) or expand

        if should_expand:
            node.setExpanded(True)

        return node

    def _add_file_node(self, parent_node, file_path: Path):
        name = file_path.name
        rel_path = file_path.relative_to(self.root_dir).as_posix()
        is_exp = rel_path.startswith("experiment/") and not name.startswith("_")
        is_base = file_path.stem == "_base"

        display_name = name
        if display_name.endswith(".yaml"):
            display_name = display_name[:-5]
        elif display_name.endswith(".yml"):
            display_name = display_name[:-4]

        if is_base:
            display_name = "group defaults"

        node = QTreeWidgetItem(parent_node, [display_name])
        if is_base:
            node.setIcon(0, svg_icon("gear", TREE_GEAR_COLORS))
        elif file_path.suffix not in (".yaml", ".yml"):
            node.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "file",
            "path": str(file_path),
            "rel_path": rel_path,
            "is_experiment": is_exp,
            "is_base": is_base,
            "base_name": display_name,
        })
        node.setData(0, Qt.ItemDataRole.ForegroundRole, None)
        node.setToolTip(0, rel_path)
        return node

    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data["type"] == "file":
            self.current_rel_path = data["rel_path"]
            self.file_selected.emit(Path(data["path"]), data["rel_path"])

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

    def toggle_search(self, checked: bool | None = None):
        """Toggle the visibility of the search/filter bar."""
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
        if obj is self.search and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self.toggle_search(False)
                return True
        return super().eventFilter(obj, event)

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

    def _write_experiment(self, group_name, filename, exp_id, paradigm=None, env=None):
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
                model = default_method_model(tree.group(group_name), tree.models)

        target_dir = self.root_dir / "experiment" / group_name
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
        return f"experiment/{group_name}/{filename}"

    def get_available_groups(self):
        """Return sorted list of experiment groups in in/config/experiment/."""
        exp_dir = self.root_dir if self.root_dir.name in ("experiment", "experiments") else (self.root_dir / "experiment")
        if not exp_dir.exists():
            return []
        groups = [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        return sorted(groups)

    def prompt_new_in_group(self, group=None):
        """Create a new experiment, in an existing group or a brand new one."""
        groups = self.get_available_groups() or ["cartpole", "mimic", "thetaide"]
        dialog = NewExperimentDialog(self._model(), groups, group=group, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        group_name, name, paradigm, env = dialog.values()
        if not group_name or not name:
            return

        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        rel_path = self._write_experiment(group_name, filename, filename[:-5], paradigm, env)
        if rel_path:
            self.populate()
            self.select_file(rel_path)

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
        new_rel = str(Path(rel_path).parent / new_name)
        self.populate()
        self.select_file(new_rel)
        return new_rel, None

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
                self.populate()
                return
            elif dlg.choice == "delete_only":
                try:
                    shutil.rmtree(full_path)
                except OSError as exc:
                    QMessageBox.critical(self, "Delete failed", str(exc))
                self.populate()
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
        try:
            shutil.rmtree(full_path)
        except OSError as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))
        self.populate()

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
            if rel.startswith("experiment") or self.mode in ("experiments", "experiment"):
                action_new = menu.addAction("New Experiment in this group…")
                group_name = Path(data["rel_path"]).name
                action_new.triggered.connect(lambda: self._create_in_specific_group(group_name))
            else:
                action_new = menu.addAction("New Component in this category…")
                cat_name = Path(data["rel_path"]).name
                action_new.triggered.connect(lambda: self._create_in_specific_component_category(cat_name))

            action_delete_dir = menu.addAction("Delete Folder…")
            action_delete_dir.triggered.connect(lambda: self.prompt_delete_dir(rel))

        menu.addSeparator()
        action_collapse = menu.addAction("Collapse All")
        action_collapse.triggered.connect(self.collapse_all)
        action_expand = menu.addAction("Expand All")
        action_expand.triggered.connect(self.expand_all)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def collapse_all(self):
        """Collapse all folders/nodes in the tree."""
        self.tree.collapseAll()

    def expand_all(self):
        """Expand all folders/nodes in the tree."""
        self.tree.expandAll()

    def _create_in_specific_group(self, group_name):
        self.prompt_new_in_group(group=group_name)

    def prompt_new_component(self):
        """Prompt to create a new component configuration (agent, env, model, etc.)."""
        categories = ["agent", "env", "model", "paradigms", "site", "hydra"]
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
