"""Full-screen, simplified, boxed configuration viewer for Theta-IDE."""
import math
import random
from pathlib import Path

import yaml
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_model import ConfigTree, default_method_model
from .config_model import add_method as add_method_to
from .config_model import remove_method as remove_method_from
from .widgets import ComboBox, DoubleSpinBox, SpinBox, label, SmoothScrollArea

# Forms stay readable on wide windows: the column of boxes stops growing past CONTENT_WIDTH,
# and fields in label/field rows are capped by kind so a number doesn't get a 1,000px box.
CONTENT_WIDTH = 860
NUMBER_FIELD_WIDTH = 200
TEXT_FIELD_WIDTH = 380


def _parse_yaml_value(text: str):
    """Convert text representation to typed YAML scalar."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() in ("null", "none", "~", ""):
        return None
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def discover_datasets(env_name: str | None = None) -> list[str]:
    """Discover available dataset files and folders in in/datasets and results/datasets."""
    found = []
    base_dirs = [Path("in/datasets"), Path("results/datasets")]
    for b in base_dirs:
        if b.is_dir():
            for p in b.glob("**/*"):
                if p.name.startswith(".") or p.name.startswith("__"):
                    continue
                if p.is_file() and p.suffix in (".pkl", ".csv", ".parquet", ".h5", ".pt"):
                    found.append(p.as_posix())
                elif p.is_dir() and any(p.iterdir()):
                    found.append(p.as_posix())
    return sorted(list(set(found)))


class CompactDoubleSpinBox(DoubleSpinBox):
    """Keeps six decimals of precision but shows 0.95 rather than 0.950000."""

    def textFromValue(self, value):
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text


def number_field(value):
    """Spin box for a YAML number: floats step at one tenth of their magnitude, and values
    that start non-negative (counts, coefficients) can't be pushed below zero."""
    if isinstance(value, float):
        spin = CompactDoubleSpinBox()
        spin.setDecimals(6)
        magnitude = math.floor(math.log10(abs(value))) if value else -1
        spin.setSingleStep(10.0 ** (magnitude - 1))
    else:
        spin = SpinBox()
    spin.setRange(0 if value >= 0 else -1000000, 100000000)
    spin.setValue(value)
    return spin

_CONFIG_TREE = None
_CONFIG_TREE_LOADED = False


def config_tree():
    """The parsed in/config tree, or None if it cannot be found.

    Callers must handle None: the viewer falls back to free-text entry so it
    still works when run outside a checkout.
    """
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    if not _CONFIG_TREE_LOADED:
        _CONFIG_TREE_LOADED = True
        try:
            _CONFIG_TREE = ConfigTree.discover()
        except (FileNotFoundError, OSError):
            _CONFIG_TREE = None
    return _CONFIG_TREE


def _leading_directives(text):
    """The comment block a config opens with, kept verbatim across a save.

    "# @package _global_" is a Hydra directive, not decoration: without it the
    file's keys are not applied at the global package level, so methods and the
    rest simply are not seen. Every one of the experiment configs starts with
    it, and yaml.safe_dump drops comments, so it has to be re-attached by hand.
    """
    kept = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        kept.append(line)
    return "".join(kept)


def _read_group_base(tree, group):
    """The group's _base.yaml as plain data, or {} when there is none."""
    if tree is None or not group:
        return {}
    path = tree.config_root / "experiment" / group / "_base.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


class ConfigBox(QFrame):
    """Themed card container representing a section of configuration."""
    def __init__(self, title=None, subtitle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        self.title_label = None
        self.subtitle_label = None

    def add_widget(self, widget):
        self.layout.addWidget(widget)

    def add_layout(self, sub_layout):
        self.layout.addLayout(sub_layout)


class TitleLabel(QLabel):
    """Clickable heading label that emits double_clicked on double-click."""
    double_clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("heading")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ConfigViewer(QWidget):
    """Modern, wide, boxed configuration viewer that displays experiment and

    component configurations in themed cards.
    """
    config_changed = pyqtSignal()
    save_requested = pyqtSignal()
    dirty_state_changed = pyqtSignal(str, bool)
    dirty_changed = pyqtSignal(bool)
    file_renamed = pyqtSignal(str, str)

    INHERIT = "(inherit from base)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_path = None
        self.current_rel_path = None
        self.raw_data = {}
        self.is_dirty = False
        self._block_updates = False
        self._is_editing_title = False
        self._committing_rename = False
        self.field_widgets = {}
        self._preamble = ""
        self.auto_save = False
        self.view_mode = "overrides"

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # Header Info Bar
        self.header_bar = QWidget()
        hb_layout = QHBoxLayout(self.header_bar)
        hb_layout.setContentsMargins(12, 6, 12, 6)
        hb_layout.setSpacing(10)

        self.group_badge = QLabel("")
        self.group_badge.setObjectName("breadcrumbGroup")
        self.group_badge.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #888888; padding: 2px 2px;"
        )
        self.group_badge.hide()
        hb_layout.addWidget(self.group_badge)

        self.breadcrumb_sep = QLabel("/")
        self.breadcrumb_sep.setObjectName("breadcrumbSep")
        self.breadcrumb_sep.setStyleSheet(
            "font-size: 16px; font-weight: 500; color: #666666; padding: 0 4px;"
        )
        self.breadcrumb_sep.hide()
        hb_layout.addWidget(self.breadcrumb_sep)

        self.file_title = TitleLabel("No config loaded")
        self.file_title.double_clicked.connect(self._start_title_edit)
        hb_layout.addWidget(self.file_title)

        self.title_editor = QLineEdit()
        self.title_editor.setObjectName("titleEditor")
        self.title_editor.setStyleSheet(
            "font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 4px;"
        )
        self.title_editor.setMinimumWidth(200)
        self.title_editor.setMaximumWidth(400)
        self.title_editor.hide()
        self.title_editor.returnPressed.connect(self._commit_title_edit)
        self.title_editor.installEventFilter(self)
        hb_layout.addWidget(self.title_editor)

        self.paradigm_badge = label("", "badge")
        self.paradigm_badge.hide()

        self.dirty_status = label("", "muted")
        self.dirty_status.hide()

        hb_layout.addStretch()

        self.chk_view_mode = QCheckBox("Show Resolved Defaults")
        self.chk_view_mode.setObjectName("chkViewMode")
        self.chk_view_mode.setStyleSheet("font-size: 12px; color: #a89984;")
        self.chk_view_mode.setToolTip("Toggle between showing only explicit overrides vs. all resolved group defaults")
        self.chk_view_mode.setChecked(False)
        self.chk_view_mode.toggled.connect(self._on_view_mode_toggled)
        self.chk_view_mode.hide()
        hb_layout.addWidget(self.chk_view_mode)

        self.btn_save = None

        root_layout.addWidget(self.header_bar)

        # Main Scroll Area holding the boxes
        self.scroll = SmoothScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("configScrollArea")
        self.scroll.verticalScrollBar().setSingleStep(16)

        self.container = QWidget()
        self.container.setMaximumWidth(CONTENT_WIDTH)
        self.boxes_layout = QVBoxLayout(self.container)
        self.boxes_layout.setContentsMargins(12, 10, 12, 16)
        self.boxes_layout.setSpacing(14)

        self.scroll.setWidget(self.container)
        root_layout.addWidget(self.scroll, 1)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "title_editor", None):
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_title_edit()
                    return True
            elif event.type() == QEvent.Type.FocusOut:
                if getattr(self, "_is_editing_title", False) and not getattr(self, "_committing_rename", False):
                    current_stem = self.current_path.stem if self.current_path else ""
                    raw_text = self.title_editor.text().strip()
                    if raw_text and raw_text != current_stem:
                        self._commit_title_edit()
                    else:
                        self._cancel_title_edit()
                return False
        return super().eventFilter(obj, event)

    def _start_title_edit(self, event=None):
        if not self.current_path or not self.current_rel_path:
            return
        self._is_editing_title = True
        self.file_title.hide()
        current_stem = self.current_path.stem
        self.title_editor.setText(current_stem)
        self.title_editor.show()
        self.title_editor.setFocus()
        self.title_editor.selectAll()

    def _cancel_title_edit(self):
        self._is_editing_title = False
        self.title_editor.hide()
        self.file_title.show()

    def _commit_title_edit(self):
        if not getattr(self, "_is_editing_title", False) or getattr(self, "_committing_rename", False):
            return
        self._committing_rename = True
        try:
            raw_text = self.title_editor.text().strip()
            new_title = raw_text
            if new_title.endswith(".yaml"):
                new_title = new_title[:-5].strip()
            elif new_title.endswith(".yml"):
                new_title = new_title[:-4].strip()

            old_title = self.current_path.stem if self.current_path else ""
            if not new_title or new_title == old_title:
                self._cancel_title_edit()
                return

            invalid_chars = set('/\\:*?"<>|')
            if any(c in invalid_chars for c in new_title):
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "The config name cannot contain invalid characters: / \\ : * ? \" < > |",
                )
                self._cancel_title_edit()
                return

            old_path = self.current_path
            old_rel = self.current_rel_path
            new_filename = f"{new_title}{old_path.suffix or '.yaml'}"
            new_path = old_path.with_name(new_filename)

            is_same_file = False
            if old_path.exists() and new_path.exists():
                try:
                    is_same_file = old_path.samefile(new_path)
                except Exception:
                    is_same_file = (old_path.name.lower() == new_path.name.lower())

            if new_path.exists() and not is_same_file:
                QMessageBox.warning(
                    self,
                    "File Already Exists",
                    f"A configuration file named '{new_filename}' already exists in this folder.",
                )
                self._cancel_title_edit()
                return

            is_experiment = (
                (old_rel and old_rel.startswith("experiment/"))
                or "methods" in self.raw_data
                or any("_base" in str(d) for d in self.raw_data.get("defaults", []))
            )
            for key in ("algorithm", "name", "architecture", "reasoner", "id"):
                if key in self.raw_data:
                    self.raw_data[key] = new_title

            if is_experiment:
                self.raw_data.pop("experiment_id", None)
                self.raw_data.pop("group", None)
            elif old_rel and old_rel.startswith("agent/"):
                self.raw_data["algorithm"] = new_title
            elif old_rel and old_rel.startswith("env/"):
                self.raw_data["name"] = new_title
            elif old_rel and old_rel.startswith("model/"):
                if "reasoner" in self.raw_data:
                    self.raw_data["reasoner"] = new_title
                else:
                    self.raw_data["architecture"] = new_title

            # Recursive string replacement for any exact matches of old_title
            def _replace_matching(d):
                if isinstance(d, dict):
                    for k, v in list(d.items()):
                        if isinstance(v, str) and v.lower() == old_title.lower():
                            d[k] = new_title
                        elif isinstance(v, (dict, list)):
                            _replace_matching(v)
                elif isinstance(d, list):
                    for idx, item in enumerate(d):
                        if isinstance(item, str) and item.lower() == old_title.lower():
                            d[idx] = new_title
                        elif isinstance(item, (dict, list)):
                            _replace_matching(item)

            _replace_matching(self.raw_data)

            # Check if methods dict has a method named after old_title
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                methods = self.raw_data["methods"]
                if old_title in methods:
                    methods[new_title] = methods.pop(old_title)

            # Rename file on disk if it exists
            if old_path.exists():
                if old_path.name.lower() == new_path.name.lower() and old_path.name != new_path.name:
                    # Case-only rename: use intermediate temporary file to guarantee APFS/FAT directory entry update
                    temp_path = old_path.with_name(f"{old_path.stem}__theta_tmp_rename{old_path.suffix}")
                    old_path.rename(temp_path)
                    temp_path.rename(new_path)
                else:
                    old_path.rename(new_path)

            self.current_path = new_path
            new_rel = str(Path(old_rel).with_name(new_filename)).replace("\\", "/")
            self.current_rel_path = new_rel

            # Save updated content to disk
            yaml_str = yaml.safe_dump(self.raw_data, sort_keys=False)
            self.current_path.write_text(self._preamble + yaml_str, encoding="utf-8")
            self.is_dirty = False
            self._update_dirty_ui()

            # End edit mode
            self._is_editing_title = False
            self.title_editor.hide()
            self.file_title.setText(new_title)
            self.file_title.setToolTip(f"{new_rel}\n(Double-click to rename)")
            self.file_title.show()

            self._render_boxes()

            self.file_renamed.emit(old_rel, new_rel)
            self.dirty_state_changed.emit(old_rel, False)
            self.dirty_state_changed.emit(new_rel, False)
            self.config_changed.emit()
            self.save_requested.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Rename Failed", f"Could not rename file:\n{exc}")
            self._cancel_title_edit()
        finally:
            self._committing_rename = False

    @property
    def txt_exp_id(self):
        class _TitleProxy:
            def __init__(proxy_self, viewer):
                proxy_self._viewer = viewer
            def text(proxy_self):
                return proxy_self._viewer.file_title.text()
            def setText(proxy_self, val):
                proxy_self._viewer.file_title.setText(val)
        return _TitleProxy(self)

    def load_file(self, file_path: Path, rel_path: str):
        """Parse YAML file and render visual boxes."""
        self.current_path = Path(file_path)
        self.current_rel_path = rel_path
        self.is_dirty = False

        try:
            content = self.current_path.read_text(encoding="utf-8")
            self._preamble = _leading_directives(content)
            data = yaml.safe_load(content) or {}
            self.raw_data = data if isinstance(data, dict) else {"content": data}
            self._clean_redundant_params()
        except Exception as exc:
            self.file_title.setText(f"Error loading {self.current_path.stem}")
            return

        self._render_boxes()
        self._update_dirty_ui()

    def _render_boxes(self):
        """Clear and build the configuration boxes."""
        self._block_updates = True
        self.field_widgets.clear()

        # Clear existing layout
        while self.boxes_layout.count():
            item = self.boxes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        # Update header
        group = self._group_name()
        is_base = (self.current_path.stem == "_base") if self.current_path else False

        if is_base:
            self.group_badge.setText(f"[{group}]" if group else "[group]")
            self.group_badge.show()
            self.breadcrumb_sep.hide()
            self.file_title.setText("Group Defaults")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Base configuration for {group})")
        elif group:
            rel_parts = Path(str(self.current_rel_path or "")).parts
            if len(rel_parts) > 3:
                sub_crumbs = " / ".join(rel_parts[2:-1])
                self.group_badge.setText(f"[{group}] / {sub_crumbs}")
            else:
                self.group_badge.setText(f"[{group}]")
            self.group_badge.show()
            self.breadcrumb_sep.show()
            self.file_title.setText(self.current_path.stem)
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Double-click to rename)")
        else:
            self.group_badge.hide()
            self.breadcrumb_sep.hide()
            self.file_title.setText(self.current_path.stem if self.current_path else "")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Double-click to rename)")

        is_experiment = (
            (self.current_rel_path and self.current_rel_path.startswith("experiment/") and "/methods/" not in self.current_rel_path) or
            "methods" in self.raw_data or
            any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )

        if is_experiment:
            self.chk_view_mode.show()
            self._render_experiment_boxes()
        else:
            self.chk_view_mode.hide()
            self._render_generic_boxes()

        self._cap_form_fields()
        self.boxes_layout.addStretch()
        self._block_updates = False

    def _on_view_mode_toggled(self, checked: bool):
        self.view_mode = "resolved" if checked else "overrides"
        self._render_boxes()

    def _get_resolved_defaults(self) -> dict:
        tree = config_tree()
        group = self._group_name()
        resolved = {}
        if tree:
            root_cfg = getattr(tree, "config_root", None)
            if root_cfg and (root_cfg / "config.yaml").is_file():
                try:
                    resolved.update(yaml.safe_load((root_cfg / "config.yaml").read_text(encoding="utf-8")) or {})
                except Exception:
                    pass
            if group:
                base_data = _read_group_base(tree, group)
                if base_data:
                    resolved.update(base_data)
        return resolved

    def _group_name(self):
        """The experiment group this file sits in, from experiment/<group>/<file>."""
        parts = Path(str(self.current_rel_path or "")).parts
        return parts[1] if len(parts) > 2 and parts[0] == "experiment" else None

    def env_selection(self):
        """The chosen environment, or None when it is inherited from the base."""
        combo = getattr(self, "combo_env", None)
        if combo is None or combo.currentData() == self.INHERIT:
            return None
        return combo.currentText()

    def _inherited_paradigm(self):
        """The paradigm this experiment inherits from its group's _base.yaml."""
        tree, group = config_tree(), self._group_name()
        if tree is None or group is None:
            return None
        return tree.group(group).paradigm

    def _inherited_env(self):
        """The environment this experiment inherits from its group's _base.yaml."""
        tree, group = config_tree(), self._group_name()
        if tree is None or group is None or group not in tree.groups:
            return None
        return tree.group(group).env

    def _clean_redundant_params(self):
        """Strip redundant parameters inherited from defaults or matching filename stem."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return
        is_experiment = (
            (self.current_rel_path and self.current_rel_path.startswith("experiment/"))
            or "methods" in self.raw_data
            or any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )
        if is_experiment:
            self.raw_data.pop("group", None)
            stem = self.current_path.stem if self.current_path else ""
            if self.raw_data.get("experiment_id") == stem:
                self.raw_data.pop("experiment_id", None)
            if self.raw_data.get("paradigm") == self._inherited_paradigm():
                self.raw_data.pop("paradigm", None)
            inh_env = self._inherited_env()
            if isinstance(self.raw_data.get("env"), str) and (
                self.raw_data["env"] == inh_env or self.raw_data["env"] == self.INHERIT
            ):
                self.raw_data.pop("env", None)
            if "description" in self.raw_data and not self.raw_data["description"]:
                self.raw_data.pop("description", None)
            if "resources" in self.raw_data and not self.raw_data["resources"]:
                self.raw_data.pop("resources", None)
            if "trainer" in self.raw_data and not self.raw_data["trainer"]:
                self.raw_data.pop("trainer", None)
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                if "params" in self.raw_data["methods"] and not self.raw_data["methods"]["params"]:
                    self.raw_data["methods"].pop("params", None)

    def _cap_form_fields(self):
        """Limit field widths in every label/field form and two-column grid."""
        for grid in self.container.findChildren(QGridLayout):
            for index in range(grid.count()):
                widget = grid.itemAt(index).widget()
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
                elif isinstance(widget, (QComboBox, QLineEdit)):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
            # An empty last column takes the slack, so fields stay next to their labels
            grid.setColumnStretch(grid.columnCount(), 1)
        for row_layout in self.container.findChildren(QHBoxLayout):
            for index in range(row_layout.count()):
                widget = row_layout.itemAt(index).widget()
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
        for form in self.container.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                widget = item.widget() if item else None
                label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if widget is not None and label_item and isinstance(label_item.widget(), QLabel):
                    # Labels sit at the top of their row; match the field height so the text lines up
                    label_item.widget().setMinimumHeight(widget.sizeHint().height())
                    label_item.widget().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
                elif isinstance(widget, (QComboBox, QLineEdit)):
                    widget.setMaximumWidth(TEXT_FIELD_WIDTH)

    def _render_experiment_boxes(self):
        """Render standard boxes for an Experiment configuration."""
        data = self.raw_data
        resolved = self._get_resolved_defaults()

        # Paradigm is inherited from group base by default.
        paradigm = data.get("paradigm") or self._inherited_paradigm() or "online_rl"
        self.paradigm_badge.hide()

        # Metadata / constraint helper widgets (not rendered into visual layout)
        group_val = data.get("group")
        if not group_val:
            parts = Path(self.current_rel_path).parts
            group_val = parts[1] if len(parts) > 2 else "ungrouped"
        self.txt_group = QLineEdit(str(group_val))
        self.txt_group.textChanged.connect(lambda v: self._on_field_edited("group", v))
        self.field_widgets["group"] = self.txt_group

        # Paradigm + Environment, driven by in/config/paradigms constraints.
        tree = config_tree()
        if tree and tree.paradigms:
            self.combo_paradigm = ComboBox()
            self.combo_paradigm.addItems(sorted(tree.paradigms))
            if paradigm in tree.paradigms:
                self.combo_paradigm.setCurrentText(paradigm)
            rules = tree.paradigms.get(self.combo_paradigm.currentText())
            if rules and rules.description:
                self.combo_paradigm.setToolTip(rules.description)
            self.combo_paradigm.currentTextChanged.connect(self._on_paradigm_changed)
            self.field_widgets["paradigm"] = self.combo_paradigm

            self.combo_env = ComboBox()
            inherited_env = tree.group(self._group_name() or "").env
            inherit_label = f"(inherit from base — {inherited_env})" if inherited_env else self.INHERIT
            self.combo_env.addItem(inherit_label, self.INHERIT)
            allowed_envs = [e.name for e in tree.environments_for(self.combo_paradigm.currentText())]
            self.combo_env.addItems(allowed_envs)
            current_env = data.get("env")
            if isinstance(current_env, str) and current_env in allowed_envs:
                self.combo_env.setCurrentText(current_env)
            self.combo_env.setToolTip(
                f"Environments compatible with {self.combo_paradigm.currentText()} "
                f"(offline_only must match)"
            )
            self.combo_env.currentTextChanged.connect(
                lambda v: self._on_field_edited("env", v) if self.env_selection() else None
            )
            self.field_widgets["env"] = self.combo_env

        # --- BOX 1: TRAINING BUDGET & ROLLOUT SCHEDULE ---
        box_budget = ConfigBox("1. Training Budget & Schedule", "Timesteps, evaluation frequency, and random seeds")

        # Experiment Description / Commit Note
        desc_row = QHBoxLayout()
        desc_row.setSpacing(10)
        desc_row.addWidget(label("Description", "eyebrow"))
        self.txt_description = QLineEdit()
        self.txt_description.setPlaceholderText("Optional commit-style notes or hypothesis for this experiment...")
        self.txt_description.setText(str(data.get("description") or ""))
        self.txt_description.textChanged.connect(
            lambda v: self._on_field_edited("description", v.strip()) if v.strip() else self.raw_data.pop("description", None)
        )
        desc_row.addWidget(self.txt_description, 1)
        self.field_widgets["description"] = self.txt_description
        box_budget.add_layout(desc_row)

        form_budget = QGridLayout()
        form_budget.setVerticalSpacing(10)
        form_budget.setHorizontalSpacing(16)

        # Total timesteps
        form_budget.addWidget(label("Total Timesteps"), 0, 0)
        self.spin_timesteps = SpinBox()
        self.spin_timesteps.setRange(100, 100_000_000)
        self.spin_timesteps.setSingleStep(1_000)
        self.spin_timesteps.setValue(int(data.get("total_timesteps") or resolved.get("total_timesteps") or 10000))
        self.spin_timesteps.valueChanged.connect(lambda v: self._on_field_edited("total_timesteps", v))
        form_budget.addWidget(self.spin_timesteps, 0, 1)
        self.field_widgets["total_timesteps"] = self.spin_timesteps

        # Seed + Randomize button
        form_budget.addWidget(label("Random Seed"), 0, 2)
        seed_row = QHBoxLayout()
        seed_row.setSpacing(6)
        self.spin_seed = SpinBox()
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(int(data.get("seed") if data.get("seed") is not None else resolved.get("seed", 42)))
        self.spin_seed.valueChanged.connect(lambda v: self._on_field_edited("seed", v))
        seed_row.addWidget(self.spin_seed, 1)
        self.field_widgets["seed"] = self.spin_seed

        btn_rand_seed = QToolButton()
        btn_rand_seed.setText("Random")
        btn_rand_seed.setToolTip("Pick a random seed")
        btn_rand_seed.clicked.connect(lambda: self.spin_seed.setValue(random.randint(1, 9999)))
        seed_row.addWidget(btn_rand_seed)
        seed_row.addStretch()
        form_budget.addLayout(seed_row, 0, 3)

        # Intervals count
        form_budget.addWidget(label("Intervals Count"), 1, 0)
        self.spin_intervals = SpinBox()
        self.spin_intervals.setRange(1, 100)
        self.spin_intervals.setValue(int(data.get("intervals_count") or resolved.get("intervals_count") or 4))
        self.spin_intervals.valueChanged.connect(lambda v: self._on_field_edited("intervals_count", v))
        form_budget.addWidget(self.spin_intervals, 1, 1)
        self.field_widgets["intervals_count"] = self.spin_intervals

        # Eval episodes
        form_budget.addWidget(label("Eval Episodes"), 1, 2)
        self.spin_eval_ep = SpinBox()
        self.spin_eval_ep.setRange(0, 1000)
        self.spin_eval_ep.setValue(int(data.get("eval_episodes") if data.get("eval_episodes") is not None else resolved.get("eval_episodes", 100)))
        self.spin_eval_ep.valueChanged.connect(lambda v: self._on_field_edited("eval_episodes", v))
        form_budget.addWidget(self.spin_eval_ep, 1, 3)
        self.field_widgets["eval_episodes"] = self.spin_eval_ep

        # Disable fields the paradigm forbids rather than letting the pipeline reject them
        if tree and paradigm in tree.paradigms:
            rules = tree.paradigms[paradigm]
            base = _read_group_base(tree, self._group_name())
            for key, widget in (("intervals_count", self.spin_intervals),
                                ("eval_episodes", self.spin_eval_ep)):
                if rules.field_enabled(key):
                    continue
                widget.setEnabled(False)
                if key in base:
                    widget.setValue(int(base[key]))
                    widget.setSuffix(f"   (fixed by {paradigm})")
                else:
                    widget.setSuffix(f"   (not used by {paradigm})")
                widget.setToolTip(rules.disabled_reason(key))

        # Checkboxes: Tensorboard, Save Dataset, Recover, No Plots
        chk_row = QHBoxLayout()
        chk_row.setSpacing(18)
        self.chk_tb = QCheckBox("Log to TensorBoard")
        self.chk_tb.setChecked(bool(data.get("tensorboard", resolved.get("tensorboard", True))))
        self.chk_tb.toggled.connect(lambda v: self._on_field_edited("tensorboard", v))
        chk_row.addWidget(self.chk_tb)

        self.chk_save_ds = QCheckBox("Save Dataset")
        self.chk_save_ds.setChecked(bool(data.get("save_dataset", resolved.get("save_dataset", False))))
        self.chk_save_ds.toggled.connect(lambda v: self._on_field_edited("save_dataset", v))
        chk_row.addWidget(self.chk_save_ds)

        self.chk_recover = QCheckBox("Recover Checkpoint")
        self.chk_recover.setChecked(bool(data.get("recover", resolved.get("recover", False))))
        self.chk_recover.toggled.connect(lambda v: self._on_field_edited("recover", v))
        chk_row.addWidget(self.chk_recover)

        self.chk_no_plot = QCheckBox("No Plots")
        self.chk_no_plot.setChecked(bool(data.get("no_plot", resolved.get("no_plot", False))))
        self.chk_no_plot.toggled.connect(lambda v: self._on_field_edited("no_plot", v))
        chk_row.addWidget(self.chk_no_plot)

        chk_row.addStretch()
        form_budget.addLayout(chk_row, 2, 0, 1, 4)

        box_budget.add_layout(form_budget)
        self.boxes_layout.addWidget(box_budget)

        # --- BOX 2: METHODS & AGENT ARCHITECTURES ---
        methods_dict = data.get("methods", {})
        box_methods = ConfigBox("2. Methods & Algorithms", "Configured agents, policies, and hyperparameter overrides")
        m_layout = QVBoxLayout()
        m_layout.setSpacing(10)

        # Universal Parameters (methods.params)
        has_params = "params" in methods_dict and isinstance(methods_dict["params"], dict)
        resolved_params = resolved.get("methods", {}).get("params", {}) if isinstance(resolved.get("methods"), dict) else {}
        show_params = has_params or (self.view_mode == "resolved" and bool(resolved_params))

        if show_params:
            p_card = QFrame()
            p_card.setObjectName("card")
            p_layout = QVBoxLayout(p_card)
            p_layout.setContentsMargins(14, 12, 14, 12)
            p_layout.setSpacing(8)

            p_header = QHBoxLayout()
            p_header.addWidget(label("Universal Parameters (methods.params)", "cardTitle"))
            p_header.addStretch()
            if has_params:
                btn_rm_params = QToolButton()
                btn_rm_params.setText("Remove Block")
                btn_rm_params.setToolTip("Remove all universal parameters")
                btn_rm_params.clicked.connect(lambda: self.remove_block("params"))
                p_header.addWidget(btn_rm_params)
            else:
                p_header.addWidget(label("(inherited from base)", "muted"))
            p_layout.addLayout(p_header)

            curr_params = methods_dict.get("params", {}) if has_params else resolved_params
            for pk, pv in list(curr_params.items()):
                p_row = QHBoxLayout()
                p_row.setSpacing(8)
                lbl_k = label(str(pk), "eyebrow")
                lbl_k.setMinimumWidth(140)
                p_row.addWidget(lbl_k)

                line_v = QLineEdit(str(pv if pv is not None else ""))
                line_v.textChanged.connect(lambda val, k=pk: self._on_universal_param_edited(k, val))
                p_row.addWidget(line_v, 1)

                if has_params and pk in methods_dict.get("params", {}):
                    btn_del_p = QToolButton()
                    btn_del_p.setText("✕")
                    btn_del_p.setToolTip(f"Remove universal parameter '{pk}'")
                    btn_del_p.clicked.connect(lambda _, k=pk: self._remove_universal_param(k))
                    p_row.addWidget(btn_del_p)
                p_layout.addLayout(p_row)

            # Row to add a new universal parameter
            add_p_row = QHBoxLayout()
            add_p_row.setSpacing(8)
            combo_add_pk = ComboBox()
            combo_add_pk.setEditable(True)
            combo_add_pk.setPlaceholderText("Param name (e.g. lr, gamma)...")
            common_universal = ["lr", "gamma", "epochs_per_interval", "eval_interval_epochs", "cql_alpha", "weight_decay", "batch_size", "ent_coef"]
            combo_add_pk.addItems(common_universal)
            combo_add_pk.setEditText("")
            add_p_row.addWidget(combo_add_pk, 1)

            txt_add_pv = QLineEdit()
            txt_add_pv.setPlaceholderText("Value...")
            add_p_row.addWidget(txt_add_pv, 1)

            btn_do_add_p = QPushButton("Add")
            btn_do_add_p.clicked.connect(
                lambda: self._add_universal_param(combo_add_pk.currentText().strip(), txt_add_pv.text().strip())
            )
            add_p_row.addWidget(btn_do_add_p)
            p_layout.addLayout(add_p_row)

            m_layout.addWidget(p_card)
        else:
            btn_create_params = QPushButton("+ Add Universal Parameters (params)")
            btn_create_params.clicked.connect(lambda: self.add_block("params"))
            m_layout.addWidget(btn_create_params)

        if not methods_dict:
            m_layout.addWidget(label("Inherits default method from base template.", "muted"))
        else:
            # Sub-cards for each method
            for m_name, m_spec in methods_dict.items():
                if m_name == "params" or not isinstance(m_spec, dict):
                    continue
                sub_card = QFrame()
                sub_card.setObjectName("card")
                sc_layout = QFormLayout(sub_card)
                sc_layout.setVerticalSpacing(8)

                title_row = QHBoxLayout()
                title_row.addWidget(label(f"Method: {m_name}", "cardTitle"))
                title_row.addStretch()
                btn_remove = QToolButton()
                btn_remove.setText("Remove")
                btn_remove.setToolTip(f"Remove method '{m_name}' from this experiment")
                btn_remove.clicked.connect(lambda _, mn=m_name: self.remove_method(mn))
                title_row.addWidget(btn_remove)
                sc_layout.addRow(title_row)

                # Agent picker
                agent_val = str(m_spec.get("agent", ""))
                paradigm_has_agents = bool(tree and paradigm in tree.paradigms and tree.agents_for(paradigm))
                txt_agent = None
                if not paradigm_has_agents and not agent_val:
                    pass
                elif tree and tree.paradigms.get(paradigm):
                    permitted = [a.name for a in tree.agents_for(paradigm)]
                    txt_agent = ComboBox()
                    txt_agent.addItems(permitted)
                    if agent_val and agent_val not in permitted:
                        txt_agent.insertItem(0, agent_val)
                        txt_agent.setToolTip(
                            f"'{agent_val}' is not permitted by {paradigm} "
                            f"(allowed: {', '.join(permitted) or 'none declared'})"
                        )
                    else:
                        txt_agent.setToolTip(f"Agents permitted by {paradigm}")
                    txt_agent.setCurrentText(agent_val)
                    txt_agent.currentTextChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "agent", v)
                    )
                else:
                    txt_agent = QLineEdit(agent_val)
                    txt_agent.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "agent", v)
                    )
                if txt_agent is not None:
                    sc_layout.addRow("Agent Algorithm", txt_agent)

                # Model picker
                model_val = m_spec.get("model", "")
                if isinstance(model_val, dict):
                    model_str = yaml.safe_dump(model_val, default_flow_style=True).strip()
                else:
                    model_str = str(model_val)
                if tree and tree.models and not isinstance(model_val, dict):
                    txt_model = ComboBox()
                    known = sorted(tree.models)
                    txt_model.addItems(known)
                    if model_str and model_str not in known:
                        txt_model.insertItem(0, model_str)
                    txt_model.setCurrentText(model_str)
                    txt_model.setToolTip("Architectures defined in in/config/model/")
                    txt_model.currentTextChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "model", v)
                    )
                else:
                    txt_model = QLineEdit(model_str)
                    txt_model.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "model", v)
                    )
                sc_layout.addRow("Model Architecture", txt_model)

                # Learning rate
                if "lr" in m_spec:
                    row_lr = QHBoxLayout()
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(float(m_spec["lr"]))
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    row_lr.addWidget(spin_lr, 1)
                    btn_del = QToolButton()
                    btn_del.setText("✕")
                    btn_del.setToolTip("Remove lr parameter")
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "lr"))
                    row_lr.addWidget(btn_del)
                    sc_layout.addRow("Learning Rate (lr)", row_lr)

                # Batch size
                if "batch_size" in m_spec:
                    row_bs = QHBoxLayout()
                    spin_bs = SpinBox()
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(int(m_spec["batch_size"]))
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    row_bs.addWidget(spin_bs, 1)
                    btn_del = QToolButton()
                    btn_del.setText("✕")
                    btn_del.setToolTip("Remove batch_size parameter")
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "batch_size"))
                    row_bs.addWidget(btn_del)
                    sc_layout.addRow("Batch Size", row_bs)

                # Gamma
                if "gamma" in m_spec:
                    row_g = QHBoxLayout()
                    spin_g = DoubleSpinBox()
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(float(m_spec["gamma"]))
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    row_g.addWidget(spin_g, 1)
                    btn_del = QToolButton()
                    btn_del.setText("✕")
                    btn_del.setToolTip("Remove gamma parameter")
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "gamma"))
                    row_g.addWidget(btn_del)
                    sc_layout.addRow("Discount Factor (γ)", row_g)

                # Extra hyperparameters (logic_lr, blender_lr, etc.)
                for extra_k, extra_v in list(m_spec.items()):
                    if extra_k in ("agent", "model", "lr", "batch_size", "gamma"):
                        continue
                    if isinstance(extra_v, (int, float, str, bool)):
                        row_extra = QHBoxLayout()
                        line_extra = QLineEdit(str(extra_v))
                        line_extra.textChanged.connect(
                            lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                        )
                        row_extra.addWidget(line_extra, 1)
                        btn_del = QToolButton()
                        btn_del.setText("✕")
                        btn_del.setToolTip(f"Remove hyperparameter '{extra_k}'")
                        btn_del.clicked.connect(lambda _, mn=m_name, ek=extra_k: self._remove_method_param(mn, ek))
                        row_extra.addWidget(btn_del)
                        sc_layout.addRow(str(extra_k), row_extra)

                # Row to add a new hyperparameter to this method
                add_h_row = QHBoxLayout()
                add_h_row.setSpacing(8)
                combo_add_hk = ComboBox()
                combo_add_hk.setEditable(True)
                combo_add_hk.setPlaceholderText("Hyperparameter name (e.g. ent_coef)...")
                common_hyper = ["ent_coef", "blend_ent_coef", "logic_lr", "blender_lr", "target_entropy", "cql_alpha", "tau", "clip_range"]
                combo_add_hk.addItems(common_hyper)
                combo_add_hk.setEditText("")
                add_h_row.addWidget(combo_add_hk, 1)

                txt_add_hv = QLineEdit()
                txt_add_hv.setPlaceholderText("Value...")
                add_h_row.addWidget(txt_add_hv, 1)

                btn_do_add_h = QPushButton("Add")
                btn_do_add_h.clicked.connect(
                    lambda _, mn=m_name, ck=combo_add_hk, tv=txt_add_hv: self._add_method_param(mn, ck.currentText().strip(), tv.text().strip())
                )
                add_h_row.addWidget(btn_do_add_h)
                sc_layout.addRow(add_h_row)

                m_layout.addWidget(sub_card)

        # Method adder row
        add_row = QHBoxLayout()
        permitted_agents = [a.name for a in tree.agents_for(paradigm)] if tree and paradigm in tree.paradigms else []
        self.new_method_is_agent = bool(permitted_agents)
        choices = permitted_agents or (sorted(tree.models) if tree else [])
        add_row.addWidget(label("Add method:" if permitted_agents else "Add model:", "muted"))
        self.combo_new_method = ComboBox()
        self.combo_new_method.addItems(choices)
        self.combo_new_method.setToolTip(
            f"Agents permitted by {paradigm}" if permitted_agents
            else f"{paradigm} has no agents; methods name an architecture"
        )
        if not permitted_agents and choices:
            self.combo_new_method.setCurrentText(
                default_method_model(tree.group(self._group_name() or "") if tree else None, choices)
            )
        add_row.addWidget(self.combo_new_method)
        self.btn_add_method = QPushButton("Add")
        self.btn_add_method.setEnabled(bool(choices))
        self.btn_add_method.clicked.connect(lambda: self.add_method())
        add_row.addWidget(self.btn_add_method)
        add_row.addStretch()
        m_layout.addLayout(add_row)

        box_methods.add_layout(m_layout)
        self.boxes_layout.addWidget(box_methods)

        # --- MODULAR BOX: ENVIRONMENT OVERRIDES ---
        has_env_dict = isinstance(data.get("env"), dict)
        resolved_env = resolved.get("env")
        show_env = has_env_dict or (self.view_mode == "resolved" and isinstance(resolved_env, dict))

        if show_env:
            box_env = ConfigBox("Environment Settings", "Simulation rollouts and vectorized environment settings")
            box_env_header = QHBoxLayout()
            box_env_header.addWidget(label("3. Environment Settings", "eyebrow"))
            box_env_header.addStretch()
            if has_env_dict:
                btn_rm_env = QToolButton()
                btn_rm_env.setText("Remove Block")
                btn_rm_env.clicked.connect(lambda: self.remove_block("env"))
                box_env_header.addWidget(btn_rm_env)
            else:
                box_env_header.addWidget(label("(inherited defaults)", "muted"))
            box_env.add_layout(box_env_header)

            grid_e = QGridLayout()
            grid_e.setVerticalSpacing(10)
            grid_e.setHorizontalSpacing(16)
            env_source = data.get("env") if has_env_dict else (resolved_env if isinstance(resolved_env, dict) else {})

            grid_e.addWidget(label("Parallel Envs (num_envs)"), 0, 0)
            spin_num_envs = SpinBox()
            spin_num_envs.setRange(1, 1024)
            spin_num_envs.setValue(int(env_source.get("num_envs") or 4))
            spin_num_envs.valueChanged.connect(lambda v: self._on_nested_edited("env", "num_envs", v))
            grid_e.addWidget(spin_num_envs, 0, 1)

            grid_e.addWidget(label("Steps per Env (num_steps)"), 0, 2)
            spin_num_steps = SpinBox()
            spin_num_steps.setRange(1, 100000)
            spin_num_steps.setValue(int(env_source.get("num_steps") or 128))
            spin_num_steps.valueChanged.connect(lambda v: self._on_nested_edited("env", "num_steps", v))
            grid_e.addWidget(spin_num_steps, 0, 3)

            box_env.add_layout(grid_e)
            self.boxes_layout.addWidget(box_env)

        # --- MODULAR BOX: TRAINER SETTINGS ---
        has_trainer = "trainer" in data and isinstance(data["trainer"], dict)
        show_trainer = has_trainer or (self.view_mode == "resolved" and "trainer" in resolved)

        if show_trainer:
            box_tr = ConfigBox("Trainer Settings", "Training accelerator and epoch constraints")
            tr_header = QHBoxLayout()
            tr_header.addWidget(label("Trainer Settings", "eyebrow"))
            tr_header.addStretch()
            if has_trainer:
                btn_rm_tr = QToolButton()
                btn_rm_tr.setText("Remove Block")
                btn_rm_tr.clicked.connect(lambda: self.remove_block("trainer"))
                tr_header.addWidget(btn_rm_tr)
            else:
                tr_header.addWidget(label("(inherited defaults)", "muted"))
            box_tr.add_layout(tr_header)

            grid_t = QGridLayout()
            grid_t.setVerticalSpacing(10)
            grid_t.setHorizontalSpacing(16)
            tr_source = data.get("trainer", {}) if has_trainer else resolved.get("trainer", {})

            grid_t.addWidget(label("Accelerator"), 0, 0)
            combo_accel = ComboBox()
            combo_accel.addItems(["cpu", "gpu", "mps", "auto"])
            combo_accel.setCurrentText(str(tr_source.get("accelerator") or "cpu"))
            combo_accel.currentTextChanged.connect(lambda v: self._on_nested_edited("trainer", "accelerator", v))
            grid_t.addWidget(combo_accel, 0, 1)

            grid_t.addWidget(label("Max Epochs"), 0, 2)
            spin_epochs = SpinBox()
            spin_epochs.setRange(1, 1000)
            spin_epochs.setValue(int(tr_source.get("max_epochs") or 1))
            spin_epochs.valueChanged.connect(lambda v: self._on_nested_edited("trainer", "max_epochs", v))
            grid_t.addWidget(spin_epochs, 0, 3)

            box_tr.add_layout(grid_t)
            self.boxes_layout.addWidget(box_tr)

        # --- MODULAR BOX: DATASET SPECIFICATION ---
        has_dataset = "dataset_path" in data or "offline_datasets" in data
        is_offline_req = paradigm in ("offline_rl", "supervised")
        show_dataset = has_dataset or is_offline_req or (self.view_mode == "resolved" and "dataset_path" in resolved)

        if show_dataset:
            box_ds = ConfigBox("Dataset Specification", "Offline transition datasets and replay buffer paths")
            ds_header = QHBoxLayout()
            ds_header.addWidget(label("Dataset Configuration", "eyebrow"))
            ds_header.addStretch()
            if has_dataset and not is_offline_req:
                btn_rm_ds = QToolButton()
                btn_rm_ds.setText("Remove Block")
                btn_rm_ds.clicked.connect(lambda: self.remove_block("dataset"))
                ds_header.addWidget(btn_rm_ds)
            elif is_offline_req:
                ds_header.addWidget(label(f"(required by {paradigm})", "badge"))
            box_ds.add_layout(ds_header)

            ds_form = QFormLayout()
            ds_form.setVerticalSpacing(10)

            # dataset_path with dropdown, browse button, and existence indicator
            ds_path_row = QHBoxLayout()
            ds_path_row.setSpacing(6)
            self.combo_dataset_path = ComboBox()
            self.combo_dataset_path.setEditable(True)
            for dp in discover_datasets():
                self.combo_dataset_path.addItem(dp)
            current_dp = str(data.get("dataset_path") or resolved.get("dataset_path") or "")
            self.combo_dataset_path.setEditText(current_dp)

            lbl_status = QLabel("")
            def _update_ds_status(p_str):
                p_obj = Path(p_str.strip())
                if p_str.strip() and p_obj.exists():
                    try:
                        sz_mb = p_obj.stat().st_size / (1024 * 1024)
                        lbl_status.setText(f"✓ Exists ({sz_mb:.1f} MB)")
                        lbl_status.setStyleSheet("color: #b8bb26; font-size: 11px; font-weight: bold;")
                    except Exception:
                        lbl_status.setText("✓ Exists")
                        lbl_status.setStyleSheet("color: #b8bb26; font-size: 11px;")
                elif p_str.strip():
                    lbl_status.setText("⚠ Not Found")
                    lbl_status.setStyleSheet("color: #fabd2f; font-size: 11px;")
                else:
                    lbl_status.setText("")

            _update_ds_status(current_dp)
            self.combo_dataset_path.currentTextChanged.connect(
                lambda v: (self._on_field_edited("dataset_path", v.strip()), _update_ds_status(v))
            )
            ds_path_row.addWidget(self.combo_dataset_path, 1)

            btn_browse_ds = QToolButton()
            btn_browse_ds.setText("Browse…")
            btn_browse_ds.clicked.connect(lambda: self._browse_dataset_path())
            ds_path_row.addWidget(btn_browse_ds)
            ds_path_row.addWidget(lbl_status)
            ds_form.addRow("Dataset Path", ds_path_row)

            # offline_datasets text line
            self.txt_offline_ds = QLineEdit()
            off_val = data.get("offline_datasets") or resolved.get("offline_datasets") or ""
            if isinstance(off_val, list):
                off_val = ", ".join(str(x) for x in off_val)
            self.txt_offline_ds.setText(str(off_val))
            self.txt_offline_ds.setPlaceholderText("Comma-separated list of dataset names (optional)...")
            self.txt_offline_ds.textChanged.connect(
                lambda v: self._on_field_edited("offline_datasets", [x.strip() for x in v.split(",") if x.strip()] if "," in v else v.strip())
            )
            ds_form.addRow("Offline Datasets", self.txt_offline_ds)

            box_ds.add_layout(ds_form)
            self.boxes_layout.addWidget(box_ds)

        # --- MODULAR BOX: COMPUTE & SLURM RESOURCES ---
        has_res = "resources" in data and isinstance(data["resources"], dict)
        show_res = has_res or (self.view_mode == "resolved" and "resources" in resolved)

        if show_res:
            box_res = ConfigBox("Compute & Slurm Resources", "Resource allocation for cluster runs (time, gpus, cores, memory)")
            res_header = QHBoxLayout()
            res_header.addWidget(label("Compute & Slurm Resources", "eyebrow"))
            res_header.addStretch()
            if has_res:
                btn_rm_res = QToolButton()
                btn_rm_res.setText("Remove Block")
                btn_rm_res.clicked.connect(lambda: self.remove_block("resources"))
                res_header.addWidget(btn_rm_res)
            else:
                res_header.addWidget(label("(inherited defaults)", "muted"))
            box_res.add_layout(res_header)

            grid_res = QGridLayout()
            grid_res.setHorizontalSpacing(16)
            grid_res.setVerticalSpacing(8)
            res_dict = data.get("resources", {}) if has_res else resolved.get("resources", {})

            grid_res.addWidget(label("Time Limit"), 0, 0)
            txt_time = QLineEdit(str(res_dict.get("time") or "02:00:00"))
            txt_time.textChanged.connect(lambda v: self._on_nested_edited("resources", "time", v))
            grid_res.addWidget(txt_time, 0, 1)

            grid_res.addWidget(label("GPUs"), 0, 2)
            txt_gpus = QLineEdit(str(res_dict.get("gpus") if res_dict.get("gpus") is not None else 0))
            txt_gpus.textChanged.connect(lambda v: self._on_nested_edited("resources", "gpus", v))
            grid_res.addWidget(txt_gpus, 0, 3)

            grid_res.addWidget(label("CPU Cores"), 1, 0)
            txt_cores = QLineEdit(str(res_dict.get("cores") if res_dict.get("cores") is not None else 4))
            txt_cores.textChanged.connect(lambda v: self._on_nested_edited("resources", "cores", v))
            grid_res.addWidget(txt_cores, 1, 1)

            grid_res.addWidget(label("Memory"), 1, 2)
            txt_mem = QLineEdit(str(res_dict.get("memory") or "16G"))
            txt_mem.textChanged.connect(lambda v: self._on_nested_edited("resources", "memory", v))
            grid_res.addWidget(txt_mem, 1, 3)

            box_res.add_layout(grid_res)
            self.boxes_layout.addWidget(box_res)

        # --- MODULAR BOX: HYPERPARAMETER SWEEPER (OPTUNA) ---
        has_sweeper = (
            ("hydra" in data and isinstance(data["hydra"], dict) and "sweeper" in data["hydra"])
            or "sweeper" in data
        )
        resolved_hydra = resolved.get("hydra", {}) if isinstance(resolved.get("hydra"), dict) else {}
        show_sweeper = has_sweeper or (self.view_mode == "resolved" and "sweeper" in resolved_hydra)

        if show_sweeper:
            box_sw = ConfigBox("Hyperparameter Sweeper (Optuna)", "Automated parameter sweeps via Hydra & Optuna")
            sw_header = QHBoxLayout()
            sw_header.addWidget(label("Hyperparameter Sweeper (Optuna)", "eyebrow"))
            sw_header.addStretch()
            if has_sweeper:
                btn_rm_sw = QToolButton()
                btn_rm_sw.setText("Remove Block")
                btn_rm_sw.clicked.connect(lambda: self.remove_block("sweeper"))
                sw_header.addWidget(btn_rm_sw)
            else:
                sw_header.addWidget(label("(inherited defaults)", "muted"))
            box_sw.add_layout(sw_header)

            sw_dict = (
                data.get("hydra", {}).get("sweeper", {})
                if "hydra" in data and isinstance(data["hydra"], dict) and "sweeper" in data["hydra"]
                else data.get("sweeper", {})
            )
            sw_form = QGridLayout()
            sw_form.setVerticalSpacing(8)
            sw_form.setHorizontalSpacing(16)

            sw_form.addWidget(label("Study Name"), 0, 0)
            txt_sw_name = QLineEdit(str(sw_dict.get("study_name") or self.current_path.stem if self.current_path else "sweep"))
            txt_sw_name.textChanged.connect(lambda v: self._on_sweeper_edited("study_name", v))
            sw_form.addWidget(txt_sw_name, 0, 1)

            sw_form.addWidget(label("Trials"), 0, 2)
            spin_trials = SpinBox()
            spin_trials.setRange(1, 10000)
            spin_trials.setValue(int(sw_dict.get("n_trials") or 50))
            spin_trials.valueChanged.connect(lambda v: self._on_sweeper_edited("n_trials", v))
            sw_form.addWidget(spin_trials, 0, 3)

            sw_form.addWidget(label("Direction"), 1, 0)
            combo_dir = ComboBox()
            combo_dir.addItems(["maximize", "minimize"])
            combo_dir.setCurrentText(str(sw_dict.get("direction") or "maximize"))
            combo_dir.currentTextChanged.connect(lambda v: self._on_sweeper_edited("direction", v))
            sw_form.addWidget(combo_dir, 1, 1)

            sw_form.addWidget(label("Parallel Jobs"), 1, 2)
            spin_jobs = SpinBox()
            spin_jobs.setRange(1, 64)
            spin_jobs.setValue(int(sw_dict.get("n_jobs") or 1))
            spin_jobs.valueChanged.connect(lambda v: self._on_sweeper_edited("n_jobs", v))
            sw_form.addWidget(spin_jobs, 1, 3)

            box_sw.add_layout(sw_form)
            self.boxes_layout.addWidget(box_sw)

        # --- ADD CONFIGURATION BLOCK DROPDOWN BUTTON ---
        btn_add_block_row = QHBoxLayout()
        self.btn_add_block = QPushButton("+ Add Configuration Block ▾")
        self.btn_add_block.setObjectName("addBlockButton")
        self.btn_add_block.setStyleSheet("padding: 6px 14px; font-weight: 500;")
        add_menu = QMenu(self)
        act_res = add_menu.addAction("Compute & Slurm Resources")
        act_res.triggered.connect(lambda: self.add_block("resources"))
        act_res.setEnabled("resources" not in data)

        act_tr = add_menu.addAction("Trainer Overrides")
        act_tr.triggered.connect(lambda: self.add_block("trainer"))
        act_tr.setEnabled("trainer" not in data)

        act_env = add_menu.addAction("Environment Overrides")
        act_env.triggered.connect(lambda: self.add_block("env"))
        act_env.setEnabled(not isinstance(data.get("env"), dict))

        act_ds = add_menu.addAction("Dataset Configuration")
        act_ds.triggered.connect(lambda: self.add_block("dataset"))
        act_ds.setEnabled("dataset_path" not in data)

        act_sw = add_menu.addAction("Hyperparameter Sweeper (Optuna)")
        act_sw.triggered.connect(lambda: self.add_block("sweeper"))
        act_sw.setEnabled(not has_sweeper)

        self.btn_add_block.setMenu(add_menu)
        btn_add_block_row.addWidget(self.btn_add_block)
        btn_add_block_row.addStretch()
        self.boxes_layout.addLayout(btn_add_block_row)

    def _render_generic_boxes(self):
        """Render clean boxes for modular YAML configs (agent/*.yaml, env/*.yaml, etc.)."""
        self.paradigm_badge.hide()
        data = self.raw_data

        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        complex_items = {k: v for k, v in data.items() if isinstance(v, (dict, list))}

        stem = self.current_path.stem if self.current_path else ""
        filtered_scalars = {}
        for k, v in scalars.items():
            if k in ("name", "algorithm", "architecture", "reasoner", "experiment_id", "id"):
                continue
            if k in ("type", "env_id") and str(v).lower() == stem.lower():
                continue
            if isinstance(v, str) and v.lower() == stem.lower() and k in ("model", "agent"):
                continue
            filtered_scalars[k] = v

        if filtered_scalars:
            box_props = ConfigBox("Parameters", f"Primary settings in {self.current_path.stem}")
            form = QFormLayout()
            form.setVerticalSpacing(8)
            for k, v in filtered_scalars.items():
                if isinstance(v, bool):
                    chk = QCheckBox()
                    chk.setChecked(v)
                    chk.toggled.connect(lambda val, key=k: self._on_field_edited(key, val))
                    form.addRow(k, chk)
                elif isinstance(v, (int, float)):
                    spin = number_field(v)
                    spin.valueChanged.connect(lambda val, key=k: self._on_field_edited(key, val))
                    form.addRow(k, spin)
                else:
                    txt = QLineEdit(str(v if v is not None else ""))
                    txt.textChanged.connect(lambda val, key=k: self._on_field_edited(key, val))
                    form.addRow(k, txt)
            box_props.add_layout(form)
            self.boxes_layout.addWidget(box_props)

        for section_name, section_val in complex_items.items():
            box_sec = ConfigBox(section_name, f"Section: {section_name}")
            if isinstance(section_val, dict):
                form_sec = QFormLayout()
                form_sec.setVerticalSpacing(8)
                for sk, sv in section_val.items():
                    if isinstance(sv, (int, float, str, bool)):
                        line = QLineEdit(str(sv))
                        line.textChanged.connect(lambda val, s=section_name, k=sk: self._on_nested_edited(s, k, val))
                        form_sec.addRow(sk, line)
                    else:
                        form_sec.addRow(sk, label(str(sv), "muted"))
                box_sec.add_layout(form_sec)
            elif isinstance(section_val, list):
                lbl = label(", ".join(str(item) for item in section_val), "muted")
                lbl.setWordWrap(True)
                box_sec.add_widget(lbl)
            self.boxes_layout.addWidget(box_sec)

    # ── Edit Handlers ──────────────────────────────────────────────────────────

    def _on_field_edited(self, key, value):
        if self._block_updates:
            return
        self.raw_data[key] = value
        self._mark_dirty()

    def add_method(self, agent=None, model=None):
        """Add a method to the open experiment. Returns its name, or None.

        The picker holds an agent where the paradigm has them and a model where
        it does not, so which one the selection means depends on the paradigm.
        """
        if agent is None and model is None:
            chosen = self.combo_new_method.currentText() if hasattr(self, "combo_new_method") else ""
            if not chosen:
                return None
            if getattr(self, "new_method_is_agent", True):
                agent, model = chosen, "dnn"
            else:
                agent, model = None, chosen
        name = add_method_to(self.raw_data, agent, model or "dnn")
        self._mark_dirty()
        self._render_boxes()
        return name

    def remove_method(self, name):
        """Remove a method from the open experiment. Returns True if it went."""
        if not remove_method_from(self.raw_data, name):
            return False
        self._mark_dirty()
        self._render_boxes()
        return True

    def _on_paradigm_changed(self, value):
        if self._block_updates:
            return
        self.raw_data["paradigm"] = value
        # An env or agent valid under the old paradigm may be forbidden under the
        # new one, so drop stale choices and rebuild against the new constraints.
        tree = config_tree()
        if tree and value in tree.paradigms:
            rules = tree.paradigms[value]
            env_name = self.raw_data.get("env")
            if isinstance(env_name, str) and env_name in tree.environments:
                if not rules.permits_environment(tree.environments[env_name]):
                    self.raw_data.pop("env", None)
        self._mark_dirty()
        self._render_boxes()

    def _on_nested_edited(self, parent_key, child_key, value):
        if self._block_updates:
            return
        if parent_key not in self.raw_data or not isinstance(self.raw_data[parent_key], dict):
            self.raw_data[parent_key] = {}
        self.raw_data[parent_key][child_key] = value
        self._mark_dirty()

    def _on_method_param_edited(self, method_name, param_key, value):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        # Parse numbers if string represents a float/int
        if isinstance(value, str):
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        m_spec[param_key] = value
        self._mark_dirty()

    def _browse_dataset_path(self):
        """Open file dialog to select a dataset file or directory."""
        initial_dir = str(Path.cwd() / "in" / "datasets")
        if not Path(initial_dir).is_dir():
            initial_dir = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            initial_dir,
            "Dataset Files (*.pkl *.csv *.parquet *.h5 *.pt);;All Files (*)",
        )
        if selected:
            try:
                rel = Path(selected).relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = selected
            if hasattr(self, "combo_dataset_path"):
                self.combo_dataset_path.setEditText(rel)
            self._on_field_edited("dataset_path", rel)

    def _on_sweeper_edited(self, key, value):
        if self._block_updates:
            return
        if "hydra" in self.raw_data and isinstance(self.raw_data["hydra"], dict) and "sweeper" in self.raw_data["hydra"]:
            self.raw_data["hydra"]["sweeper"][key] = _parse_yaml_value(value)
        else:
            self.raw_data.setdefault("sweeper", {})[key] = _parse_yaml_value(value)
        self._mark_dirty()

    def remove_block(self, block_name: str):
        """Remove a configuration block from the open experiment."""
        if block_name == "resources":
            self.raw_data.pop("resources", None)
        elif block_name == "trainer":
            self.raw_data.pop("trainer", None)
        elif block_name == "env":
            if isinstance(self.raw_data.get("env"), dict):
                self.raw_data.pop("env", None)
        elif block_name == "dataset":
            self.raw_data.pop("dataset_path", None)
            self.raw_data.pop("offline_datasets", None)
        elif block_name == "sweeper":
            self.raw_data.pop("sweeper", None)
            if "hydra" in self.raw_data and isinstance(self.raw_data["hydra"], dict):
                self.raw_data["hydra"].pop("sweeper", None)
                if not self.raw_data["hydra"]:
                    self.raw_data.pop("hydra", None)
        elif block_name == "params":
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                self.raw_data["methods"].pop("params", None)
        self._mark_dirty()
        self._render_boxes()

    def add_block(self, block_name: str):
        """Add an optional configuration block to the open experiment."""
        if block_name == "resources":
            self.raw_data["resources"] = {"time": "02:00:00", "gpus": 1, "cores": 8, "memory": "16G"}
        elif block_name == "trainer":
            self.raw_data["trainer"] = {"accelerator": "cpu", "max_epochs": 25}
        elif block_name == "env":
            self.raw_data["env"] = {"num_envs": 4, "num_steps": 128}
        elif block_name == "dataset":
            self.raw_data["dataset_path"] = "in/datasets/mimic"
        elif block_name == "sweeper":
            exp_stem = self.current_path.stem if self.current_path else "sweep"
            self.raw_data.setdefault("hydra", {})["sweeper"] = {
                "sampler": {"_target_": "optuna.samplers.TPESampler", "seed": 42},
                "direction": "maximize",
                "storage": "sqlite:///results/optuna/optuna.db",
                "study_name": exp_stem,
                "n_trials": 50,
                "n_jobs": 1,
            }
        elif block_name == "params":
            self.raw_data.setdefault("methods", {})["params"] = {"lr": 0.0003}
        self._mark_dirty()
        self._render_boxes()

    def _on_universal_param_edited(self, key, value):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        params = methods.setdefault("params", {})
        params[key] = _parse_yaml_value(value)
        self._mark_dirty()

    def _remove_universal_param(self, key):
        if "methods" in self.raw_data and "params" in self.raw_data["methods"]:
            self.raw_data["methods"]["params"].pop(key, None)
            if not self.raw_data["methods"]["params"]:
                self.raw_data["methods"].pop("params", None)
            self._mark_dirty()
            self._render_boxes()

    def _add_universal_param(self, key, value):
        if not key:
            return
        methods = self.raw_data.setdefault("methods", {})
        params = methods.setdefault("params", {})
        params[key] = _parse_yaml_value(value)
        self._mark_dirty()
        self._render_boxes()

    def _remove_method_param(self, method_name, param_key):
        if "methods" in self.raw_data and method_name in self.raw_data["methods"]:
            self.raw_data["methods"][method_name].pop(param_key, None)
            self._mark_dirty()
            self._render_boxes()

    def _add_method_param(self, method_name, param_key, value):
        if not param_key:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        m_spec[param_key] = _parse_yaml_value(value)
        self._mark_dirty()
        self._render_boxes()

    def _mark_dirty(self):
        self.is_dirty = True
        if getattr(self, "auto_save", False):
            self.save_to_disk()
        else:
            self._update_dirty_ui()
        self.config_changed.emit()

    def _update_dirty_ui(self):
        if self.is_dirty and not getattr(self, "auto_save", False):
            self.dirty_status.setText("● Modified (unsaved)")
            self.dirty_status.show()
            self.dirty_changed.emit(True)
        else:
            self.is_dirty = False
            self.dirty_status.setText("")
            self.dirty_status.hide()
            self.dirty_changed.emit(False)

    def save_to_disk(self, show_error: bool = False):
        """Save current YAML dictionary back to disk."""
        if not self.current_path:
            return False
        try:
            self._clean_redundant_params()
            yaml_str = yaml.safe_dump(self.raw_data, sort_keys=False)
            self.current_path.write_text(self._preamble + yaml_str, encoding="utf-8")
            self.is_dirty = False
            self._update_dirty_ui()
            if self.current_rel_path:
                self.dirty_state_changed.emit(self.current_rel_path, False)
            self.save_requested.emit()
            return True
        except Exception as exc:
            if show_error:
                QMessageBox.critical(self, "Save Error", f"Could not save file to disk:\n{exc}")
            return False

    def get_overrides(self):
        """Build command-line Hydra overrides matching current UI values."""
        overrides = []
        data = self.raw_data

        exp_id = data.get("experiment_id") or (self.current_path.stem if self.current_path else None)
        if exp_id:
            overrides.append(f"++experiment_id='{exp_id}'")

        if "description" in data and data["description"]:
            overrides.append(f"++description='{data['description']}'")

        # Hydra config-group selections (defaults in in/config/config.yaml)
        for group in ("paradigm", "env"):
            value = data.get(group)
            if isinstance(value, str) and value and value != self.INHERIT:
                overrides.append(f"{group}={value}")

        if "seed" in data:
            overrides.append(f"seed={data['seed']}")
        if "total_timesteps" in data:
            overrides.append(f"total_timesteps={data['total_timesteps']}")
        if "tensorboard" in data:
            overrides.append(f"tensorboard={str(data['tensorboard']).lower()}")
        if "save_dataset" in data and data["save_dataset"]:
            overrides.append(f"save_dataset={str(data['save_dataset']).lower()}")
        if "recover" in data and data["recover"]:
            overrides.append(f"recover={str(data['recover']).lower()}")
        if "no_plot" in data and data["no_plot"]:
            overrides.append(f"no_plot={str(data['no_plot']).lower()}")

        if "dataset_path" in data and data["dataset_path"]:
            overrides.append(f"dataset_path='{data['dataset_path']}'")

        # Slurm resources overrides
        if "resources" in data and isinstance(data["resources"], dict):
            for k, v in data["resources"].items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++resources.{k}='{v}'" if isinstance(v, str) else f"++resources.{k}={v}")

        # Universal params overrides
        if "methods" in data and isinstance(data["methods"], dict) and "params" in data["methods"]:
            for k, v in data["methods"]["params"].items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++methods.params.{k}={v}")

        # Methods overrides
        for m_name, m_spec in data.get("methods", {}).items():
            if m_name == "params" or not isinstance(m_spec, dict):
                continue
            for k, v in m_spec.items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++methods.{m_name}.{k}={v}")

        return overrides
