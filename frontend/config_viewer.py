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
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
        self.file_title.setText(self.current_path.stem)
        self.file_title.setToolTip(f"{self.current_rel_path}\n(Double-click to rename)")

        is_experiment = (
            self.current_rel_path.startswith("experiment/") or
            "methods" in self.raw_data or
            any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )

        if is_experiment:
            self._render_experiment_boxes()
        else:
            self._render_generic_boxes()

        self._cap_form_fields()
        self.boxes_layout.addStretch()
        self._block_updates = False

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
        form_budget = QGridLayout()
        form_budget.setVerticalSpacing(10)
        form_budget.setHorizontalSpacing(16)

        # Total timesteps
        form_budget.addWidget(label("Total Timesteps"), 0, 0)
        self.spin_timesteps = SpinBox()
        self.spin_timesteps.setRange(100, 100_000_000)
        self.spin_timesteps.setSingleStep(1_000)
        self.spin_timesteps.setValue(int(data.get("total_timesteps") or 10000))
        self.spin_timesteps.valueChanged.connect(lambda v: self._on_field_edited("total_timesteps", v))
        form_budget.addWidget(self.spin_timesteps, 0, 1)
        self.field_widgets["total_timesteps"] = self.spin_timesteps

        # Seed + Randomize button
        form_budget.addWidget(label("Random Seed"), 0, 2)
        seed_row = QHBoxLayout()
        seed_row.setSpacing(6)
        self.spin_seed = SpinBox()
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(int(data.get("seed") if data.get("seed") is not None else 42))
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
        self.spin_intervals.setValue(int(data.get("intervals_count") or 4))
        self.spin_intervals.valueChanged.connect(lambda v: self._on_field_edited("intervals_count", v))
        form_budget.addWidget(self.spin_intervals, 1, 1)
        self.field_widgets["intervals_count"] = self.spin_intervals

        # Eval episodes
        form_budget.addWidget(label("Eval Episodes"), 1, 2)
        self.spin_eval_ep = SpinBox()
        self.spin_eval_ep.setRange(0, 1000)
        self.spin_eval_ep.setValue(int(data.get("eval_episodes") if data.get("eval_episodes") is not None else 100))
        self.spin_eval_ep.valueChanged.connect(lambda v: self._on_field_edited("eval_episodes", v))
        form_budget.addWidget(self.spin_eval_ep, 1, 3)
        self.field_widgets["eval_episodes"] = self.spin_eval_ep

        # Disable fields the paradigm forbids rather than letting the pipeline
        # reject them at launch. A greyed spin box is hard to tell from an
        # editable one in a dark theme, and leaving the widget default showing
        # would advertise a value the run will not use, so show what the group
        # base actually pins and say where it came from.
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

        # Checkboxes: Tensorboard, Save Dataset, Recover
        chk_row = QHBoxLayout()
        chk_row.setSpacing(18)
        self.chk_tb = QCheckBox("Log to TensorBoard")
        self.chk_tb.setChecked(bool(data.get("tensorboard", True)))
        self.chk_tb.toggled.connect(lambda v: self._on_field_edited("tensorboard", v))
        chk_row.addWidget(self.chk_tb)

        self.chk_save_ds = QCheckBox("Save Dataset")
        self.chk_save_ds.setChecked(bool(data.get("save_dataset", False)))
        self.chk_save_ds.toggled.connect(lambda v: self._on_field_edited("save_dataset", v))
        chk_row.addWidget(self.chk_save_ds)

        self.chk_recover = QCheckBox("Recover Checkpoint")
        self.chk_recover.setChecked(bool(data.get("recover", False)))
        self.chk_recover.toggled.connect(lambda v: self._on_field_edited("recover", v))
        chk_row.addWidget(self.chk_recover)

        self.chk_no_plot = QCheckBox("No Plots")
        self.chk_no_plot.setChecked(bool(data.get("no_plot", False)))
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

        if not methods_dict:
            m_layout.addWidget(label("Inherits default method from base template.", "muted"))
        else:
            # If universal params exist
            if "params" in methods_dict:
                p_card = QFrame()
                p_card.setObjectName("card")
                p_layout = QFormLayout(p_card)
                p_layout.addRow(label("Universal Parameters (params:)", "eyebrow"))
                for k, v in methods_dict["params"].items():
                    if isinstance(v, (int, float, str, bool)):
                        p_layout.addRow(k, label(str(v), "muted"))
                m_layout.addWidget(p_card)

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

                # Agent — restricted to what this paradigm permits, and omitted
                # entirely where it permits none: a supervised method names an
                # architecture and has no policy to choose actions with, so an
                # empty agent picker would only invite an invalid edit.
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
                        # Never silently rewrite what is already on disk.
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

                # Model — a dict value is a nested override, so keep it as text
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
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(float(m_spec["lr"]))
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    sc_layout.addRow("Learning Rate (lr)", spin_lr)

                # Batch size
                if "batch_size" in m_spec:
                    spin_bs = SpinBox()
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(int(m_spec["batch_size"]))
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    sc_layout.addRow("Batch Size", spin_bs)

                # Gamma
                if "gamma" in m_spec:
                    spin_g = DoubleSpinBox()
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(float(m_spec["gamma"]))
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    sc_layout.addRow("Discount Factor (γ)", spin_g)

                # Extra hyperparameters (logic_lr, blender_lr, etc.)
                for extra_k, extra_v in m_spec.items():
                    if extra_k in ("agent", "model", "lr", "batch_size", "gamma"):
                        continue
                    if isinstance(extra_v, (int, float, str, bool)):
                        line_extra = QLineEdit(str(extra_v))
                        line_extra.textChanged.connect(
                            lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                        )
                        sc_layout.addRow(extra_k, line_extra)

                m_layout.addWidget(sub_card)

        # Adding a method is how an experiment becomes a comparison, and 17 of
        # the 46 experiments configure more than one.
        add_row = QHBoxLayout()
        # A paradigm that declares no agents picks a method by architecture
        # instead: supervised learning has no policy, so its methods are
        # model-only (ep_lstm, ep_transformer).
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
            # Alphabetically first across every model is blendrl, a neuro-symbolic
            # RL architecture; follow what the group already uses instead.
            self.combo_new_method.setCurrentText(
                default_method_model(tree.group(self._group_name() or "") if tree else None, choices)
            )
        add_row.addWidget(self.combo_new_method)
        self.btn_add_method = QPushButton("Add")
        self.btn_add_method.setEnabled(bool(choices))
        # Swallow clicked(bool): a stray False would read as the agent name.
        self.btn_add_method.clicked.connect(lambda: self.add_method())
        add_row.addWidget(self.btn_add_method)
        add_row.addStretch()
        m_layout.addLayout(add_row)

        box_methods.add_layout(m_layout)
        self.boxes_layout.addWidget(box_methods)

        # --- BOX 3: ENVIRONMENT & TRAINER SETTINGS ---
        box_env_trainer = ConfigBox("3. Environment & Trainer", "Simulation rollouts and training accelerator options")
        grid_et = QGridLayout()
        grid_et.setVerticalSpacing(10)
        grid_et.setHorizontalSpacing(16)

        env_cfg = data.get("env", {})
        if isinstance(env_cfg, dict):
            grid_et.addWidget(label("Parallel Envs (num_envs)"), 0, 0)
            spin_num_envs = SpinBox()
            spin_num_envs.setRange(1, 1024)
            spin_num_envs.setValue(int(env_cfg.get("num_envs") or 4))
            spin_num_envs.valueChanged.connect(lambda v: self._on_nested_edited("env", "num_envs", v))
            grid_et.addWidget(spin_num_envs, 0, 1)

            grid_et.addWidget(label("Steps per Env (num_steps)"), 0, 2)
            spin_num_steps = SpinBox()
            spin_num_steps.setRange(1, 100000)
            spin_num_steps.setValue(int(env_cfg.get("num_steps") or 128))
            spin_num_steps.valueChanged.connect(lambda v: self._on_nested_edited("env", "num_steps", v))
            grid_et.addWidget(spin_num_steps, 0, 3)

        trainer_cfg = data.get("trainer", {})
        if isinstance(trainer_cfg, dict):
            grid_et.addWidget(label("Accelerator"), 1, 0)
            combo_accel = ComboBox()
            combo_accel.addItems(["cpu", "gpu", "mps", "auto"])
            combo_accel.setCurrentText(str(trainer_cfg.get("accelerator") or "cpu"))
            combo_accel.currentTextChanged.connect(lambda v: self._on_nested_edited("trainer", "accelerator", v))
            grid_et.addWidget(combo_accel, 1, 1)

            grid_et.addWidget(label("Max Epochs"), 1, 2)
            spin_epochs = SpinBox()
            spin_epochs.setRange(1, 1000)
            spin_epochs.setValue(int(trainer_cfg.get("max_epochs") or 1))
            spin_epochs.valueChanged.connect(lambda v: self._on_nested_edited("trainer", "max_epochs", v))
            grid_et.addWidget(spin_epochs, 1, 3)

        box_env_trainer.add_layout(grid_et)
        self.boxes_layout.addWidget(box_env_trainer)

        # --- BOX 4: CLUSTER & HARDWARE RESOURCES ---
        box_res = ConfigBox("4. Compute & Slurm Resources", "Resource allocation for cluster runs (time, gpus, cores)")
        grid_res = QGridLayout()
        grid_res.setHorizontalSpacing(16)
        grid_res.setVerticalSpacing(8)

        res_dict = data.get("resources", {}) or {}
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

        # Methods overrides
        for m_name, m_spec in data.get("methods", {}).items():
            if m_name == "params" or not isinstance(m_spec, dict):
                continue
            for k, v in m_spec.items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++methods.{m_name}.{k}={v}")

        return overrides
