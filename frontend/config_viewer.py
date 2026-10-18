"""Full-screen, simplified, boxed configuration viewer for Theta-IDE."""
import copy
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_model import ConfigTree, default_method_model
from .config_model import add_method as add_method_to
from .config_model import remove_method as remove_method_from
from .sidetabs import svg_icon
from .theme import theme_color
from .widgets import ComboBox, DoubleSpinBox, SpinBox, label, SmoothScrollArea, ToggleSlider

# Forms stay readable on wide windows: the column of boxes stops growing past CONTENT_WIDTH,
# and fields in label/field rows are capped by kind so a number doesn't get a 1,000px box.
CONTENT_WIDTH = 960
NUMBER_FIELD_WIDTH = 140
TEXT_FIELD_WIDTH = 220

SUBTLE_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "muted",
    (QIcon.Mode.Active, QIcon.State.Off): "text",
    (QIcon.Mode.Selected, QIcon.State.Off): "accent",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
}


def _make_subtle_tool_button(icon_name: str, tooltip: str, size: int = 24, icon_size: int = 14) -> QToolButton:
    """Create a subtle icon-only tool button (trash, close, plus) without prominent borders."""
    btn = QToolButton()
    btn.setIcon(svg_icon(icon_name, SUBTLE_ICON_COLORS))
    btn.setIconSize(QSize(icon_size, icon_size))
    btn.setFixedSize(size, size)
    btn.setToolTip(tooltip)
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: none;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
        }
    """)
    return btn


def _make_param_adder_button(
    tooltip: str,
    candidates: list[str],
    on_chosen,
    parent_widget: QWidget | None = None,
    allow_custom: bool = True,
) -> QToolButton:
    """Create a subtle plus button that opens a focused dropdown menu for selecting a parameter."""
    btn = _make_subtle_tool_button("plus", tooltip, size=22, icon_size=12)
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: 1px dashed #504945;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
            border-color: #a89984;
        }
        QToolButton::menu-indicator {
            image: none;
            width: 0px;
        }
    """)
    menu = QMenu(btn)
    menu.setStyleSheet("""
        QMenu {
            background-color: #282828;
            border: 1px solid #504945;
            border-radius: 4px;
            padding: 4px 0px;
        }
        QMenu::item {
            padding: 5px 18px 5px 12px;
            color: #ebdbb2;
            font-size: 12px;
        }
        QMenu::item:selected {
            background-color: #3c3836;
            color: #fbf1c7;
        }
        QMenu::separator {
            height: 1px;
            background: #3c3836;
            margin: 4px 6px;
        }
    """)
    for cand in candidates:
        act = menu.addAction(cand)
        act.triggered.connect(lambda _, c=cand: on_chosen(c))

    if allow_custom:
        if candidates:
            menu.addSeparator()
        act_custom = menu.addAction("+ Custom...")
        def _prompt_custom():
            val, ok = QInputDialog.getText(parent_widget or btn, "Add Parameter", "Parameter name:")
            if ok and val.strip():
                on_chosen(val.strip())
        act_custom.triggered.connect(_prompt_custom)

    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    return btn



def _parse_yaml_value(text: str):
    """Convert text representation to typed YAML scalar, list, or dict."""
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
        pass
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            return yaml.safe_load(stripped)
        except Exception:
            pass
    return stripped


TRAINING_HYPERPARAMETERS = (
    "lr",
    "batch_size",
    "gamma",
    "epochs",
    "eval_every_n_epochs",
    "weight_decay",
    "from_study",
)


def get_agent_candidate_hyperparameters(
    agent_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to an agent algorithm."""
    if not agent_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        agent_params = tree.agent_hyperparameters(agent_name)
        for k in agent_params.keys():
            if k not in ("algorithm", "name", "type") and k not in TRAINING_HYPERPARAMETERS:
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    fallbacks = {
        "cql": ["cql_alpha", "soft_target_tau", "use_actor", "target_entropy", "cql_weight"],
        "ppo": [
            "clip_coef",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "gae_lambda",
            "num_steps",
            "num_envs",
            "update_epochs",
            "max_grad_norm",
            "target_kl",
        ],
        "iql": ["expectile", "temperature", "soft_target_tau"],
    }
    for k in fallbacks.get(agent_name, []):
        if k not in current_keys and k not in candidates:
            candidates.append(k)
    return candidates


def get_model_candidate_hyperparameters(
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to a model architecture (or composite top-level)."""
    if not model_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        model_params = tree.model_parameters(model_name)
        for k in model_params:
            if k not in ("architecture", "type", "lightning_module", "name", "slots"):
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    if not candidates:
        if model_name == "dnn":
            candidates = ["hidden_sizes", "activation"]
        elif model_name == "dueling_resnet":
            candidates = ["hidden_sizes"]
        elif model_name in ("dense_autoencoder", "autoencoder"):
            candidates = ["hidden_dim", "latent_dim", "dropout"]
        candidates = [k for k in candidates if k not in current_keys]
    return candidates


def get_training_candidate_hyperparameters(current_keys: set[str] | None = None) -> list[str]:
    """Candidate general training hyperparameters for a method."""
    current_keys = current_keys or set()
    return [k for k in TRAINING_HYPERPARAMETERS if k not in current_keys]


def get_method_candidate_hyperparameters(
    agent_name: str | None,
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Dynamically determine valid hyperparameters for a method based on its agent and model."""
    current_keys = current_keys or set()
    candidates: list[str] = []

    # 1. Base training parameters
    for p in get_training_candidate_hyperparameters(current_keys):
        if p not in candidates:
            candidates.append(p)

    # 2. Agent-specific parameters
    for k in get_agent_candidate_hyperparameters(agent_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    # 3. Model-specific parameters
    for k in get_model_candidate_hyperparameters(model_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    return candidates


def get_method_candidate_subcomponents(
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Return available subcomponent names for a composite model that are not yet added."""
    if not model_name:
        return []
    current_keys = current_keys or set()
    subcomps = tree.model_subcomponents(model_name) if tree else {}
    if not subcomps and tree:
        slots = tree.model_slots(model_name)
        if slots:
            subcomps = {s: {} for s in slots}
    return [k for k in subcomps.keys() if k not in current_keys]


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


class IndentedHierarchyBlock(QFrame):
    """Indented container with a subtle vertical guide line denoting hierarchy in Theta-IDE."""

    def __init__(self, parent=None, level: int = 1):
        super().__init__(parent)
        self.setObjectName(f"hierarchyBlock_lvl{level}")
        left_margin = 12 if level == 1 else 14
        self.setStyleSheet(f"""
            QFrame#hierarchyBlock_lvl{level} {{
                border-left: 2px solid #504945;
                margin-left: {left_margin}px;
                padding-left: 10px;
                background: transparent;
                border-top: none;
                border-right: none;
                border-bottom: none;
            }}
        """)
        self.form_layout = QFormLayout(self)
        self.form_layout.setContentsMargins(0, 4, 0, 6)
        self.form_layout.setVerticalSpacing(6)
        self.form_layout.setHorizontalSpacing(10)

    def addRow(self, label_or_widget, widget=None):
        if widget is not None:
            self.form_layout.addRow(label_or_widget, widget)
        else:
            self.form_layout.addRow(label_or_widget)


# Top-level experiment keys the five fixed boxes already render. Everything else
# goes through _render_remaining_sections, so a paradigm's own settings are
# editable rather than invisible: every supervised experiment keeps its real
# configuration under `early_prediction:`, and none of it used to appear at all.
FIXED_BOX_KEYS = frozenset(
    {
        "defaults",
        "paradigm",
        "experiment_id",
        "seed",
        "total_timesteps",
        "eval_episodes",
        "epochs",
        "eval_every_n_epochs",
        "methods",
        "env",
        "site",
        "trainer",
        "resources",
        "tensorboard",
        "no_plot",
        "save_dataset",
        "recover",
        # Claimed by the panel overhaul: the sweep box edits hydra.sweeper, and
        # the dataset row edits dataset_path, so repeating them here would show
        # the same setting twice.
        "hydra",
        "dataset_path",
        "description",
        "eval_freq_steps",
        "eval_freq",
        "experiment_id",
    }
)


def parse_list_field(text):
    """Comma-separated text back to a YAML list, keeping each item's type.

    `plots` holds names while `eval_protocol.horizons` holds numbers, so items
    go through the YAML scalar parser instead of all becoming strings.
    """
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(yaml.safe_load(part))
        except yaml.YAMLError:
            items.append(part)
    return items


_CONFIG_TREE = None
_CONFIG_TREE_LOADED = False


def invalidate_config_tree():
    """Clear cached in/config tree so subsequent lookups re-scan the filesystem."""
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    _CONFIG_TREE = None
    _CONFIG_TREE_LOADED = False


def config_tree(force_reload: bool = False):
    """The parsed in/config tree, or None if it cannot be found.

    Callers must handle None: the viewer falls back to free-text entry so it
    still works when run outside a checkout.
    """
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    if force_reload or not _CONFIG_TREE_LOADED:
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
    """Clickable heading label that emits clicked and double_clicked on click."""
    clicked = pyqtSignal()
    double_clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("heading")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)



class InlineTitleEditor(QLineEdit):
    """Line editor supporting cancellation on Escape key press."""
    escape_pressed = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MethodDragHandle(QToolButton):
    """Grip icon button to initiate dragging and reordering of method cards."""

    def __init__(self, method_name: str, parent=None):
        super().__init__(parent)
        self.method_name = method_name
        self.setIcon(svg_icon("grip", SUBTLE_ICON_COLORS))
        self.setIconSize(QSize(14, 14))
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to reorder method")
        self.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 3px;
                padding: 0;
            }
            QToolButton:hover {
                background: #3c3836;
            }
        """)
        self._drag_start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._start_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-theta-method", QByteArray(self.method_name.encode("utf-8")))
        drag.setMimeData(mime)

        p = self.parent()
        while p and not isinstance(p, (ConfigBox, QFrame)):
            p = p.parent()
        target_widget = p or self
        pixmap = target_widget.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, 15))
        drag.exec(Qt.DropAction.MoveAction)


class MethodCardsContainer(QWidget):
    """Container holding method cards arranged in 2 columns that accepts drops and displays drop indicator line."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.setAcceptDrops(True)
        self._drop_index = None
        self._cards: list[tuple[str, QWidget]] = []

    def add_method_card(self, method_name: str, card: QWidget):
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        idx = len(self._cards)
        row = idx // 2
        col = idx % 2
        self._cards.append((method_name, card))
        self.grid.addWidget(card, row, col)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            pos = event.position().toPoint()
            idx = self._calculate_drop_index(pos)
            if idx != self._drop_index:
                self._drop_index = idx
                self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_index = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            method_name = bytes(event.mimeData().data("application/x-theta-method")).decode("utf-8")
            target_idx = self._drop_index
            self._drop_index = None
            self.update()
            if target_idx is not None:
                self.viewer.reorder_method(method_name, target_idx)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _calculate_drop_index(self, pos: QPoint) -> int:
        if not self._cards:
            return 0
        for i, (_, card) in enumerate(self._cards):
            geom = card.geometry()
            if pos.y() < geom.top():
                return i
            if geom.top() <= pos.y() <= geom.bottom():
                if pos.x() < geom.center().x():
                    return i
                col = i % 2
                if col == 1 or i == len(self._cards) - 1:
                    if pos.x() >= geom.center().x():
                        return i + 1
        return len(self._cards)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_index is not None and self._cards:
            painter = QPainter(self)
            try:
                acc_color = QColor(theme_color("accent"))
            except Exception:
                acc_color = QColor("#fabd2f")
            pen = QPen(acc_color, 3)
            painter.setPen(pen)
            if self._drop_index < len(self._cards):
                card = self._cards[self._drop_index][1]
                g = card.geometry()
                painter.drawLine(g.left() - 4, g.top() + 4, g.left() - 4, g.bottom() - 4)
            else:
                card = self._cards[-1][1]
                g = card.geometry()
                painter.drawLine(g.right() + 4, g.top() + 4, g.right() + 4, g.bottom() - 4)
            painter.end()


class ConfigViewer(QWidget):
    """Modern, wide, boxed configuration viewer that displays experiment and

    component configurations in themed cards.
    """
    config_changed = pyqtSignal()
    save_requested = pyqtSignal()
    dirty_state_changed = pyqtSignal(str, bool)
    dirty_changed = pyqtSignal(bool)
    file_renamed = pyqtSignal(str, str)
    group_renamed = pyqtSignal(str, str)
    dir_renamed = pyqtSignal(str, str)
    edit_code_requested = pyqtSignal(object, str)
    export_component_requested = pyqtSignal(object, str)
    delete_component_requested = pyqtSignal(object, str)

    INHERIT = "(inherit from base)"

    @property
    def tree(self):
        return config_tree()

    def __init__(self, settings_manager=None, parent=None):
        if isinstance(settings_manager, QWidget) and parent is None:
            parent = settings_manager
            settings_manager = None
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.is_component_mode = False
        self.current_path = None
        self.current_rel_path = None
        self.raw_data = {}
        self.is_dirty = False
        self._block_updates = False
        self._is_editing_title = False
        self._committing_rename = False
        self._is_editing_group = False
        self._committing_group_rename = False
        self.field_widgets = {}
        self._preamble = ""
        self.auto_save = False
        # Field edits (typing) auto-save after a pause instead of on every keystroke: each save
        # rewrites the YAML, invalidates the config tree cache (a rescan of every config on the
        # next lookup) and triggers a pipeline compose.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(400)
        self._autosave_timer.timeout.connect(self.flush_pending_save)
        self._pending_new_method = False
        self._pending_new_method_spec = {}
        self.spin_timesteps = SpinBox()
        self.spin_seed = SpinBox()
        self.spin_eval_ep = SpinBox()
        self.txt_description = QLineEdit()
        self.chk_tb = None
        self.chk_save_ds = None
        self.chk_recover = None
        self.chk_no_plot = None
        self.view_mode = "resolved" if (settings_manager and settings_manager.show_resolved_defaults) else "overrides"
        if self.settings_manager:
            self.settings_manager.changed.connect(self._on_settings_changed)

        self._init_ui()

    def _on_settings_changed(self):
        if self.settings_manager:
            new_mode = "resolved" if self.settings_manager.show_resolved_defaults else "overrides"
            if new_mode != self.view_mode:
                self.view_mode = new_mode
                self.chk_view_mode.blockSignals(True)
                self.chk_view_mode.setChecked(new_mode == "resolved")
                self.chk_view_mode.blockSignals(False)
            self._render_boxes()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # Header Info Bar
        self.header_bar = QWidget()
        hb_layout = QHBoxLayout(self.header_bar)
        hb_layout.setContentsMargins(12, 6, 12, 6)
        hb_layout.setSpacing(10)

        self.group_badge = TitleLabel("")
        self.group_badge.setObjectName("heading")
        self.group_badge.setToolTip("Group (click to rename)")
        self.group_badge.clicked.connect(self._start_group_edit)
        self.group_badge.double_clicked.connect(self._start_group_edit)
        self.group_badge.hide()
        hb_layout.addWidget(self.group_badge)

        self.group_editor = QLineEdit()
        self.group_editor.setObjectName("groupEditor")
        self.group_editor.setStyleSheet(
            "font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 4px;"
        )
        self.group_editor.setMinimumWidth(120)
        self.group_editor.setMaximumWidth(300)
        self.group_editor.hide()
        self.group_editor.returnPressed.connect(self._commit_group_edit)
        self.group_editor.installEventFilter(self)
        hb_layout.addWidget(self.group_editor)

        self.breadcrumb_sep = QLabel("/")
        self.breadcrumb_sep.setObjectName("breadcrumbSep")
        self.breadcrumb_sep.setStyleSheet(
            "font-size: 20px; font-weight: 500; color: #888888; padding: 0 4px;"
        )
        self.breadcrumb_sep.hide()
        hb_layout.addWidget(self.breadcrumb_sep)

        self.file_title = TitleLabel("No config loaded")
        self.file_title.clicked.connect(self._start_title_edit)
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

        self.btn_edit_code = _make_subtle_tool_button(
            "pencil", "Open component source code in editor", size=26, icon_size=15
        )
        self.btn_edit_code.setObjectName("btnEditCode")
        self.btn_edit_code.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_code.hide()
        self.btn_edit_code.clicked.connect(self._on_edit_code_clicked)
        hb_layout.addWidget(self.btn_edit_code)

        self.btn_export_component = _make_subtle_tool_button(
            "export", "Export component package for Theta-Hub or GitHub", size=26, icon_size=15
        )
        self.btn_export_component.setObjectName("btnExportComponent")
        self.btn_export_component.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_component.hide()
        self.btn_export_component.clicked.connect(self._on_export_component_clicked)
        hb_layout.addWidget(self.btn_export_component)

        self.btn_delete_component = _make_subtle_tool_button(
            "trash", "Uninstall component and all its files", size=26, icon_size=15
        )
        self.btn_delete_component.setObjectName("btnDeleteComponent")
        self.btn_delete_component.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_component.hide()
        self.btn_delete_component.clicked.connect(self._on_delete_component_clicked)
        hb_layout.addWidget(self.btn_delete_component)

        # Maintained for programmatic and test compatibility; hidden from visual header
        self.chk_view_mode = QCheckBox("Show Resolved Defaults")
        self.chk_view_mode.setObjectName("chkViewMode")
        self.chk_view_mode.setChecked(self.view_mode == "resolved")
        self.chk_view_mode.toggled.connect(self._on_view_mode_toggled)
        self.chk_view_mode.hide()

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
        elif obj is getattr(self, "group_editor", None):
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_group_edit()
                    return True
            elif event.type() == QEvent.Type.FocusOut:
                if getattr(self, "_is_editing_group", False) and not getattr(self, "_committing_group_rename", False):
                    grp = self._group_name() or ""
                    raw_text = self.group_editor.text().strip()
                    if raw_text.startswith("[") and raw_text.endswith("]"):
                        raw_text = raw_text[1:-1].strip()
                    if raw_text and raw_text != grp:
                        self._commit_group_edit()
                    else:
                        self._cancel_group_edit()
                return False
        return super().eventFilter(obj, event)

    def _start_group_edit(self, event=None):
        grp = self._group_name()
        if not grp or not self.current_path or not self.current_rel_path:
            return
        self._is_editing_group = True
        self.group_badge.hide()
        self.group_editor.setText(grp)
        self.group_editor.show()
        self.group_editor.setFocus()
        self.group_editor.selectAll()

    def _cancel_group_edit(self):
        self._is_editing_group = False
        self.group_editor.hide()
        self.group_badge.show()

    def _commit_group_edit(self):
        if not getattr(self, "_is_editing_group", False) or getattr(self, "_committing_group_rename", False):
            return
        self._committing_group_rename = True
        try:
            old_grp = self._group_name()
            new_grp = self.group_editor.text().strip()
            if new_grp.startswith("[") and new_grp.endswith("]"):
                new_grp = new_grp[1:-1].strip()

            if not new_grp or new_grp == old_grp:
                self._cancel_group_edit()
                return

            invalid_chars = set('/\\:*?"<>|')
            if any(c in invalid_chars for c in new_grp):
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "The group name cannot contain invalid characters: / \\ : * ? \" < > |",
                )
                self._cancel_group_edit()
                return

            if self.current_path and self.current_rel_path:
                rel_parts_count = len(Path(self.current_rel_path).parts)
                cur = self.current_path
                for _ in range(rel_parts_count):
                    cur = cur.parent
                config_root = cur
            else:
                tree = config_tree()
                if not tree or not hasattr(tree, "config_root"):
                    from .config_tree import find_config_root
                    config_root = find_config_root()
                else:
                    config_root = tree.config_root

            old_group_dir = config_root / "experiment" / old_grp
            if not old_group_dir.exists() or not old_group_dir.is_dir():
                QMessageBox.warning(self, "Rename Failed", f"Group folder '{old_grp}' does not exist on disk.")
                self._cancel_group_edit()
                return

            new_group_dir = config_root / "experiment" / new_grp
            if new_group_dir.exists() and old_group_dir.resolve() != new_group_dir.resolve():
                QMessageBox.warning(
                    self,
                    "Folder Already Exists",
                    f"A group folder named '{new_grp}' already exists.",
                )
                self._cancel_group_edit()
                return

            if old_group_dir.name.lower() == new_group_dir.name.lower() and old_group_dir.name != new_group_dir.name:
                temp_dir = old_group_dir.with_name(f"{old_group_dir.name}__theta_tmp_grp")
                old_group_dir.rename(temp_dir)
                temp_dir.rename(new_group_dir)
            else:
                old_group_dir.rename(new_group_dir)

            rel_parts = list(Path(str(self.current_rel_path)).parts)
            if len(rel_parts) >= 2 and rel_parts[0] == "experiment":
                rel_parts[1] = new_grp
            new_rel = "/".join(rel_parts)
            old_rel = self.current_rel_path

            self.current_rel_path = new_rel
            self.current_path = config_root / new_rel

            self._is_editing_group = False
            self.group_editor.hide()
            self.group_badge.setText(str(new_grp))
            self.group_badge.show()

            self.file_renamed.emit(old_rel, new_rel)
            self.dir_renamed.emit(f"experiment/{old_grp}", f"experiment/{new_grp}")
            self.group_renamed.emit(old_grp, new_grp)
            self.config_changed.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Rename Failed", f"Could not rename group:\n{exc}")
            self._cancel_group_edit()
        finally:
            self._committing_group_rename = False

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

    def _on_edit_code_clicked(self):
        if self.current_path:
            self.edit_code_requested.emit(self.current_path, self.current_rel_path or "")

    def _on_export_component_clicked(self):
        if self.current_path:
            self.export_component_requested.emit(self.current_path, self.current_rel_path or "")

    def _on_delete_component_clicked(self):
        if self.current_path:
            self.delete_component_requested.emit(self.current_path, self.current_rel_path or "")

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
            self.file_title.setToolTip(f"{new_rel}\n(Click to rename)")
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

    def reload_data(self, force_tree_refresh: bool = True):
        """Reload the config and refresh all visual cards/dropdowns."""
        if force_tree_refresh:
            invalidate_config_tree()
        if self.current_path and self.current_path.exists():
            if not self.is_dirty:
                self.load_file(self.current_path, self.current_rel_path or "")
            else:
                self._render_boxes()
        elif self.raw_data:
            self._render_boxes()

    def load_file(self, file_path: Path, rel_path: str):
        """Parse YAML file and render visual boxes."""
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

        self.current_path = Path(file_path)
        self.current_rel_path = rel_path
        self.is_dirty = False
        self._pending_new_method = False
        self._pending_new_method_spec = {}

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
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    sw = sub.widget()
                    if sw:
                        sw.setParent(None)
                        sw.deleteLater()

        # Update header
        group = self._group_name()
        is_base = (self.current_path.stem == "_base") if self.current_path else False

        if is_base:
            self.group_badge.setText(str(group) if group else "group")
            self.group_badge.setToolTip(f"{self.current_rel_path}\n(Base configuration for {group})")
            self.group_badge.show()
            self.breadcrumb_sep.hide()
            self.file_title.setText("Group Defaults")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Base configuration for {group})")
        elif group:
            rel_parts = Path(str(self.current_rel_path or "")).parts
            if len(rel_parts) > 3:
                sub_crumbs = " / ".join(rel_parts[2:-1])
                self.group_badge.setText(f"{group} / {sub_crumbs}")
            else:
                self.group_badge.setText(str(group))
            self.group_badge.setToolTip(f"Group: {group}\n(Click to rename group)")
            self.group_badge.show()
            self.breadcrumb_sep.show()
            self.file_title.setText(self.current_path.stem)
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Click to rename)")
        else:
            self.group_badge.hide()
            self.breadcrumb_sep.hide()
            self.file_title.setText(self.current_path.stem if self.current_path else "")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Click to rename)")

        if getattr(self, "is_component_mode", False) and self.current_path:
            self.btn_edit_code.show()
            self.btn_export_component.show()
            self.btn_delete_component.show()
        else:
            self.btn_edit_code.hide()
            self.btn_export_component.hide()
            self.btn_delete_component.hide()


        is_experiment = (
            (self.current_rel_path and self.current_rel_path.startswith("experiment/") and "/methods/" not in self.current_rel_path) or
            "methods" in self.raw_data or
            any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )

        if self.settings_manager:
            self.view_mode = "resolved" if self.settings_manager.show_resolved_defaults else "overrides"
            self.chk_view_mode.blockSignals(True)
            self.chk_view_mode.setChecked(self.view_mode == "resolved")
            self.chk_view_mode.blockSignals(False)

        if is_experiment:
            self._render_experiment_boxes()
        else:
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

    def _is_base_config(self) -> bool:
        """True if the currently open configuration is a group defaults (_base.yaml) file."""
        return bool(self.current_path and self.current_path.stem == "_base")

    def _get_explicit_env(self) -> str | None:
        """The environment explicitly overridden in defaults or raw_data."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return None
        for entry in self.raw_data.get("defaults") or []:
            if isinstance(entry, dict):
                if "override /env" in entry:
                    val = entry.get("override /env")
                    return str(val) if val is not None else None
                elif "/env" in entry:
                    val = entry.get("/env")
                    return str(val) if val is not None else None
        val = self.raw_data.get("env")
        return str(val) if isinstance(val, str) else None

    def _set_experiment_env(self, env_name: str | None):
        """Update or insert override /env in defaults, removing legacy scalar env."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return
        self.raw_data.pop("env", None)
        defaults = self.raw_data.get("defaults")

        if env_name is None or env_name == self.INHERIT or env_name in ("", "— Select Environment —"):
            if isinstance(defaults, list):
                self.raw_data["defaults"] = [
                    d for d in defaults
                    if not (isinstance(d, dict) and ("override /env" in d or "/env" in d))
                ]
            self._mark_dirty()
            return

        if not isinstance(defaults, list):
            defaults = []
            self.raw_data["defaults"] = defaults

        found = False
        for entry in defaults:
            if isinstance(entry, dict) and ("override /env" in entry or "/env" in entry):
                entry.clear()
                entry["override /env"] = env_name
                found = True
                break
        if not found:
            defaults.append({"override /env": env_name})
        self._mark_dirty()

    def _get_base_env(self) -> str | None:
        """The environment configured in a group's _base.yaml defaults."""
        return self._get_explicit_env()

    def _set_base_env(self, env_name: str | None):
        """Update or insert override /env in the group's _base.yaml defaults."""
        self._set_experiment_env(env_name)

    def _clean_redundant_params(self):
        """Strip redundant parameters inherited from defaults or matching filename stem."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return
        is_base = self._is_base_config()
        if is_base:
            self.raw_data.pop("group", None)
            self.raw_data.pop("experiment_id", None)
            # A group base may declare one default method, which new experiments inherit, so its
            # concrete methods are kept. (They used to be stripped here, on load and on save, which
            # deleted the default method from disk the first time the base was opened and saved.)
            # The one-method limit is enforced where methods are added.
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                if not self.raw_data["methods"]:
                    self.raw_data.pop("methods", None)
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
            inh_paradigm = self._inherited_paradigm()
            if inh_paradigm is not None and self.raw_data.get("paradigm") == inh_paradigm:
                self.raw_data.pop("paradigm", None)
            inh_env = self._inherited_env()
            explicit_env = self._get_explicit_env()
            if explicit_env is not None and (
                (inh_env is not None and explicit_env == inh_env) or explicit_env == self.INHERIT
            ):
                self._set_experiment_env(None)
            elif isinstance(self.raw_data.get("env"), str):
                self._set_experiment_env(self.raw_data.get("env"))
            if "description" in self.raw_data and not self.raw_data["description"]:
                self.raw_data.pop("description", None)
            if "resources" in self.raw_data and not self.raw_data["resources"]:
                self.raw_data.pop("resources", None)
            if "trainer" in self.raw_data and not self.raw_data["trainer"]:
                self.raw_data.pop("trainer", None)
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                if "params" in self.raw_data["methods"] and not self.raw_data["methods"]["params"]:
                    self.raw_data["methods"].pop("params", None)
            # Strip redundant default flags matching root/base defaults
            if self.raw_data.get("tensorboard") is False:
                self.raw_data.pop("tensorboard", None)
            if self.raw_data.get("save_dataset") is False:
                self.raw_data.pop("save_dataset", None)
            if self.raw_data.get("recover") is False:
                self.raw_data.pop("recover", None)
            if self.raw_data.get("no_plot") is False:
                self.raw_data.pop("no_plot", None)

    def _cap_form_fields(self):
        """Limit field widths in every label/field form and two-column grid."""
        for grid in self.container.findChildren(QGridLayout):
            if isinstance(grid.parentWidget(), MethodCardsContainer):
                continue
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

        is_base = (self.current_path.stem == "_base") if self.current_path else False

        # Paradigm + Environment, driven by in/config/paradigms constraints.
        tree = config_tree()
        if is_base:
            box_base = ConfigBox("1. Group Paradigm & Environment", "Default paradigm and environment applied to all experiments in this group")
            form_base = QGridLayout()
            form_base.setVerticalSpacing(10)
            form_base.setHorizontalSpacing(16)

            form_base.addWidget(label("Paradigm"), 0, 0)
            self.combo_paradigm = ComboBox()
            self.combo_paradigm.addItem("— Select Paradigm —", None)
            if tree and tree.paradigms:
                self.combo_paradigm.addItems(sorted(tree.paradigms))
            base_p = data.get("paradigm")
            if base_p and tree and base_p in tree.paradigms:
                self.combo_paradigm.setCurrentText(base_p)
            else:
                self.combo_paradigm.setCurrentIndex(0)
            form_base.addWidget(self.combo_paradigm, 0, 1)

            form_base.addWidget(label("Environment"), 0, 2)
            self.combo_env = ComboBox()
            self.combo_env.addItem("— Select Environment —", None)

            def _refresh_base_env_choices(selected_p):
                curr_text = self.combo_env.currentText()
                self.combo_env.blockSignals(True)
                self.combo_env.clear()
                self.combo_env.addItem("— Select Environment —", None)
                if selected_p and tree and selected_p in tree.paradigms:
                    allowed = [e.name for e in tree.environments_for(selected_p)]
                elif tree:
                    allowed = sorted(list(set(e.name for e in tree.environments.values())))
                else:
                    allowed = []
                self.combo_env.addItems(allowed)
                if curr_text and curr_text in allowed:
                    self.combo_env.setCurrentText(curr_text)
                else:
                    base_e = self._get_base_env()
                    if base_e:
                        if base_e not in allowed:
                            self.combo_env.addItem(base_e, base_e)
                        self.combo_env.setCurrentText(base_e)
                    elif curr_text and curr_text not in ("", "— Select Environment —"):
                        self.combo_env.addItem(curr_text, curr_text)
                        self.combo_env.setCurrentText(curr_text)
                    else:
                        self.combo_env.setCurrentIndex(0)
                self.combo_env.blockSignals(False)

            base_env = self._get_base_env()
            if base_p and tree and base_p in tree.paradigms:
                allowed_init = [e.name for e in tree.environments_for(base_p)]
            elif tree:
                allowed_init = sorted(list(set(e.name for e in tree.environments.values())))
            else:
                allowed_init = []
            self.combo_env.addItems(allowed_init)
            if base_env:
                if base_env not in allowed_init:
                    self.combo_env.addItem(base_env, base_env)
                    self.combo_env.setToolTip(f"Missing component: Environment '{base_env}' is not installed in in/config/env/. Install it via Component Hub.")
                self.combo_env.setCurrentText(base_env)
            else:
                self.combo_env.setCurrentIndex(0)
            form_base.addWidget(self.combo_env, 0, 3)

            def _on_base_p_changed(p_text):
                val = None if p_text in ("", "— Select Paradigm —") else p_text
                self.raw_data["paradigm"] = val
                _refresh_base_env_choices(val)
                self._mark_dirty()

            def _on_base_e_changed(e_text):
                val = None if e_text in ("", "— Select Environment —") else e_text
                self._set_base_env(val)

            self.combo_paradigm.currentTextChanged.connect(_on_base_p_changed)
            self.combo_env.currentTextChanged.connect(_on_base_e_changed)

            self.field_widgets["paradigm"] = self.combo_paradigm
            self.field_widgets["env"] = self.combo_env

            box_base.add_layout(form_base)
            self.boxes_layout.addWidget(box_base)
        else:
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
                explicit_env = self._get_explicit_env()
                if explicit_env and explicit_env != inherited_env:
                    if explicit_env not in allowed_envs:
                        self.combo_env.addItem(explicit_env, explicit_env)
                        self.combo_env.setToolTip(f"Missing component: Environment '{explicit_env}' is not installed in in/config/env/. Install it via Component Hub.")
                    self.combo_env.setCurrentText(explicit_env)
                else:
                    self.combo_env.setCurrentIndex(0)
                self.combo_env.setToolTip(
                    f"Environments for {self.combo_paradigm.currentText()}"
                )
                self.combo_env.currentTextChanged.connect(
                    lambda v: self._on_experiment_env_changed(v)
                )
                self.field_widgets["env"] = self.combo_env

        # --- BOX 1: TRAINING BUDGET & ROLLOUT SCHEDULE ---
        budget_num = "2" if is_base else "1"
        box_budget = ConfigBox(f"{budget_num}. Training Budget & Schedule", "Timesteps, evaluation frequency, and random seeds")

        # Prompt rows if experiment requires paradigm or env (not provided by group defaults)
        needs_paradigm_prompt = not is_base and "paradigm" in data
        needs_env_prompt = not is_base and (self._get_explicit_env() is not None or "env" in data)
        if needs_paradigm_prompt or needs_env_prompt:
            prompt_grid = QGridLayout()
            prompt_grid.setVerticalSpacing(8)
            prompt_grid.setHorizontalSpacing(16)
            col = 0
            if needs_paradigm_prompt:
                if not data.get("paradigm") or data.get("paradigm") not in tree.paradigms:
                    if self.combo_paradigm.findText("— Select Paradigm —") == -1:
                        self.combo_paradigm.insertItem(0, "— Select Paradigm —", None)
                        self.combo_paradigm.setCurrentIndex(0)
                prompt_grid.addWidget(label("Paradigm", "accent"), 0, col)
                prompt_grid.addWidget(self.combo_paradigm, 0, col + 1)
                col += 2
            if needs_env_prompt:
                if "env" in data:
                    exp_env = self._get_explicit_env()
                    if not exp_env:
                        if self.combo_env.findText("— Select Environment —") == -1:
                            self.combo_env.insertItem(1, "— Select Environment —", None)
                            self.combo_env.setCurrentIndex(1)
                prompt_grid.addWidget(label("Environment", "accent"), 0, col)
                prompt_grid.addWidget(self.combo_env, 0, col + 1)
                col += 2
            box_budget.add_layout(prompt_grid)

        # Experiment Description / Commit Note
        desc_row = QHBoxLayout()
        desc_row.setSpacing(10)
        desc_row.addWidget(label("Description"))
        self.txt_description = QLineEdit()
        self.txt_description.setMaximumWidth(520)
        self.txt_description.setPlaceholderText("Optional commit-style notes or hypothesis for this experiment...")
        self.txt_description.setText(str(data.get("description") or ""))
        self.txt_description.textChanged.connect(
            lambda v: self._on_field_edited("description", v.strip()) if v.strip() else self.raw_data.pop("description", None)
        )
        desc_row.addWidget(self.txt_description)
        if "description" in data and data["description"]:
            btn_del_desc = _make_subtle_tool_button("close", "Clear description override", size=20, icon_size=12)
            btn_del_desc.clicked.connect(lambda: (self.txt_description.setText(""), self.raw_data.pop("description", None), self._mark_dirty(), self._render_boxes()))
            desc_row.addWidget(btn_del_desc)
        desc_row.addStretch()
        self.field_widgets["description"] = self.txt_description
        box_budget.add_layout(desc_row)

        form_budget = QGridLayout()
        form_budget.setVerticalSpacing(10)
        form_budget.setHorizontalSpacing(16)

        # Baseline budget widget instantiation
        self.spin_timesteps = SpinBox()
        self.spin_timesteps.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_timesteps.setRange(100, 100_000_000)
        self.spin_timesteps.setSingleStep(1_000)
        self.spin_timesteps.setValue(int(data.get("total_timesteps") or resolved.get("total_timesteps") or 10000))
        self.spin_timesteps.valueChanged.connect(lambda v: self._on_field_edited("total_timesteps", v))
        self.field_widgets["total_timesteps"] = self.spin_timesteps

        self.spin_seed = SpinBox()
        self.spin_seed.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(int(data.get("seed") if data.get("seed") is not None else resolved.get("seed", 42)))
        self.spin_seed.valueChanged.connect(lambda v: self._on_field_edited("seed", v))
        self.field_widgets["seed"] = self.spin_seed

        self.spin_eval_ep = SpinBox()
        self.spin_eval_ep.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_eval_ep.setRange(0, 1000)
        self.spin_eval_ep.setValue(int(data.get("eval_episodes") if data.get("eval_episodes") is not None else resolved.get("eval_episodes", 100)))
        self.spin_eval_ep.valueChanged.connect(lambda v: self._on_field_edited("eval_episodes", v))
        self.field_widgets["eval_episodes"] = self.spin_eval_ep

        eval_ep_enabled = True
        is_online = (paradigm == "online_rl")
        if tree and paradigm in tree.paradigms:
            rules = tree.paradigms[paradigm]
            base = _read_group_base(tree, self._group_name())
            if not rules.field_enabled("eval_episodes"):
                eval_ep_enabled = False
                self.spin_eval_ep.setEnabled(False)
                if "eval_episodes" in base:
                    self.spin_eval_ep.setValue(int(base["eval_episodes"]))
                    self.spin_eval_ep.setSuffix(f"   (fixed by {paradigm})")
                else:
                    self.spin_eval_ep.setSuffix(f"   (not used by {paradigm})")
                self.spin_eval_ep.setToolTip(rules.disabled_reason("eval_episodes"))

        show_timesteps = is_online and ("total_timesteps" in data or self.view_mode == "resolved")
        show_seed = ("seed" in data or self.view_mode == "resolved")
        show_eval_ep = is_online and ("eval_episodes" in data or not eval_ep_enabled or self.view_mode == "resolved")
        show_eval_freq = is_online and ("eval_freq_steps" in data or "eval_freq" in data or (self.view_mode == "resolved" and ("eval_freq_steps" in resolved or "eval_freq" in resolved)))
        show_epochs = not is_online and ("epochs" in data or self.view_mode == "resolved" or "epochs" in resolved)
        show_eval_every = not is_online and ("eval_every_n_epochs" in data or self.view_mode == "resolved" or "eval_every_n_epochs" in resolved)

        # Row layout for epochs
        self.spin_epochs = SpinBox()
        self.spin_epochs.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_epochs.setRange(1, 100000)
        self.spin_epochs.setValue(int(data.get("epochs") or resolved.get("epochs") or 10))
        self.spin_epochs.valueChanged.connect(lambda v: self._on_field_edited("epochs", v))
        self.field_widgets["epochs"] = self.spin_epochs

        row_epochs = QHBoxLayout()
        row_epochs.setSpacing(6)
        row_epochs.addWidget(self.spin_epochs)
        if "epochs" in data:
            btn_del_ep = _make_subtle_tool_button("close", "Remove epochs override to rely on group default", size=20, icon_size=12)
            btn_del_ep.clicked.connect(lambda: (self.raw_data.pop("epochs", None), self._mark_dirty(), self._render_boxes()))
            row_epochs.addWidget(btn_del_ep)
        else:
            row_epochs.addWidget(label("(inherited)", "muted"))
        row_epochs.addStretch()

        # Row layout for eval_every_n_epochs
        self.spin_eval_every = SpinBox()
        self.spin_eval_every.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_eval_every.setRange(1, 1000)
        self.spin_eval_every.setValue(int(data.get("eval_every_n_epochs") or resolved.get("eval_every_n_epochs") or 1))
        self.spin_eval_every.valueChanged.connect(lambda v: self._on_field_edited("eval_every_n_epochs", v))
        self.field_widgets["eval_every_n_epochs"] = self.spin_eval_every

        row_eval_every = QHBoxLayout()
        row_eval_every.setSpacing(6)
        row_eval_every.addWidget(self.spin_eval_every)
        if "eval_every_n_epochs" in data:
            btn_del_ee = _make_subtle_tool_button("close", "Remove eval_every_n_epochs override to rely on group default", size=20, icon_size=12)
            btn_del_ee.clicked.connect(lambda: (self.raw_data.pop("eval_every_n_epochs", None), self._mark_dirty(), self._render_boxes()))
            row_eval_every.addWidget(btn_del_ee)
        else:
            row_eval_every.addWidget(label("(inherited)", "muted"))
        row_eval_every.addStretch()

        # Row layout for timesteps
        row_timesteps = QHBoxLayout()
        row_timesteps.setSpacing(6)
        row_timesteps.addWidget(self.spin_timesteps)
        if "total_timesteps" in data:
            btn_del_tt = _make_subtle_tool_button("close", "Remove total_timesteps override to rely on group default", size=20, icon_size=12)
            btn_del_tt.clicked.connect(lambda: (self.raw_data.pop("total_timesteps", None), self._mark_dirty(), self._render_boxes()))
            row_timesteps.addWidget(btn_del_tt)
        else:
            row_timesteps.addWidget(label("(inherited)", "muted"))
        row_timesteps.addStretch()

        # Row layout for seed
        row_seed = QHBoxLayout()
        row_seed.setSpacing(6)
        row_seed.addWidget(self.spin_seed)
        btn_rand_seed = QToolButton()
        btn_rand_seed.setText("Random")
        btn_rand_seed.setToolTip("Pick a random seed")
        btn_rand_seed.clicked.connect(lambda: self.spin_seed.setValue(random.randint(1, 9999)))
        row_seed.addWidget(btn_rand_seed)
        if "seed" in data:
            btn_del_seed = _make_subtle_tool_button("close", "Remove seed override to rely on group default", size=20, icon_size=12)
            btn_del_seed.clicked.connect(lambda: (self.raw_data.pop("seed", None), self._mark_dirty(), self._render_boxes()))
            row_seed.addWidget(btn_del_seed)
        else:
            row_seed.addWidget(label("(inherited)", "muted"))
        row_seed.addStretch()

        # Row layout for eval_episodes
        row_eval_ep = QHBoxLayout()
        row_eval_ep.setSpacing(6)
        row_eval_ep.addWidget(self.spin_eval_ep)
        if "eval_episodes" in data and eval_ep_enabled:
            btn_del_ee = _make_subtle_tool_button("close", "Remove eval_episodes override to rely on group default", size=20, icon_size=12)
            btn_del_ee.clicked.connect(lambda: (self.raw_data.pop("eval_episodes", None), self._mark_dirty(), self._render_boxes()))
            row_eval_ep.addWidget(btn_del_ee)
        elif eval_ep_enabled:
            row_eval_ep.addWidget(label("(inherited)", "muted"))
        row_eval_ep.addStretch()

        # Row layout for eval frequency (steps)
        row_eval_freq = None
        if show_eval_freq:
            freq_key = "eval_freq_steps" if ("eval_freq_steps" in data or "eval_freq_steps" in resolved) else "eval_freq"
            spin_eval_freq = SpinBox()
            spin_eval_freq.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_eval_freq.setRange(10, 100_000_000)
            spin_eval_freq.setSingleStep(1_000)
            spin_eval_freq.setValue(int(data.get(freq_key) or resolved.get(freq_key) or 10000))
            spin_eval_freq.valueChanged.connect(lambda v, fk=freq_key: self._on_field_edited(fk, v))
            self.field_widgets[freq_key] = spin_eval_freq
            row_eval_freq = QHBoxLayout()
            row_eval_freq.setSpacing(6)
            row_eval_freq.addWidget(spin_eval_freq)
            if freq_key in data:
                btn_del_ef = _make_subtle_tool_button("close", f"Remove '{freq_key}' override to rely on group default", size=20, icon_size=12)
                btn_del_ef.clicked.connect(lambda _, fk=freq_key: (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes()))
                row_eval_freq.addWidget(btn_del_ef)
            else:
                row_eval_freq.addWidget(label("(inherited)", "muted"))
            row_eval_freq.addStretch()

        # Extra custom experiment parameters (top-level scalars)
        standard_top_level_keys = {
            "defaults", "paradigm", "experiment_id", "description",
            "seed", "total_timesteps", "eval_episodes", "eval_freq_steps", "eval_freq",
            "epochs", "eval_every_n_epochs",
            "methods", "env", "site", "trainer", "resources", "dataset", "dataset_path",
            "tensorboard", "no_plot", "save_dataset", "recover",
            "hydra", "plots", "tuning", "sweeper"
        }
        extra_experiment_items = []
        for extra_k, extra_v in list(data.items()):
            if extra_k in standard_top_level_keys:
                continue
            if isinstance(extra_v, (int, float, str, bool)):
                row_extra = QHBoxLayout()
                row_extra.setSpacing(6)
                if isinstance(extra_v, bool):
                    chk_extra = ToggleSlider(checked=extra_v)
                    chk_extra.toggled.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(chk_extra)
                elif isinstance(extra_v, (int, float)):
                    spin_extra = number_field(extra_v)
                    spin_extra.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_extra.valueChanged.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(spin_extra)
                else:
                    line_extra = QLineEdit(str(extra_v if extra_v is not None else ""))
                    line_extra.setMaximumWidth(TEXT_FIELD_WIDTH)
                    line_extra.textChanged.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(line_extra)
                btn_del_extra = _make_subtle_tool_button("close", f"Remove '{extra_k}' parameter", size=20, icon_size=12)
                btn_del_extra.clicked.connect(lambda _, ek=extra_k: (self.raw_data.pop(ek, None), self._mark_dirty(), self._render_boxes()))
                row_extra.addWidget(btn_del_extra)
                row_extra.addStretch()
                extra_experiment_items.append((label(str(extra_k)), row_extra))

        grid_items = []
        if is_online:
            if show_timesteps:
                grid_items.append((label("Total Timesteps"), row_timesteps))
            if show_seed:
                grid_items.append((label("Random Seed"), row_seed))
            if show_eval_ep:
                grid_items.append((label("Eval Episodes"), row_eval_ep))
            if show_eval_freq and row_eval_freq:
                grid_items.append((label("Eval Frequency"), row_eval_freq))
        else:
            if show_epochs:
                grid_items.append((label("Epochs"), row_epochs))
            if show_seed:
                grid_items.append((label("Random Seed"), row_seed))
            if show_eval_every:
                grid_items.append((label("Eval Frequency (Epochs)"), row_eval_every))
        for extra_lbl, extra_row in extra_experiment_items:
            grid_items.append((extra_lbl, extra_row))

        curr_row = 0
        for idx, (lbl_widget, val_layout) in enumerate(grid_items):
            c_col = 0 if (idx % 2 == 0) else 2
            c_row = idx // 2
            form_budget.addWidget(lbl_widget, c_row, c_col)
            form_budget.addLayout(val_layout, c_row, c_col + 1)
            curr_row = max(curr_row, c_row + 1)

        # Runtime & Logging Options (TensorBoard, Save Dataset, Recover, No Plots)
        OPTIONAL_RUNTIME_FLAGS = [
            ("tensorboard", "Log to TensorBoard"),
            ("save_dataset", "Save Dataset"),
            ("recover", "Recover Checkpoint"),
            ("no_plot", "No Plots"),
        ]

        chk_row = QHBoxLayout()
        chk_row.setSpacing(14)
        chk_widgets = {}

        # Show checkboxes only for flags explicitly present in the experiment config (or when showing resolved defaults)
        active_flags = [
            (k, lbl) for k, lbl in OPTIONAL_RUNTIME_FLAGS
            if k in data or self.view_mode == "resolved"
        ]

        for flag_key, flag_label in active_flags:
            row_flag = QHBoxLayout()
            row_flag.setSpacing(6)
            chk = ToggleSlider(checked=bool(data.get(flag_key, resolved.get(flag_key, False))))
            chk.toggled.connect(
                lambda v, fk=flag_key: (
                    self._on_field_edited(fk, True)
                    if v
                    else (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes())
                )
            )
            row_flag.addWidget(chk)
            row_flag.addWidget(label(flag_label))
            chk_widgets[flag_key] = chk

            if flag_key in data:
                btn_del_flag = _make_subtle_tool_button("close", f"Remove '{flag_key}' override", size=18, icon_size=11)
                btn_del_flag.clicked.connect(
                    lambda _, fk=flag_key: (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes())
                )
                row_flag.addWidget(btn_del_flag)
            else:
                row_flag.addWidget(label("(inherited)", "muted"))

            chk_row.addLayout(row_flag)

        self.chk_tb = chk_widgets.get("tensorboard")
        self.chk_save_ds = chk_widgets.get("save_dataset")
        self.chk_recover = chk_widgets.get("recover")
        self.chk_no_plot = chk_widgets.get("no_plot")

        # Option adder button for any unadded experiment parameter or flag
        if self.view_mode != "resolved":
            btn_add_opt = _make_subtle_tool_button("plus", "Add option", size=24, icon_size=14)
            btn_add_opt.setObjectName("addOptionButton")
            btn_add_opt.setStyleSheet("""
                QToolButton#addOptionButton {
                    background: transparent;
                    border: 1px dashed #504945;
                    border-radius: 4px;
                    padding: 0px;
                }
                QToolButton#addOptionButton:hover {
                    background: #32302f;
                    border-color: #a89984;
                }
                QToolButton#addOptionButton::menu-indicator {
                    image: none;
                    width: 0px;
                }
            """)
            btn_add_opt.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            opt_menu = QMenu(btn_add_opt)


            # 1. Budget & Timing Candidates
            budget_cands = []
            if is_online:
                if "total_timesteps" not in data:
                    budget_cands.append(("total_timesteps", "Total Timesteps", int(resolved.get("total_timesteps") or 10000)))
                if "seed" not in data:
                    budget_cands.append(("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)))
                if eval_ep_enabled and "eval_episodes" not in data:
                    budget_cands.append(("eval_episodes", "Eval Episodes", int(resolved.get("eval_episodes") or 100)))
                if "eval_freq_steps" not in data and "eval_freq" not in data:
                    budget_cands.append(("eval_freq_steps", "Eval Frequency (Steps)", int(resolved.get("eval_freq_steps") or 10000)))
            else:
                if "epochs" not in data:
                    budget_cands.append(("epochs", "Epochs", int(resolved.get("epochs") or 10)))
                if "seed" not in data:
                    budget_cands.append(("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)))
                if "eval_every_n_epochs" not in data:
                    budget_cands.append(("eval_every_n_epochs", "Eval Frequency (Epochs)", int(resolved.get("eval_every_n_epochs") or 1)))

            for bk, blbl, bdef in budget_cands:
                act = opt_menu.addAction(blbl)
                act.triggered.connect(lambda _, k=bk, d=bdef: self._add_experiment_param(k, d))

            # 2. Runtime Flags
            unadded_flags = [(k, lbl) for k, lbl in OPTIONAL_RUNTIME_FLAGS if k not in data]
            if unadded_flags:
                if budget_cands:
                    opt_menu.addSeparator()
                for fk, flbl in unadded_flags:
                    act = opt_menu.addAction(flbl)
                    act.triggered.connect(lambda _, k=fk: self._add_experiment_param(k, True))

            # 3. Custom Parameter
            opt_menu.addSeparator()
            act_custom = opt_menu.addAction("+ Custom Parameter...")
            act_custom.triggered.connect(self._prompt_add_custom_experiment_param)

            btn_add_opt.setMenu(opt_menu)
            chk_row.addWidget(btn_add_opt)

        chk_row.addStretch()
        form_budget.addLayout(chk_row, curr_row, 0, 1, 4)

        box_budget.add_layout(form_budget)
        self.boxes_layout.addWidget(box_budget)

        # --- BOX 2 / 3: METHODS & AGENT ARCHITECTURES ---
        methods_dict = data.get("methods", {})
        methods_num = "3" if is_base else "2"
        box_methods = ConfigBox("Methods", "Configured agents, policies, and hyperparameter overrides")
        m_layout = QVBoxLayout()
        m_layout.setSpacing(10)

        # Universal / Shared Parameters (methods.params)
        has_params = "params" in methods_dict and isinstance(methods_dict["params"], dict)
        resolved_params = resolved.get("methods", {}).get("params", {}) if isinstance(resolved.get("methods"), dict) else {}
        show_params = has_params or (self.view_mode == "resolved" and bool(resolved_params))

        # Top-left Methods header with title and subtle plus button to add shared parameters
        methods_header = QHBoxLayout()
        methods_header.setSpacing(8)
        lbl_methods_title = label("Methods", "heading")
        lbl_methods_title.setStyleSheet("font-size: 23px; font-weight: 600;")
        methods_header.addWidget(lbl_methods_title)

        if not show_params:
            btn_create_params = _make_subtle_tool_button("plus", "Add Shared Parameters (params)", size=24, icon_size=14)
            btn_create_params.clicked.connect(lambda: self.add_block("params"))
            methods_header.addWidget(btn_create_params)

        methods_header.addStretch()
        m_layout.addLayout(methods_header)

        if show_params:
            p_card = QFrame()
            p_card.setObjectName("card")
            p_layout = QVBoxLayout(p_card)
            p_layout.setContentsMargins(14, 12, 14, 12)
            p_layout.setSpacing(8)

            p_header = QHBoxLayout()
            p_header.setSpacing(8)
            lbl_shared = label("Shared Parameters", "cardTitle")
            lbl_shared.setStyleSheet("font-size: 16px; font-weight: 600;")
            p_header.addWidget(lbl_shared)

            if not has_params:
                p_header.addWidget(label("(inherited from base)", "muted"))
            p_header.addStretch()
            if has_params:
                btn_rm_params = _make_subtle_tool_button("trash", "Remove Shared Parameters block", size=24, icon_size=14)
                btn_rm_params.clicked.connect(lambda: self.remove_block("params"))
                p_header.addWidget(btn_rm_params)
            p_layout.addLayout(p_header)

            curr_params = methods_dict.get("params", {}) if has_params else resolved_params

            common_universal = ["lr", "gamma", "epochs", "eval_every_n_epochs", "cql_alpha", "weight_decay", "batch_size", "ent_coef"]
            if tree:
                for a in sorted(tree.agents):
                    common_universal.append(f"agent.{a}")
                for m in sorted(tree.models):
                    common_universal.append(f"model.{m}")
            cand_u = [u for u in common_universal if u not in curr_params]
            btn_add_u = _make_param_adder_button("Add shared parameter", cand_u, lambda pk: self._add_universal_param(pk), parent_widget=self)

            rendered_u_rows = []

            def _render_param_item(k_path: str, val: Any, indent: int = 0):
                if isinstance(val, dict):
                    grp_hdr = QHBoxLayout()
                    grp_hdr.setSpacing(6)
                    if indent > 0:
                        grp_hdr.addSpacing(indent * 14)
                    grp_lbl = label(f"[{k_path}]", "muted")
                    grp_hdr.addWidget(grp_lbl)
                    grp_hdr.addStretch()
                    if has_params:
                        btn_del_grp = _make_subtle_tool_button("close", f"Remove parameter group '{k_path}'", size=20, icon_size=12)
                        btn_del_grp.clicked.connect(lambda _, kp=k_path: self._remove_universal_param(kp))
                        grp_hdr.addWidget(btn_del_grp)
                    p_layout.addLayout(grp_hdr)
                    for sub_k, sub_v in list(val.items()):
                        sub_path = f"{k_path}.{sub_k}" if k_path else sub_k
                        _render_param_item(sub_path, sub_v, indent + 1)
                else:
                    p_row = QHBoxLayout()
                    p_row.setSpacing(8)
                    if indent > 0:
                        p_row.addSpacing(indent * 14)
                    lbl_k = label(str(k_path))
                    lbl_k.setMinimumWidth(160)
                    p_row.addWidget(lbl_k)

                    if isinstance(val, bool):
                        slider_v = ToggleSlider(checked=val)
                        slider_v.toggled.connect(lambda v, kp=k_path: self._on_universal_param_edited(kp, v))
                        p_row.addWidget(slider_v)
                    elif isinstance(val, (int, float)):
                        spin_v = number_field(val)
                        spin_v.setMaximumWidth(NUMBER_FIELD_WIDTH)
                        spin_v.valueChanged.connect(lambda v, kp=k_path: self._on_universal_param_edited(kp, v))
                        p_row.addWidget(spin_v)
                    else:
                        line_v = QLineEdit(str(val if val is not None else ""))
                        line_v.setMaximumWidth(TEXT_FIELD_WIDTH)
                        line_v.textChanged.connect(lambda v, kp=k_path: self._on_universal_param_edited(kp, v))
                        p_row.addWidget(line_v)

                    if has_params:
                        btn_del_p = _make_subtle_tool_button("close", f"Remove parameter '{k_path}'", size=20, icon_size=12)
                        btn_del_p.clicked.connect(lambda _, kp=k_path: self._remove_universal_param(kp))
                        p_row.addWidget(btn_del_p)
                    rendered_u_rows.append(p_row)

            for pk, pv in list(curr_params.items()):
                _render_param_item(pk, pv, indent=0)

            if rendered_u_rows:
                rendered_u_rows[-1].addWidget(btn_add_u)
                for r in rendered_u_rows:
                    r.addStretch()
                    p_layout.addLayout(r)
            else:
                row_empty_p = QHBoxLayout()
                row_empty_p.addWidget(label("No shared parameters configured.", "muted"))
                row_empty_p.addWidget(btn_add_u)
                row_empty_p.addStretch()
                p_layout.addLayout(row_empty_p)

            m_layout.addWidget(p_card)

        has_methods = any(k != "params" and isinstance(v, dict) for k, v in methods_dict.items()) if methods_dict else False
        pending_new = getattr(self, "_pending_new_method", False)

        if is_base and not has_methods and not pending_new:
            note_row = QHBoxLayout()
            note_row.addWidget(label("No default method configured for this group.", "muted"))
            note_row.addStretch()
            btn_add_base_m = _make_subtle_tool_button("plus", "Add default method for this group", size=24, icon_size=14)
            btn_add_base_m.clicked.connect(lambda: self._start_add_new_method())
            note_row.addWidget(btn_add_base_m)
            m_layout.addLayout(note_row)
        elif not is_base and not has_methods and not pending_new:
            empty_row = QHBoxLayout()
            empty_row.addWidget(label("Inherits default method from base template.", "muted"))
            empty_row.addStretch()
            btn_add_first = _make_subtle_tool_button("plus", "Add method", size=24, icon_size=14)
            btn_add_first.clicked.connect(lambda: self._start_add_new_method())
            empty_row.addWidget(btn_add_first)
            m_layout.addLayout(empty_row)
        else:
            # Sub-cards for each method wrapped in MethodCardsContainer for drag reordering
            method_container = MethodCardsContainer(self)

            show_ag_defs = bool(self.settings_manager and (self.settings_manager.show_method_agent_defaults or self.settings_manager.show_all_default_params)) or (self.view_mode == "resolved")
            show_mo_defs = bool(self.settings_manager and (self.settings_manager.show_method_model_defaults or self.settings_manager.show_all_default_params)) or (self.view_mode == "resolved")
            show_all_defs = bool(self.settings_manager and self.settings_manager.show_all_default_params) or (self.view_mode == "resolved")

            for m_name, m_spec in methods_dict.items():
                if m_name == "params" or not isinstance(m_spec, dict):
                    continue
                sub_card = QFrame()
                sub_card.setObjectName("card")
                card_vbox = QVBoxLayout(sub_card)
                card_vbox.setContentsMargins(10, 10, 10, 10)
                card_vbox.setSpacing(8)

                sc_layout = QFormLayout()
                sc_layout.setContentsMargins(0, 0, 0, 0)
                sc_layout.setVerticalSpacing(8)

                title_row = QHBoxLayout()
                title_row.setSpacing(6)

                btn_grip = MethodDragHandle(m_name, sub_card)
                title_row.addWidget(btn_grip)

                m_title = TitleLabel(m_name)
                m_title.setObjectName("cardTitle")
                m_title.setToolTip(f"{m_name}\n(Click to rename)")
                title_row.addWidget(m_title)

                m_edit = InlineTitleEditor(m_name)
                m_edit.setObjectName("methodEditor")
                m_edit.setStyleSheet("font-size: 16px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
                m_edit.hide()
                title_row.addWidget(m_edit)

                committing_m_rename = [False]

                def _start_m_rename(lbl=m_title, edt=m_edit, name=m_name):
                    committing_m_rename[0] = False
                    lbl.hide()
                    edt.setText(name)
                    edt.show()
                    edt.setFocus()
                    edt.selectAll()

                def _cancel_m_rename(lbl=m_title, edt=m_edit):
                    committing_m_rename[0] = True
                    edt.hide()
                    lbl.show()

                def _commit_m_rename(lbl=m_title, edt=m_edit, name=m_name):
                    if committing_m_rename[0]:
                        return
                    committing_m_rename[0] = True
                    new_val = edt.text().strip()
                    if not new_val or new_val == name:
                        edt.hide()
                        lbl.show()
                        return
                    existing_keys = set(self.raw_data.get("methods", {}).keys()) - {"params", name}
                    err = self._validate_method_name(new_val, existing_keys)
                    if err:
                        QMessageBox.warning(self, "Invalid Method Name", err)
                        edt.hide()
                        lbl.show()
                        return
                    self.rename_method(name, new_val)

                m_title.clicked.connect(_start_m_rename)
                m_title.double_clicked.connect(_start_m_rename)
                m_edit.returnPressed.connect(_commit_m_rename)
                m_edit.escape_pressed.connect(_cancel_m_rename)
                m_edit.editingFinished.connect(_commit_m_rename)

                title_row.addStretch()

                btn_remove = _make_subtle_tool_button("trash", f"Remove method '{m_name}' from this experiment", size=24, icon_size=14)
                btn_remove.clicked.connect(lambda _, mn=m_name: self.remove_method(mn))
                title_row.addWidget(btn_remove)
                sc_layout.addRow(title_row)

                # --- AGENT SECTION ---
                agent_raw = m_spec.get("agent")
                if isinstance(agent_raw, dict):
                    agent_val = (agent_raw.get("algorithm") or agent_raw.get("name") or next((k for k in agent_raw if k != "name"), ""))
                else:
                    agent_val = str(agent_raw or "")
                paradigm_has_agents = bool(tree and paradigm in tree.paradigms and tree.agents_for(paradigm))
                txt_agent = None
                if not paradigm_has_agents and not agent_val:
                    pass
                elif tree and tree.paradigms.get(paradigm):
                    permitted = [a.name for a in tree.agents_for(paradigm)]
                    txt_agent = ComboBox()
                    txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_agent.setPlaceholderText("Select algorithm...")
                    txt_agent.addItems(permitted)
                    if agent_val:
                        if agent_val not in permitted:
                            txt_agent.insertItem(0, agent_val)
                            is_missing = not (
                                tree and (
                                    agent_val in tree.agents
                                    or any(a.name == agent_val or a.algorithm == agent_val for a in tree.agents.values())
                                    or (tree.config_root / "agent" / f"{agent_val}.yaml").is_file()
                                )
                            )
                            if is_missing:
                                txt_agent.setToolTip(f"Missing component: Algorithm '{agent_val}' is not installed in in/config/agent/. Install it via Component Hub.")
                            else:
                                txt_agent.setToolTip(
                                    f"'{agent_val}' is not permitted by {paradigm} "
                                    f"(allowed: {', '.join(permitted) or 'none declared'})"
                                )
                        else:
                            txt_agent.setToolTip(f"Agents permitted by {paradigm}")
                        txt_agent.setCurrentText(agent_val)
                    else:
                        txt_agent.setCurrentIndex(-1)
                        txt_agent.setToolTip(f"Select an agent permitted by {paradigm}")

                    def _on_agent_changed(v, mn=m_name, cur_ag=agent_val):
                        if self._block_updates or v == cur_ag:
                            return
                        methods = self.raw_data.setdefault("methods", {})
                        ms = methods.setdefault(mn, {})
                        if v:
                            ms["agent"] = v
                        else:
                            ms.pop("agent", None)
                        self._mark_dirty()
                        self._render_boxes()

                    txt_agent.currentTextChanged.connect(_on_agent_changed)
                else:
                    txt_agent = QLineEdit(agent_val)
                    txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_agent.setPlaceholderText("Algorithm...")
                    txt_agent.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "agent", v)
                    )

                agent_keys_present = set()
                if txt_agent is not None:
                    agent_rows = []
                    btn_add_ap = None
                    if agent_val:
                        cand_agent = get_agent_candidate_hyperparameters(agent_val, tree, set(m_spec.keys()))
                        btn_add_ap = _make_param_adder_button(
                            f"Add {agent_val or 'agent'} parameter",
                            cand_agent,
                            lambda p, mn=m_name: self._add_method_param(mn, p),
                            parent_widget=sub_card
                        )

                        agent_defaults = dict(tree.agent_hyperparameters(agent_val)) if (tree and agent_val) else {}
                        agent_param_keys = set(agent_defaults.keys()) - set(TRAINING_HYPERPARAMETERS) - {"algorithm", "name", "type"}

                        # 1. Overridden keys in m_spec
                        _m_raw = m_spec.get("model", "")
                        _m_name = next(iter(_m_raw)) if isinstance(_m_raw, dict) and _m_raw else str(_m_raw or "")
                        known_model_slots = set(tree.model_slots(_m_name)) if (tree and _m_name in tree.models) else set()
                        for ak, av in list(m_spec.items()):
                            if ak in ("agent", "model", "base") or ak in known_model_slots:
                                continue
                            if ak in agent_param_keys or (agent_val and ak.startswith(f"{agent_val}_")):
                                agent_keys_present.add(ak)
                                row_ap = QHBoxLayout()
                                if isinstance(av, bool):
                                    cb_ap = ToggleSlider(checked=av)
                                    cb_ap.toggled.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                    row_ap.addWidget(cb_ap)
                                elif isinstance(av, (int, float)):
                                    spin_ap = number_field(av)
                                    spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                    spin_ap.valueChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                    row_ap.addWidget(spin_ap)
                                else:
                                    line_ap = QLineEdit(str(av if av is not None else ""))
                                    line_ap.setMaximumWidth(TEXT_FIELD_WIDTH)
                                    line_ap.textChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                    row_ap.addWidget(line_ap)
                                btn_del_ap = _make_subtle_tool_button("close", f"Remove '{ak}'", size=20, icon_size=12)
                                btn_del_ap.clicked.connect(lambda _, mn=m_name, pk=ak: self._remove_method_param(mn, pk))
                                row_ap.addWidget(btn_del_ap)
                                agent_rows.append((str(ak), row_ap))

                        # 2. Defaults if show_ag_defs enabled
                        if show_ag_defs:
                            for ak in sorted(agent_param_keys):
                                if ak not in agent_keys_present:
                                    def_val = agent_defaults.get(ak)
                                    if def_val is None:
                                        continue
                                    row_ap = QHBoxLayout()
                                    if isinstance(def_val, bool):
                                        cb_ap = ToggleSlider(checked=def_val)
                                        cb_ap.toggled.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                        row_ap.addWidget(cb_ap)
                                    elif isinstance(def_val, (int, float)):
                                        spin_ap = number_field(def_val)
                                        spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                        spin_ap.valueChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                        row_ap.addWidget(spin_ap)
                                    else:
                                        line_ap = QLineEdit(str(def_val if def_val is not None else ""))
                                        line_ap.setMaximumWidth(TEXT_FIELD_WIDTH)
                                        line_ap.textChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                        row_ap.addWidget(line_ap)
                                    lbl_def = QLabel("(default)")
                                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                                    row_ap.addWidget(lbl_def)
                                    agent_rows.append((str(ak), row_ap))

                    row_ag = QHBoxLayout()
                    row_ag.addWidget(txt_agent)
                    if btn_add_ap:
                        row_ag.addWidget(btn_add_ap)
                    row_ag.addStretch()
                    sc_layout.addRow("Algorithm", row_ag)

                    if agent_rows:
                        agent_block = IndentedHierarchyBlock(level=1)
                        for r_lbl, r_box in agent_rows:
                            r_box.addStretch()
                            agent_block.addRow(r_lbl, r_box)
                        sc_layout.addRow(agent_block)

                # --- MODEL SECTION ---
                model_raw = m_spec.get("model", "")
                if isinstance(model_raw, dict):
                    model_name = next(iter(model_raw))
                    model_inner = model_raw.get(model_name) if isinstance(model_raw.get(model_name), dict) else {}
                else:
                    model_name = str(model_raw or "")
                    model_inner = {}

                if tree and tree.models:
                    txt_model = ComboBox()
                    txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_model.setPlaceholderText("Select model...")
                    known = sorted(tree.models)
                    txt_model.addItems(known)
                    if model_name:
                        if model_name not in known:
                            txt_model.insertItem(0, model_name)
                            mo_stem = model_name.split("/")[-1]
                            is_missing = not (
                                tree and (
                                    model_name in tree.models
                                    or mo_stem in tree.models
                                    or (tree.config_root / "model" / f"{model_name}.yaml").is_file()
                                    or (tree.config_root / "model" / "basics" / f"{mo_stem}.yaml").is_file()
                                )
                            )
                            if is_missing:
                                txt_model.setToolTip(f"Missing component: Model '{model_name}' is not installed in in/config/model/. Install it via Component Hub.")
                            else:
                                txt_model.setToolTip("Architectures defined in in/config/model/")
                        else:
                            txt_model.setToolTip("Architectures defined in in/config/model/")
                        txt_model.setCurrentText(model_name)

                    def _on_model_changed(v, mn=m_name, cur_mn=model_name):
                        if self._block_updates or v == cur_mn:
                            return
                        methods = self.raw_data.setdefault("methods", {})
                        ms = methods.setdefault(mn, {})
                        if not v:
                            ms.pop("model", None)
                        elif tree and v in tree.models and tree.models[v].is_composite:
                            if not isinstance(ms.get("model"), dict):
                                ms["model"] = v
                        else:
                            ms["model"] = v
                            prev_slots = tree.model_slots(cur_mn) if (tree and cur_mn in tree.models) else []
                            for sk in prev_slots:
                                ms.pop(sk, None)
                        self._mark_dirty()
                        self._render_boxes()

                    txt_model.currentTextChanged.connect(_on_model_changed)
                else:
                    txt_model = QLineEdit(model_name)
                    txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_model.setPlaceholderText("Model...")
                    txt_model.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "model", v)
                    )

                model_keys_present = set()
                model_rows = []
                btn_add_mp = None
                known_slots = []

                if model_name:
                    is_composite_model = bool(
                        tree and model_name in tree.models and tree.models[model_name].is_composite
                    )
                    composite_param_keys = []
                    if tree and model_name in tree.models:
                        composite_param_keys = [
                            k for k, v in tree.model_parameters(model_name).items()
                            if not isinstance(v, dict) and k not in ("architecture", "type", "lightning_module", "name", "slots")
                        ]

                    if is_composite_model:
                        for bk in composite_param_keys:
                            val = None
                            if isinstance(model_inner, dict) and bk in model_inner:
                                val = model_inner[bk]
                            elif bk in m_spec:
                                val = m_spec[bk]
                            if val is not None:
                                model_keys_present.add(bk)
                                row_bp = QHBoxLayout()
                                if isinstance(val, bool):
                                    cb_bp = ToggleSlider(checked=val)
                                    cb_bp.toggled.connect(lambda v, mn=m_name, pk=bk: self._on_method_param_edited(mn, pk, v))
                                    row_bp.addWidget(cb_bp)
                                elif isinstance(val, (int, float)):
                                    spin_bp = number_field(val)
                                    spin_bp.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                    spin_bp.valueChanged.connect(lambda v, mn=m_name, pk=bk: self._on_method_param_edited(mn, pk, v))
                                    row_bp.addWidget(spin_bp)
                                else:
                                    line_bp = QLineEdit(str(val))
                                    line_bp.setMaximumWidth(TEXT_FIELD_WIDTH)
                                    line_bp.textChanged.connect(lambda v, mn=m_name, pk=bk: self._on_method_param_edited(mn, pk, v))
                                    row_bp.addWidget(line_bp)
                                btn_del_bp = _make_subtle_tool_button("close", f"Remove '{bk}'", size=20, icon_size=12)
                                btn_del_bp.clicked.connect(lambda _, mn=m_name, pk=bk: self._remove_method_param(mn, pk))
                                row_bp.addWidget(btn_del_bp)
                                model_rows.append((str(bk), row_bp))

                        # Determine known subcomponent slot names for this composite model
                        known_slots = []
                        if tree and model_name in tree.models:
                            known_slots = tree.model_slots(model_name)

                        subcomps_present = {}
                        if isinstance(model_inner, dict):
                            for sk, sv in model_inner.items():
                                if isinstance(sv, dict) or sk in known_slots:
                                    subcomps_present[sk] = sv
                        for sk in known_slots:
                            if sk in m_spec and sk not in subcomps_present:
                                subcomps_present[sk] = m_spec[sk]

                        for s_name, s_val in list(subcomps_present.items()):
                            sub_params = {}
                            cur_type = ""
                            if isinstance(s_val, str):
                                cur_type = s_val
                            elif isinstance(s_val, dict):
                                # Detect if submodel is nested under a model name key (e.g. {nsfr: {rules: default}})
                                nested_model_key = next((k for k in s_val if (tree and k in tree.models)), None)
                                if nested_model_key:
                                    cur_type = nested_model_key
                                    sub_params = s_val[nested_model_key] if isinstance(s_val[nested_model_key], dict) else {}
                                else:
                                    cur_type = s_val.get("type", "")
                                    sub_params = {k: v for k, v in s_val.items() if k not in ("type", "architecture", "reasoner")}

                            choices = [
                                m for m in (sorted(tree.models) if tree else [])
                                if (not tree or not tree.models[m].is_composite)
                            ]

                            sub_candidates = []
                            if tree and cur_type and cur_type in tree.models:
                                sub_candidates = [k for k in tree.model_parameters(cur_type).keys() if k not in ("architecture", "type")]
                            elif tree and model_name in tree.models:
                                # Fall back to candidate keys defined in composite model default slot
                                slot_def = tree.models[model_name].parameters.get(s_name)
                                if isinstance(slot_def, dict):
                                    sub_candidates = list(slot_def.keys())
                            filtered_sub_cand = [c for c in sub_candidates if c not in sub_params]
                            btn_add_sub_p = None
                            if filtered_sub_cand:
                                btn_add_sub_p = _make_param_adder_button(
                                    f"Add {s_name} parameter",
                                    filtered_sub_cand,
                                    lambda p, mn=m_name, sn=s_name: self._add_method_subcomponent_param(mn, sn, p),
                                    parent_widget=sub_card
                                )

                            row_gen = QHBoxLayout()
                            if choices and (cur_type in choices or cur_type or not sub_params):
                                combo_gen = ComboBox()
                                combo_gen.setMaximumWidth(TEXT_FIELD_WIDTH)
                                combo_gen.setToolTip(f"Select {s_name} module architecture")
                                combo_gen.addItems(choices)
                                if cur_type and cur_type not in choices:
                                    combo_gen.insertItem(0, cur_type)
                                if cur_type:
                                    combo_gen.setCurrentText(cur_type)
                                combo_gen.currentTextChanged.connect(
                                    lambda v, mn=m_name, sn=s_name: self._on_method_subcomponent_type_changed(mn, sn, v)
                                )
                                row_gen.addWidget(combo_gen)
                            elif cur_type:
                                line_gen = QLineEdit(str(cur_type))
                                line_gen.setMaximumWidth(TEXT_FIELD_WIDTH)
                                line_gen.textChanged.connect(
                                    lambda v, mn=m_name, sn=s_name: self._on_method_subcomponent_type_changed(mn, sn, v)
                                )
                                row_gen.addWidget(line_gen)

                            btn_del_sc = _make_subtle_tool_button("close", f"Remove '{s_name}' subcomponent", size=20, icon_size=12)
                            btn_del_sc.clicked.connect(lambda _, mn=m_name, sn=s_name: self._remove_method_subcomponent(mn, sn))
                            row_gen.addWidget(btn_del_sc)
                            if btn_add_sub_p:
                                row_gen.addWidget(btn_add_sub_p)

                            slot_title = f"{s_name.replace('_', ' ').title()} Module"
                            model_rows.append((slot_title, row_gen))

                            sub_block = IndentedHierarchyBlock(level=2)
                            sub_rows = []
                            for sp_k, sp_v in list(sub_params.items()):
                                sp_row = QHBoxLayout()
                                sp_line = QLineEdit(str(sp_v if sp_v is not None else ""))
                                sp_line.setMaximumWidth(TEXT_FIELD_WIDTH)
                                sp_line.textChanged.connect(
                                    lambda v, mn=m_name, sn=s_name, pk=sp_k: self._on_method_subcomponent_param_edited(mn, sn, pk, v)
                                )
                                sp_row.addWidget(sp_line)
                                btn_sp_del = _make_subtle_tool_button("close", f"Remove '{sp_k}'", size=20, icon_size=12)
                                btn_sp_del.clicked.connect(
                                    lambda _, mn=m_name, sn=s_name, pk=sp_k: self._remove_method_subcomponent_param(mn, sn, pk)
                                )
                                sp_row.addWidget(btn_sp_del)
                                sub_rows.append((str(sp_k), sp_row))

                            if show_mo_defs and tree:
                                def_sub_dict = {}
                                if cur_type and cur_type in tree.models:
                                    def_sub_dict = dict(tree.model_parameters(cur_type))
                                elif model_name in tree.models:
                                    slot_template = tree.models[model_name].parameters.get(s_name)
                                    if isinstance(slot_template, dict):
                                        def_sub_dict = dict(slot_template)
                                for dp_k, dp_v in def_sub_dict.items():
                                    if dp_k not in sub_params and dp_k not in ("architecture", "type", "reasoner", "mode"):
                                        dp_row = QHBoxLayout()
                                        dp_line = QLineEdit(str(dp_v if dp_v is not None else ""))
                                        dp_line.setMaximumWidth(TEXT_FIELD_WIDTH)
                                        dp_line.textChanged.connect(
                                            lambda v, mn=m_name, sn=s_name, pk=dp_k: self._on_method_subcomponent_param_edited(mn, sn, pk, v)
                                        )
                                        dp_row.addWidget(dp_line)
                                        lbl_def = QLabel("(default)")
                                        lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                                        dp_row.addWidget(lbl_def)
                                        sub_rows.append((str(dp_k), dp_row))

                            if sub_rows:
                                for sr_lbl, sr_box in sub_rows:
                                    sr_box.addStretch()
                                    sub_block.addRow(sr_lbl, sr_box)
                                model_rows.append(("", sub_block))

                        curr_comp_keys = set()
                        if isinstance(model_inner, dict):
                            curr_comp_keys.update(model_inner.keys())
                        curr_comp_keys.update(m_spec.keys())

                        cand_subcomps = get_method_candidate_subcomponents(model_name, tree, set(subcomps_present.keys()))
                        cand_comp = cand_subcomps + [k for k in composite_param_keys if k not in curr_comp_keys]
                        if cand_comp:
                            btn_add_mp = _make_param_adder_button(
                                f"Add {model_name} parameter",
                                cand_comp,
                                lambda p, mn=m_name: self._add_method_param(mn, p),
                                parent_widget=sub_card
                            )

                    else:
                        model_defs = dict(tree.model_parameters(model_name)) if tree else {}
                        model_keys_raw = set(model_defs.keys())
                        model_keys = model_keys_raw - {"architecture", "type", "lightning_module"}
                        if model_name == "dnn":
                            model_keys.update(["hidden_sizes", "activation"])

                        for mk in list(model_keys):
                            val = None
                            if mk in m_spec:
                                val = m_spec[mk]
                            elif isinstance(model_inner, dict) and mk in model_inner:
                                val = model_inner[mk]
                            if val is not None:
                                model_keys_present.add(mk)
                                if isinstance(val, dict):
                                    sub_block = IndentedHierarchyBlock(level=2)
                                    hdr_grp = QHBoxLayout()
                                    hdr_grp.addWidget(label(f"[{mk}]", "muted"))
                                    hdr_grp.addStretch()
                                    btn_del_grp = _make_subtle_tool_button("close", f"Remove parameter group '{mk}'", size=20, icon_size=12)
                                    btn_del_grp.clicked.connect(lambda _, mn=m_name, pk=mk: self._remove_method_param(mn, pk))
                                    hdr_grp.addWidget(btn_del_grp)
                                    model_rows.append(("", hdr_grp))
                                    for nk, nv in val.items():
                                        row_n = QHBoxLayout()
                                        if isinstance(nv, bool):
                                            cb_n = ToggleSlider(checked=nv)
                                            cb_n.toggled.connect(lambda v, mn=m_name, pk=mk, ck=nk: self._on_method_nested_param_edited(mn, pk, ck, v))
                                            row_n.addWidget(cb_n)
                                        elif isinstance(nv, (int, float)):
                                            spin_n = number_field(nv)
                                            spin_n.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                            spin_n.valueChanged.connect(lambda v, mn=m_name, pk=mk, ck=nk: self._on_method_nested_param_edited(mn, pk, ck, v))
                                            row_n.addWidget(spin_n)
                                        elif isinstance(nv, list):
                                            line_n = QLineEdit(str(list(nv)))
                                            line_n.setMaximumWidth(TEXT_FIELD_WIDTH)
                                            line_n.textChanged.connect(lambda v, mn=m_name, pk=mk, ck=nk: self._on_method_nested_param_edited(mn, pk, ck, v))
                                            row_n.addWidget(line_n)
                                        else:
                                            line_n = QLineEdit(str(nv if nv is not None else ""))
                                            line_n.setMaximumWidth(TEXT_FIELD_WIDTH)
                                            line_n.textChanged.connect(lambda v, mn=m_name, pk=mk, ck=nk: self._on_method_nested_param_edited(mn, pk, ck, v))
                                            row_n.addWidget(line_n)
                                        btn_del_nk = _make_subtle_tool_button("close", f"Remove '{nk}'", size=20, icon_size=12)
                                        btn_del_nk.clicked.connect(lambda _, mn=m_name, pk=mk, ck=nk: self._remove_method_nested_param(mn, pk, ck))
                                        row_n.addWidget(btn_del_nk)
                                        row_n.addStretch()
                                        sub_block.addRow(str(nk), row_n)
                                    model_rows.append(("", sub_block))
                                else:
                                    row_mp = QHBoxLayout()
                                    if isinstance(val, bool):
                                        cb_mp = ToggleSlider(checked=val)
                                        cb_mp.toggled.connect(lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v))
                                        row_mp.addWidget(cb_mp)
                                    elif isinstance(val, (int, float)):
                                        spin_mp = number_field(val)
                                        spin_mp.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                        spin_mp.valueChanged.connect(lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v))
                                        row_mp.addWidget(spin_mp)
                                    elif isinstance(val, list):
                                        line_mp = QLineEdit(str(list(val)))
                                        line_mp.setMaximumWidth(TEXT_FIELD_WIDTH)
                                        line_mp.textChanged.connect(
                                            lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v)
                                        )
                                        row_mp.addWidget(line_mp)
                                    else:
                                        line_mp = QLineEdit(str(val))
                                        line_mp.setMaximumWidth(TEXT_FIELD_WIDTH)
                                        line_mp.textChanged.connect(
                                            lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v)
                                        )
                                        row_mp.addWidget(line_mp)
                                    btn_del_mp = _make_subtle_tool_button("close", f"Remove '{mk}'", size=20, icon_size=12)
                                    btn_del_mp.clicked.connect(lambda _, mn=m_name, pk=mk: self._remove_method_param(mn, pk))
                                    row_mp.addWidget(btn_del_mp)
                                    model_rows.append((str(mk), row_mp))

                        if show_mo_defs:
                            for mk in sorted(model_keys):
                                if mk not in model_keys_present:
                                    def_val = model_defs.get(mk)
                                    if def_val is None:
                                        if mk == "hidden_sizes" and model_name == "dnn":
                                            def_val = "[64, 64]"
                                        elif mk == "activation" and model_name == "dnn":
                                            def_val = "relu"
                                        else:
                                            continue
                                    row_mp = QHBoxLayout()
                                    if isinstance(def_val, bool):
                                        cb_mp = ToggleSlider(checked=def_val)
                                        cb_mp.toggled.connect(lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v))
                                        row_mp.addWidget(cb_mp)
                                    elif isinstance(def_val, (int, float)):
                                        spin_mp = number_field(def_val)
                                        spin_mp.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                        spin_mp.valueChanged.connect(lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v))
                                        row_mp.addWidget(spin_mp)
                                    else:
                                        line_mp = QLineEdit(str(def_val if def_val is not None else ""))
                                        line_mp.setMaximumWidth(TEXT_FIELD_WIDTH)
                                        line_mp.textChanged.connect(
                                            lambda v, mn=m_name, pk=mk: self._on_method_param_edited(mn, pk, v)
                                        )
                                        row_mp.addWidget(line_mp)
                                    lbl_def = QLabel("(default)")
                                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                                    row_mp.addWidget(lbl_def)
                                    model_rows.append((str(mk), row_mp))

                        curr_model_keys = set(m_spec.keys())
                        if isinstance(model_inner, dict):
                            curr_model_keys.update(model_inner.keys())
                        cand_model = get_model_candidate_hyperparameters(model_name, tree, curr_model_keys)
                        if cand_model:
                            btn_add_mp = _make_param_adder_button(
                                f"Add {model_name} parameter",
                                cand_model,
                                lambda p, mn=m_name: self._add_method_param(mn, p),
                                parent_widget=sub_card
                            )

                    row_mo = QHBoxLayout()
                    row_mo.addWidget(txt_model)
                    if btn_add_mp:
                        row_mo.addWidget(btn_add_mp)
                    row_mo.addStretch()
                    sc_layout.addRow("Model", row_mo)

                    if model_rows:
                        model_block = IndentedHierarchyBlock(level=1)
                        for r_lbl, r_box in model_rows:
                            if isinstance(r_box, QWidget):
                                model_block.addRow(r_box)
                            else:
                                r_box.addStretch()
                                model_block.addRow(r_lbl, r_box)
                        sc_layout.addRow(model_block)
                else:
                    sc_layout.addRow("Model", txt_model)

                # --- TRAINING PARAMETERS SECTION (UNINDENTED, NO HEADER) ---
                train_rows = []
                u_params = self.raw_data.get("methods", {}).get("params", {})
                base_def_name, base_def_spec = self._get_group_default_method()
                base_def_spec = base_def_spec or {}

                if "lr" in m_spec:
                    row_lr = QHBoxLayout()
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(float(m_spec["lr"]))
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    row_lr.addWidget(spin_lr)
                    btn_del = _make_subtle_tool_button("close", "Remove lr parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "lr"))
                    row_lr.addWidget(btn_del)
                    train_rows.append(("Learning Rate (lr)", row_lr))
                elif show_all_defs:
                    def_lr = float(u_params.get("lr") or base_def_spec.get("lr") or 0.0003)
                    row_lr = QHBoxLayout()
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(def_lr)
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    row_lr.addWidget(spin_lr)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_lr.addWidget(lbl_def)
                    train_rows.append(("Learning Rate (lr)", row_lr))

                if "batch_size" in m_spec:
                    row_bs = QHBoxLayout()
                    spin_bs = SpinBox()
                    spin_bs.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(int(m_spec["batch_size"]))
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    row_bs.addWidget(spin_bs)
                    btn_del = _make_subtle_tool_button("close", "Remove batch_size parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "batch_size"))
                    row_bs.addWidget(btn_del)
                    train_rows.append(("Batch Size", row_bs))
                elif show_all_defs:
                    def_bs = int(u_params.get("batch_size") or base_def_spec.get("batch_size") or 64)
                    row_bs = QHBoxLayout()
                    spin_bs = SpinBox()
                    spin_bs.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(def_bs)
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    row_bs.addWidget(spin_bs)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_bs.addWidget(lbl_def)
                    train_rows.append(("Batch Size", row_bs))

                if "gamma" in m_spec:
                    row_g = QHBoxLayout()
                    spin_g = DoubleSpinBox()
                    spin_g.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(float(m_spec["gamma"]))
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    row_g.addWidget(spin_g)
                    btn_del = _make_subtle_tool_button("close", "Remove gamma parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "gamma"))
                    row_g.addWidget(btn_del)
                    train_rows.append(("Discount Factor (γ)", row_g))
                elif show_all_defs:
                    def_gamma = float(u_params.get("gamma") or base_def_spec.get("gamma") or 0.99)
                    row_g = QHBoxLayout()
                    spin_g = DoubleSpinBox()
                    spin_g.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(def_gamma)
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    row_g.addWidget(spin_g)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_g.addWidget(lbl_def)
                    train_rows.append(("Discount Factor (γ)", row_g))

                consumed_keys = {"agent", "model", "base", "lr", "batch_size", "gamma"}
                if known_slots:
                    consumed_keys.update(known_slots)
                consumed_keys.update(agent_keys_present)
                consumed_keys.update(model_keys_present)

                for extra_k, extra_v in list(m_spec.items()):
                    if extra_k in consumed_keys:
                        continue
                    if isinstance(extra_v, (int, float, str, bool)):
                        row_extra = QHBoxLayout()
                        if isinstance(extra_v, bool):
                            slider_extra = ToggleSlider(checked=extra_v)
                            slider_extra.toggled.connect(
                                lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                            )
                            row_extra.addWidget(slider_extra)
                        elif isinstance(extra_v, (int, float)):
                            spin_extra = number_field(extra_v)
                            spin_extra.setMaximumWidth(NUMBER_FIELD_WIDTH)
                            spin_extra.valueChanged.connect(
                                lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                            )
                            row_extra.addWidget(spin_extra)
                        else:
                            line_extra = QLineEdit(str(extra_v if extra_v is not None else ""))
                            line_extra.setMaximumWidth(TEXT_FIELD_WIDTH)
                            line_extra.textChanged.connect(
                                lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                            )
                            row_extra.addWidget(line_extra)
                        btn_del = _make_subtle_tool_button("close", f"Remove '{extra_k}'", size=20, icon_size=12)
                        btn_del.clicked.connect(lambda _, mn=m_name, ek=extra_k: self._remove_method_param(mn, ek))
                        row_extra.addWidget(btn_del)
                        train_rows.append((str(extra_k), row_extra))

                cand_train = get_training_candidate_hyperparameters(set(m_spec.keys()))
                btn_add_tp = None
                if cand_train:
                    btn_add_tp = _make_param_adder_button(
                        "Add training parameter",
                        cand_train,
                        lambda p, mn=m_name: self._add_method_param(mn, p),
                        parent_widget=sub_card
                    )

                if train_rows:
                    if btn_add_tp:
                        train_rows[-1][1].addWidget(btn_add_tp)
                    for t_lbl, t_row in train_rows:
                        t_row.addStretch()
                        sc_layout.addRow(t_lbl, t_row)
                elif btn_add_tp:
                    row_tr = QHBoxLayout()
                    row_tr.addWidget(btn_add_tp)
                    row_tr.addStretch()
                    sc_layout.addRow("Training", row_tr)

                # Bottom actions row: duplicate and plus icons in the bottom right
                if not is_base:
                    bottom_row = QHBoxLayout()
                    bottom_row.addStretch()
                    btn_dup = _make_subtle_tool_button("copy", f"Duplicate method '{m_name}'", size=24, icon_size=14)
                    btn_dup.clicked.connect(lambda _, mn=m_name: self.duplicate_method(mn))
                    bottom_row.addWidget(btn_dup)

                    btn_plus = _make_subtle_tool_button("plus", "Add new method", size=24, icon_size=14)
                    btn_plus.clicked.connect(lambda _: self._start_add_new_method())
                    bottom_row.addWidget(btn_plus)

                card_vbox.addLayout(sc_layout)
                card_vbox.addStretch()
                if not is_base:
                    card_vbox.addLayout(bottom_row)

                method_container.add_method_card(m_name, sub_card)

            if pending_new:
                pending_card = QFrame()
                pending_card.setObjectName("card")
                pc_vbox = QVBoxLayout(pending_card)
                pc_vbox.setContentsMargins(10, 10, 10, 10)
                pc_vbox.setSpacing(8)

                pc_layout = QFormLayout()
                pc_layout.setContentsMargins(0, 0, 0, 0)
                pc_layout.setVerticalSpacing(8)

                p_title_row = QHBoxLayout()
                p_title_row.setSpacing(6)

                p_edit = InlineTitleEditor()
                p_edit.setObjectName("newMethodEditor")
                def_name, _ = self._get_group_default_method()
                if def_name:
                    p_edit.setPlaceholderText(f"Method name (e.g. {def_name})...")
                else:
                    p_edit.setPlaceholderText("Method name (e.g. ppo_baseline)...")
                p_edit.setStyleSheet("font-size: 16px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
                p_title_row.addWidget(p_edit)
                p_title_row.addStretch()

                btn_cancel_p = _make_subtle_tool_button("close", "Cancel creating method", size=24, icon_size=14)
                btn_cancel_p.clicked.connect(self._cancel_new_method)
                p_title_row.addWidget(btn_cancel_p)
                pc_layout.addRow(p_title_row)

                # Agent Algorithm row
                if tree and tree.paradigms.get(paradigm):
                    p_permitted = [a.name for a in tree.agents_for(paradigm)]
                    p_txt_agent = ComboBox()
                    p_txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    p_txt_agent.setPlaceholderText("Select algorithm...")
                    p_txt_agent.addItems(p_permitted)
                    init_agent = self._pending_new_method_spec.get("agent")
                    if init_agent and init_agent in p_permitted:
                        p_txt_agent.setCurrentText(str(init_agent))
                    else:
                        p_txt_agent.setCurrentIndex(-1)
                    p_txt_agent.currentTextChanged.connect(
                        lambda v: self._pending_new_method_spec.__setitem__("agent", v) if v else self._pending_new_method_spec.pop("agent", None)
                    )
                    pc_layout.addRow("Algorithm", p_txt_agent)

                # Model row
                if tree and tree.models:
                    p_txt_model = ComboBox()
                    p_txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                    p_txt_model.setPlaceholderText("Select model...")
                    p_txt_model.addItems(sorted(tree.models))
                    init_model = self._pending_new_method_spec.get("model")
                    if init_model and init_model in tree.models:
                        p_txt_model.setCurrentText(str(init_model))
                    else:
                        p_txt_model.setCurrentIndex(-1)
                    p_txt_model.currentTextChanged.connect(
                        lambda v: self._pending_new_method_spec.__setitem__("model", v) if v else self._pending_new_method_spec.pop("model", None)
                    )
                    pc_layout.addRow("Model", p_txt_model)


                committing_p = [False]

                def _commit_p():
                    if committing_p[0]:
                        return
                    name = p_edit.text().strip()
                    if not name:
                        committing_p[0] = True
                        self._cancel_new_method()
                        return

                    existing = set(self.raw_data.get("methods", {}).keys()) - {"params"}
                    err = self._validate_method_name(name, existing)
                    if err:
                        QMessageBox.warning(self, "Invalid Method Name", err)
                        p_edit.setFocus()
                        p_edit.selectAll()
                        return

                    committing_p[0] = True
                    methods = self.raw_data.setdefault("methods", {})
                    methods[name] = dict(getattr(self, "_pending_new_method_spec", {}))
                    self._pending_new_method = False
                    self._pending_new_method_spec = {}
                    self._mark_dirty()
                    self._render_boxes()

                def _cancel_p():
                    if committing_p[0]:
                        return
                    committing_p[0] = True
                    self._cancel_new_method()

                p_edit.returnPressed.connect(_commit_p)
                p_edit.escape_pressed.connect(_cancel_p)

                def _on_p_blur():
                    if committing_p[0]:
                        return
                    if not p_edit.text().strip():
                        _cancel_p()

                p_edit.editingFinished.connect(_on_p_blur)

                pc_vbox.addLayout(pc_layout)
                pc_vbox.addStretch()
                method_container.add_method_card("", pending_card)
                QTimer.singleShot(0, p_edit.setFocus)

            m_layout.addWidget(method_container)

        box_methods.add_layout(m_layout)
        self.boxes_layout.addWidget(box_methods)

        # --- MODULAR BOX: ENVIRONMENT OVERRIDES ---
        has_env_dict = isinstance(data.get("env"), dict)
        resolved_env = resolved.get("env")
        show_env = has_env_dict or (self.view_mode == "resolved" and isinstance(resolved_env, dict))

        if show_env:
            box_env = ConfigBox("Environment Settings", "Simulation rollouts and vectorized environment settings")
            box_env_header = QHBoxLayout()
            box_env_header.addStretch()
            if has_env_dict:
                btn_rm_env = _make_subtle_tool_button("trash", "Remove environment overrides", size=24, icon_size=14)
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
            spin_num_envs.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_num_envs.setRange(1, 1024)
            spin_num_envs.setValue(int(env_source.get("num_envs") or 4))
            spin_num_envs.valueChanged.connect(lambda v: self._on_nested_edited("env", "num_envs", v))
            grid_e.addWidget(spin_num_envs, 0, 1)

            grid_e.addWidget(label("Steps per Env (num_steps)"), 0, 2)
            spin_num_steps = SpinBox()
            spin_num_steps.setMaximumWidth(NUMBER_FIELD_WIDTH)
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
            tr_header.addStretch()
            if has_trainer:
                btn_rm_tr = _make_subtle_tool_button("trash", "Remove trainer overrides", size=24, icon_size=14)
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
            combo_accel.setMaximumWidth(NUMBER_FIELD_WIDTH)
            combo_accel.addItems(["cpu", "gpu", "mps", "auto"])
            combo_accel.setCurrentText(str(tr_source.get("accelerator") or "cpu"))
            combo_accel.currentTextChanged.connect(lambda v: self._on_nested_edited("trainer", "accelerator", v))
            grid_t.addWidget(combo_accel, 0, 1)

            grid_t.addWidget(label("Max Epochs"), 0, 2)
            spin_epochs = SpinBox()
            spin_epochs.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_epochs.setRange(1, 1000)
            spin_epochs.setValue(int(tr_source.get("max_epochs") or 1))
            spin_epochs.valueChanged.connect(lambda v: self._on_nested_edited("trainer", "max_epochs", v))
            grid_t.addWidget(spin_epochs, 0, 3)

            box_tr.add_layout(grid_t)
            self.boxes_layout.addWidget(box_tr)

        # --- MODULAR BOX: DATASET SPECIFICATION ---
        has_dataset = "dataset_path" in data
        is_offline_req = paradigm in ("offline_rl", "supervised")
        show_dataset = has_dataset or is_offline_req or (self.view_mode == "resolved" and "dataset_path" in resolved)

        if show_dataset:
            box_ds = ConfigBox("Dataset Specification", "Offline transition datasets and replay buffer paths")
            ds_header = QHBoxLayout()
            ds_header.addStretch()
            if has_dataset and not is_offline_req:
                btn_rm_ds = _make_subtle_tool_button("trash", "Remove dataset configuration", size=24, icon_size=14)
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
            self.combo_dataset_path.setMaximumWidth(320)
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
            ds_path_row.addWidget(self.combo_dataset_path)

            btn_browse_ds = QToolButton()
            btn_browse_ds.setText("Browse…")
            btn_browse_ds.clicked.connect(lambda: self._browse_dataset_path())
            ds_path_row.addWidget(btn_browse_ds)
            ds_path_row.addWidget(lbl_status)
            ds_path_row.addStretch()
            ds_form.addRow("Dataset Path", ds_path_row)

            box_ds.add_layout(ds_form)
            self.boxes_layout.addWidget(box_ds)

        # --- MODULAR BOX: COMPUTE & SLURM RESOURCES ---
        has_res = "resources" in data and isinstance(data["resources"], dict)
        show_res = has_res or (self.view_mode == "resolved" and "resources" in resolved)

        if show_res:
            box_res = ConfigBox("Compute & Slurm Resources", "Resource allocation for cluster runs (time, gpus, cores, memory)")
            res_header = QHBoxLayout()
            res_header.addStretch()
            if has_res:
                btn_rm_res = _make_subtle_tool_button("trash", "Remove compute resources overrides", size=24, icon_size=14)
                btn_rm_res.clicked.connect(lambda: self.remove_block("resources"))
                res_header.addWidget(btn_rm_res)
            else:
                res_header.addWidget(label("(inherited defaults)", "muted"))
            box_res.add_layout(res_header)

            grid_res = QGridLayout()
            grid_res.setHorizontalSpacing(16)
            grid_res.setVerticalSpacing(8)
            res_dict = data.get("resources", {}) if has_res else resolved.get("resources", {})

            # Site Profile Dropdown
            grid_res.addWidget(label("Site Profile"), 0, 0)
            combo_site = ComboBox()
            combo_site.setMaximumWidth(NUMBER_FIELD_WIDTH)
            sites = list(tree.sites) if tree and hasattr(tree, "sites") else ["local", "ncshare", "arc"]
            if "local" not in sites:
                sites.insert(0, "local")
            current_site = str(data.get("site") or resolved.get("site") or "local")
            for s in sites:
                combo_site.addItem(s)
            s_idx = combo_site.findText(current_site)
            if s_idx != -1:
                combo_site.setCurrentIndex(s_idx)
            combo_site.currentTextChanged.connect(
                lambda v: self._on_field_edited("site", v)
            )
            grid_res.addWidget(combo_site, 0, 1)

            grid_res.addWidget(label("Time Limit"), 0, 2)
            txt_time = QLineEdit(str(res_dict.get("time") or "02:00:00"))
            txt_time.setMaximumWidth(NUMBER_FIELD_WIDTH)
            txt_time.textChanged.connect(lambda v: self._on_nested_edited("resources", "time", v))
            grid_res.addWidget(txt_time, 0, 3)

            grid_res.addWidget(label("GPUs"), 1, 0)
            txt_gpus = QLineEdit(str(res_dict.get("gpus") if res_dict.get("gpus") is not None else 0))
            txt_gpus.setMaximumWidth(NUMBER_FIELD_WIDTH)
            txt_gpus.textChanged.connect(lambda v: self._on_nested_edited("resources", "gpus", v))
            grid_res.addWidget(txt_gpus, 1, 1)

            grid_res.addWidget(label("CPU Cores"), 1, 2)
            txt_cores = QLineEdit(str(res_dict.get("cores") if res_dict.get("cores") is not None else 4))
            txt_cores.setMaximumWidth(NUMBER_FIELD_WIDTH)
            txt_cores.textChanged.connect(lambda v: self._on_nested_edited("resources", "cores", v))
            grid_res.addWidget(txt_cores, 1, 3)

            grid_res.addWidget(label("Memory"), 2, 0)
            txt_mem = QLineEdit(str(res_dict.get("memory") or "16G"))
            txt_mem.setMaximumWidth(NUMBER_FIELD_WIDTH)
            txt_mem.textChanged.connect(lambda v: self._on_nested_edited("resources", "memory", v))
            grid_res.addWidget(txt_mem, 2, 1)

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
            sw_header.addStretch()
            if has_sweeper:
                btn_rm_sw = _make_subtle_tool_button("trash", "Remove hyperparameter sweeper overrides", size=24, icon_size=14)
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
            txt_sw_name.setMaximumWidth(200)
            txt_sw_name.textChanged.connect(lambda v: self._on_sweeper_edited("study_name", v))
            sw_form.addWidget(txt_sw_name, 0, 1)

            sw_form.addWidget(label("Trials"), 0, 2)
            spin_trials = SpinBox()
            spin_trials.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_trials.setRange(1, 10000)
            spin_trials.setValue(int(sw_dict.get("n_trials") or 50))
            spin_trials.valueChanged.connect(lambda v: self._on_sweeper_edited("n_trials", v))
            sw_form.addWidget(spin_trials, 0, 3)

            sw_form.addWidget(label("Direction"), 1, 0)
            combo_dir = ComboBox()
            combo_dir.setMaximumWidth(NUMBER_FIELD_WIDTH)
            combo_dir.addItems(["maximize", "minimize"])
            combo_dir.setCurrentText(str(sw_dict.get("direction") or "maximize"))
            combo_dir.currentTextChanged.connect(lambda v: self._on_sweeper_edited("direction", v))
            sw_form.addWidget(combo_dir, 1, 1)

            sw_form.addWidget(label("Parallel Jobs"), 1, 2)
            spin_jobs = SpinBox()
            spin_jobs.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_jobs.setRange(1, 64)
            spin_jobs.setValue(int(sw_dict.get("n_jobs") or 1))
            spin_jobs.valueChanged.connect(lambda v: self._on_sweeper_edited("n_jobs", v))
            sw_form.addWidget(spin_jobs, 1, 3)

            box_sw.add_layout(sw_form)
            self.boxes_layout.addWidget(box_sw)

        # --- ADD CONFIGURATION BLOCK DROPDOWN BUTTON ---
        btn_add_block_row = QHBoxLayout()
        self.btn_add_block = _make_subtle_tool_button("plus", "Add override", size=26, icon_size=14)
        self.btn_add_block.setObjectName("addBlockButton")
        self.btn_add_block.setStyleSheet("""
            QToolButton#addBlockButton {
                background: transparent;
                border: 1px dashed #504945;
                border-radius: 4px;
                padding: 0px;
            }
            QToolButton#addBlockButton:hover {
                background: #32302f;
                border-color: #a89984;
            }
            QToolButton#addBlockButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """)
        self.btn_add_block.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QMenu(self.btn_add_block)
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

        self._render_remaining_sections()

    def _scalar_widget(self, value, write):
        """A widget matching the YAML type, so an int survives a round trip.

        The bool test comes first: bool is a subclass of int, and a checkbox
        rendered as a spin box would write 0/1 back over true/false.
        """
        if isinstance(value, bool):
            widget = ToggleSlider(checked=value)
            widget.toggled.connect(write)
        elif isinstance(value, (int, float)):
            widget = number_field(value)
            widget.valueChanged.connect(write)
        else:
            widget = QLineEdit("" if value is None else str(value))
            widget.textChanged.connect(write)
        return widget

    def _list_field(self, items, write):
        """Comma-separated editor for a YAML list, item types preserved."""
        field = QLineEdit(", ".join(str(item) for item in items))
        field.setToolTip("Comma-separated list")
        field.textChanged.connect(lambda text: write(parse_list_field(text)))
        return field

    def _render_remaining_sections(self):
        """Surface every top-level key the fixed boxes do not cover.

        Without this an experiment's paradigm-specific configuration is simply
        absent from the panel - 20 settings for supervised, which keeps all of
        it under `early_prediction:` - even though saving preserves it. Values
        nested more than one level stay read-only rather than being flattened
        into something that would not round trip.
        """
        leftovers = {k: v for k, v in self.raw_data.items() if k not in FIXED_BOX_KEYS}
        if not leftovers:
            return

        scalars = {k: v for k, v in leftovers.items() if not isinstance(v, (dict, list))}
        sections = {k: v for k, v in leftovers.items() if isinstance(v, (dict, list))}

        if scalars:
            box = ConfigBox()
            # ConfigBox drops its title argument since the param box headers were
            # removed, and an unlabelled box of tau_min/tau_max tells you nothing,
            # so these sections carry their own heading the way method cards do.
            box.add_widget(label("Additional Settings", "cardTitle"))
            form = QFormLayout()
            form.setVerticalSpacing(8)
            for key, value in scalars.items():
                form.addRow(key, self._scalar_widget(value, lambda v, k=key: self._on_field_edited(k, v)))
            box.add_layout(form)
            self.boxes_layout.addWidget(box)

        for name, value in sections.items():
            box = ConfigBox()
            box.add_widget(label(name, "cardTitle"))
            form = QFormLayout()
            form.setVerticalSpacing(8)
            if isinstance(value, list):
                form.addRow("items", self._list_field(value, lambda v, k=name: self._on_field_edited(k, v)))
            else:
                for key, sub in value.items():
                    if isinstance(sub, list):
                        form.addRow(
                            key,
                            self._list_field(sub, lambda v, s=name, k=key: self._on_nested_edited(s, k, v)),
                        )
                    elif isinstance(sub, dict):
                        nested = label(yaml.safe_dump(sub, default_flow_style=True).strip(), "muted")
                        nested.setWordWrap(True)
                        nested.setToolTip(f"Nested under {name}.{key}; edit this one in the YAML tab")
                        form.addRow(key, nested)
                    else:
                        form.addRow(
                            key,
                            self._scalar_widget(sub, lambda v, s=name, k=key: self._on_nested_edited(s, k, v)),
                        )
            box.add_layout(form)
            self.boxes_layout.addWidget(box)

    def _render_generic_boxes(self):
        """Render clean boxes for modular YAML configs (agent/*.yaml, env/*.yaml, model/*.yaml, etc.)."""
        self.paradigm_badge.hide()
        data = self.raw_data

        stem = self.current_path.stem if self.current_path else ""
        leaf_params = {}
        nested_sections = {}

        for k, v in data.items():
            if k in ("name", "algorithm", "architecture", "reasoner", "experiment_id", "id"):
                continue
            if k in ("type", "env_id") and str(v).lower() == stem.lower():
                continue
            if isinstance(v, str) and v.lower() == stem.lower() and k in ("model", "agent"):
                continue
            if isinstance(v, dict):
                nested_sections[k] = v
            else:
                leaf_params[k] = v

        # --- Parameters Card (Scalars and Lists) ---
        box_props = ConfigBox()
        hdr_layout = QHBoxLayout()
        hdr_layout.setSpacing(8)
        lbl_card_title = label("Parameters", "cardTitle")
        hdr_layout.addWidget(lbl_card_title)
        hdr_layout.addWidget(label(f"Settings in {stem}", "muted"))
        hdr_layout.addStretch()

        btn_add_p = _make_subtle_tool_button("plus", "Add parameter", size=22, icon_size=12)
        btn_add_p.clicked.connect(self._prompt_add_generic_leaf_param)
        hdr_layout.addWidget(btn_add_p)
        box_props.add_layout(hdr_layout)

        if leaf_params:
            form = QFormLayout()
            form.setVerticalSpacing(8)
            for k, v in leaf_params.items():
                row = QHBoxLayout()
                row.setSpacing(8)
                if isinstance(v, bool):
                    chk = ToggleSlider(checked=v)
                    chk.toggled.connect(lambda val, key=k: self._on_field_edited(key, val))
                    row.addWidget(chk)
                elif isinstance(v, (int, float)):
                    spin = number_field(v)
                    spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin.valueChanged.connect(lambda val, key=k: self._on_field_edited(key, val))
                    row.addWidget(spin)
                elif isinstance(v, list):
                    txt = QLineEdit(str(list(v)))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH + 80)
                    txt.setToolTip("YAML list, e.g. [64, 64]")
                    txt.textChanged.connect(lambda val, key=k: self._on_field_edited(key, _parse_yaml_value(val)))
                    row.addWidget(txt)
                else:
                    txt = QLineEdit(str(v if v is not None else ""))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt.textChanged.connect(lambda val, key=k: self._on_field_edited(key, _parse_yaml_value(val)))
                    row.addWidget(txt)
                btn_del = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del.clicked.connect(lambda _, key=k: self._remove_generic_field(key))
                row.addWidget(btn_del)
                row.addStretch()
                form.addRow(k, row)
            box_props.add_layout(form)
        else:
            empty_row = QHBoxLayout()
            empty_row.addWidget(label("No parameters configured. Click '+' to add one.", "muted"))
            empty_row.addStretch()
            box_props.add_layout(empty_row)

        self.boxes_layout.addWidget(box_props)

        # --- Nested Sections (Hierarchical Parameters) ---
        for section_name, section_val in nested_sections.items():
            box_sec = ConfigBox()
            sec_hdr = QHBoxLayout()
            sec_hdr.setSpacing(8)
            sec_hdr.addWidget(label(section_name, "cardTitle"))
            sec_hdr.addStretch()

            btn_add_sec_p = _make_subtle_tool_button("plus", f"Add parameter to '{section_name}'", size=22, icon_size=12)
            btn_add_sec_p.clicked.connect(lambda _, sn=section_name: self._prompt_add_nested_generic_param((sn,)))
            sec_hdr.addWidget(btn_add_sec_p)

            btn_del_sec = _make_subtle_tool_button("close", f"Remove section '{section_name}'", size=22, icon_size=12)
            btn_del_sec.clicked.connect(lambda _, sn=section_name: self._remove_generic_field(sn))
            sec_hdr.addWidget(btn_del_sec)
            box_sec.add_layout(sec_hdr)

            if isinstance(section_val, dict):
                form_sec = QFormLayout()
                form_sec.setVerticalSpacing(8)
                self._render_generic_dict_contents(form_sec, section_val, (section_name,), level=1)
                box_sec.add_layout(form_sec)
            self.boxes_layout.addWidget(box_sec)

    def _render_generic_dict_contents(self, parent_layout, d: dict, path: tuple[str, ...], level: int = 1):
        """Recursively render hierarchical nested dictionary contents in the generic component viewer."""
        for k, v in d.items():
            if isinstance(v, dict):
                sub_block = IndentedHierarchyBlock(level=level)
                sub_hdr = QHBoxLayout()
                sub_hdr.addWidget(label(f"[{k}]", "muted"))
                sub_hdr.addStretch()
                btn_add_sub = _make_subtle_tool_button("plus", f"Add parameter to '{k}'", size=20, icon_size=12)
                btn_add_sub.clicked.connect(lambda _, p=path + (k,): self._prompt_add_nested_generic_param(p))
                sub_hdr.addWidget(btn_add_sub)
                btn_del_sub = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del_sub.clicked.connect(lambda _, p=path + (k,): self._remove_generic_path(p))
                sub_hdr.addWidget(btn_del_sub)

                sub_block.addRow("", sub_hdr)
                self._render_generic_dict_contents(sub_block, v, path + (k,), level=level + 1)
                parent_layout.addRow("", sub_block)
            else:
                row = QHBoxLayout()
                row.setSpacing(8)
                if isinstance(v, bool):
                    chk = ToggleSlider(checked=v)
                    chk.toggled.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, val))
                    row.addWidget(chk)
                elif isinstance(v, (int, float)):
                    spin = number_field(v)
                    spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin.valueChanged.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, val))
                    row.addWidget(spin)
                elif isinstance(v, list):
                    txt = QLineEdit(str(list(v)))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH + 80)
                    txt.setToolTip("YAML list, e.g. [64, 64]")
                    txt.textChanged.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, _parse_yaml_value(val)))
                    row.addWidget(txt)
                else:
                    txt = QLineEdit(str(v if v is not None else ""))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt.textChanged.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, _parse_yaml_value(val)))
                    row.addWidget(txt)
                btn_del = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del.clicked.connect(lambda _, p=path + (k,): self._remove_generic_path(p))
                row.addWidget(btn_del)
                row.addStretch()
                parent_layout.addRow(k, row)

    def _prompt_add_generic_leaf_param(self):
        """Prompt user to add a new parameter to the component configuration."""
        key, ok = QInputDialog.getText(self, "Add Parameter", "Parameter name:")
        if not ok or not key.strip():
            return
        key = key.strip()
        val_str, ok2 = QInputDialog.getText(self, "Parameter Value", f"Initial value for '{key}':")
        if not ok2:
            return
        self.raw_data[key] = _parse_yaml_value(val_str)
        self._mark_dirty()
        self._render_boxes()

    def _prompt_add_nested_generic_param(self, path: tuple[str, ...]):
        """Prompt user to add a new parameter inside a nested section."""
        path_str = ".".join(path)
        key, ok = QInputDialog.getText(self, "Add Parameter", f"Parameter name inside '{path_str}':")
        if not ok or not key.strip():
            return
        key = key.strip()
        val_str, ok2 = QInputDialog.getText(self, "Parameter Value", f"Initial value for '{key}':")
        if not ok2:
            return
        self._on_generic_path_edited(path + (key,), _parse_yaml_value(val_str))
        self._render_boxes()

    def _on_generic_path_edited(self, path: tuple[str, ...], value: Any):
        if self._block_updates or not path:
            return
        cur = self.raw_data
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = _parse_yaml_value(value) if isinstance(value, str) else value
        self._mark_dirty()

    def _remove_generic_field(self, key: str):
        self.raw_data.pop(key, None)
        self._mark_dirty()
        self._render_boxes()

    def _remove_generic_path(self, path: tuple[str, ...]):
        if not path:
            return
        cur = self.raw_data
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                return
            cur = cur[p]
        cur.pop(path[-1], None)
        self._mark_dirty()
        self._render_boxes()

    # ── Edit Handlers ──────────────────────────────────────────────────────────

    def _on_field_edited(self, key, value):
        if self._block_updates:
            return
        self.raw_data[key] = _parse_yaml_value(value) if isinstance(value, str) else value
        self._mark_dirty()

    def _validate_method_name(self, name: str, existing_names: set[str]) -> str | None:
        """Return an error message if invalid, or None if valid."""
        name = name.strip()
        if not name:
            return "Method name cannot be empty."
        if name == "params":
            return "The method name 'params' is reserved for universal parameters."
        if name in existing_names:
            return f"A method named '{name}' already exists in this experiment."
        if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
            return "Method name contains invalid characters. Use only letters, digits, hyphens, and underscores."
        return None

    def duplicate_method(self, name: str) -> str | None:
        """Duplicate an existing method card with all its configuration."""
        if self._is_base_config():
            return None
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or name not in methods:
            return None
        source_spec = methods[name]

        existing_keys = set(methods.keys())
        base = f"{name}_copy"
        if base not in existing_keys:
            dup_name = base
        else:
            n = 2
            while f"{base}_{n}" in existing_keys:
                n += 1
            dup_name = f"{base}_{n}"

        new_methods = {}
        for k, v in methods.items():
            new_methods[k] = v
            if k == name:
                new_methods[dup_name] = copy.deepcopy(source_spec)

        self.raw_data["methods"] = new_methods
        self._mark_dirty()
        self._render_boxes()
        return dup_name

    def _get_group_default_method(self) -> tuple[str | None, dict | None]:
        """Return (method_name, method_spec) for the default method declared in group's _base.yaml."""
        base_data = None
        if self.current_path:
            local_base = self.current_path.parent / "_base.yaml"
            if local_base.is_file():
                try:
                    base_data = yaml.safe_load(local_base.read_text(encoding="utf-8")) or {}
                except Exception:
                    pass
        if not base_data:
            tree = config_tree()
            group = self._group_name()
            if tree and group:
                base_data = _read_group_base(tree, group)
        if base_data:
            base_methods = base_data.get("methods", {})
            if isinstance(base_methods, dict):
                for k, v in base_methods.items():
                    if k != "params" and isinstance(v, dict):
                        return k, copy.deepcopy(v)
        return None, None

    def _start_add_new_method(self):
        """Initiate adding a new method card with focus on the name editor."""
        if self._is_base_config():
            methods = self.raw_data.get("methods", {})
            if any(k != "params" and isinstance(v, dict) for k, v in methods.items()):
                return
            self._pending_new_method = True
            self._pending_new_method_spec = {}
            self._render_boxes()
            return

        def_name, def_spec = self._get_group_default_method()
        self._pending_new_method = True
        self._pending_new_method_spec = copy.deepcopy(def_spec) if def_spec else {}
        self._render_boxes()

    def _cancel_new_method(self):
        """Cancel creating a new method."""
        self._pending_new_method = False
        self._pending_new_method_spec = {}
        self._render_boxes()

    def add_method(self, agent=None, model=None, name=None):
        """Add a method to the open experiment or base defaults. Returns its name, or None."""
        if self._is_base_config():
            methods = self.raw_data.setdefault("methods", {})
            if any(k != "params" and isinstance(v, dict) for k, v in methods.items()):
                return None
            meth_name = name or "default_method"
            spec = {}
            if agent:
                spec["agent"] = agent
            if model:
                spec["model"] = model
            methods[meth_name] = spec
            self._mark_dirty()
            self._render_boxes()
            return meth_name

        def_name, def_spec = self._get_group_default_method()
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm() or "online_rl"
        tree = config_tree()
        p_obj = tree.paradigms.get(paradigm) if tree and paradigm in tree.paradigms else None
        default_model = (
            (def_spec.get("model") if def_spec else None)
            or (p_obj.raw.get("default_model") if p_obj else None)
            or "dnn"
        )

        methods = self.raw_data.setdefault("methods", {})
        if name:
            meth_name = name
        elif agent or model:
            from .config_model import unique_method_name
            meth_name = unique_method_name(methods, agent, model or default_model)
        elif def_name and def_name not in methods:
            meth_name = def_name
        else:
            base = "method"
            if base not in methods:
                meth_name = base
            else:
                n = 2
                while f"{base}_{n}" in methods:
                    n += 1
                meth_name = f"{base}_{n}"

        spec = copy.deepcopy(def_spec) if def_spec else {}
        if agent:
            spec["agent"] = agent
        if model:
            spec["model"] = model
        methods[meth_name] = spec
        self._mark_dirty()
        self._render_boxes()
        return meth_name


    def _on_experiment_env_changed(self, value):
        if self._block_updates:
            return
        selected = self.env_selection()
        self._set_experiment_env(selected)
        self._mark_dirty()
        self._render_boxes()

    def remove_method(self, name):
        """Remove a method from the open experiment. Returns True if it went."""
        if not remove_method_from(self.raw_data, name):
            return False
        self._mark_dirty()
        self._render_boxes()
        return True

    def rename_method(self, old_name: str, new_name: str) -> bool:
        """Rename a method in the open experiment while preserving dict ordering."""
        if not old_name or not new_name or old_name == new_name:
            return False
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or old_name not in methods:
            return False
        new_methods = {}
        for k, v in methods.items():
            if k == old_name:
                new_methods[new_name] = v
            else:
                new_methods[k] = v
        self.raw_data["methods"] = new_methods
        self._mark_dirty()
        self._render_boxes()
        return True

    def reorder_method(self, method_name: str, target_index: int) -> bool:
        """Reorder method cards by moving method_name to target_index."""
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or method_name not in methods:
            return False
        has_params = "params" in methods
        params_val = methods.get("params")

        method_keys = [k for k in methods.keys() if k != "params"]
        if method_name not in method_keys:
            return False
        method_keys.remove(method_name)
        target_index = max(0, min(target_index, len(method_keys)))
        method_keys.insert(target_index, method_name)

        new_methods = {}
        if has_params:
            new_methods["params"] = params_val
        for k in method_keys:
            new_methods[k] = methods[k]

        self.raw_data["methods"] = new_methods
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
            env_name = self._get_explicit_env()
            if isinstance(env_name, str) and env_name in tree.environments:
                if not rules.permits_environment(tree.environments[env_name]):
                    self._set_experiment_env(None)
            self._pin_fields_the_paradigm_forbids(rules)
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
        parsed_val = _parse_yaml_value(value) if isinstance(value, str) else value
        if isinstance(m_spec.get("model"), dict):
            model_name = next(iter(m_spec["model"]))
            if isinstance(m_spec["model"].get(model_name), dict) and param_key in m_spec["model"][model_name]:
                m_spec["model"][model_name][param_key] = parsed_val
                self._mark_dirty()
                return
        m_spec[param_key] = parsed_val
        self._mark_dirty()

    def _on_method_nested_param_edited(self, method_name: str, parent_key: str, child_key: str, value: Any):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        parsed_val = _parse_yaml_value(value) if isinstance(value, str) else value
        if isinstance(m_spec.get("model"), dict):
            model_name = next(iter(m_spec["model"]))
            m_dict = m_spec["model"].setdefault(model_name, {})
            if parent_key in m_dict and isinstance(m_dict[parent_key], dict):
                m_dict[parent_key][child_key] = parsed_val
                self._mark_dirty()
                return
        if parent_key not in m_spec or not isinstance(m_spec[parent_key], dict):
            m_spec[parent_key] = {}
        m_spec[parent_key][child_key] = parsed_val
        self._mark_dirty()

    def _remove_method_nested_param(self, method_name: str, parent_key: str, child_key: str):
        methods = self.raw_data.get("methods", {})
        m_spec = methods.get(method_name, {})
        if isinstance(m_spec.get("model"), dict):
            for m_dict in m_spec["model"].values():
                if isinstance(m_dict, dict) and parent_key in m_dict and isinstance(m_dict[parent_key], dict):
                    m_dict[parent_key].pop(child_key, None)
        if parent_key in m_spec and isinstance(m_spec[parent_key], dict):
            m_spec[parent_key].pop(child_key, None)
        self._mark_dirty()
        self._render_boxes()

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
        if "." in key:
            parts = key.split(".")
            target = params
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = _parse_yaml_value(value)
        else:
            params[key] = _parse_yaml_value(value)
        self._mark_dirty()

    def _remove_universal_param(self, key):
        if "methods" in self.raw_data and "params" in self.raw_data["methods"]:
            params = self.raw_data["methods"]["params"]
            if "." in key:
                parts = key.split(".")
                target = params
                for p in parts[:-1]:
                    if isinstance(target, dict) and p in target:
                        target = target[p]
                    else:
                        target = None
                        break
                if isinstance(target, dict):
                    target.pop(parts[-1], None)
                if len(parts) > 1 and parts[0] in params and isinstance(params[parts[0]], dict) and not params[parts[0]]:
                    params.pop(parts[0], None)
            else:
                params.pop(key, None)
            if not self.raw_data["methods"]["params"]:
                self.raw_data["methods"].pop("params", None)
            self._mark_dirty()
            self._render_boxes()

    def _add_universal_param(self, key, value=None):
        if not key:
            return
        if value is None or value == "":
            value = self._resolve_default_param_value("", key)
        methods = self.raw_data.setdefault("methods", {})
        params = methods.setdefault("params", {})
        if "." in key:
            parts = key.split(".")
            target = params
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = _parse_yaml_value(value)
        else:
            params[key] = _parse_yaml_value(value)
        self._mark_dirty()
        self._render_boxes()

    def _resolve_default_param_value(self, method_name: str, param_key: str):
        """Look up hierarchical default for param_key:
        1. methods.params in current file
        2. Group _base.yaml (methods.params or methods[method_name])
        3. in/config/agent/<agent>.yaml
        4. in/config/model/<model>.yaml
        5. Known defaults fallback
        """
        methods = self.raw_data.get("methods")
        if isinstance(methods, dict):
            univ_params = methods.get("params")
            if isinstance(univ_params, dict) and param_key in univ_params:
                return univ_params[param_key]

        m_spec = (
            methods.get(method_name)
            if isinstance(methods, dict) and isinstance(methods.get(method_name), dict)
            else {}
        )
        agent_raw = m_spec.get("agent")
        agent_name = (
            (agent_raw.get("name") or agent_raw.get("algorithm") or next((k for k in agent_raw if k != "name"), None))
            if isinstance(agent_raw, dict)
            else agent_raw
        )
        model_raw = m_spec.get("model")
        model_name = next(iter(model_raw)) if isinstance(model_raw, dict) else model_raw

        tree = config_tree()
        group = self._group_name()

        # 2. Group _base.yaml
        if tree and group:
            base_data = _read_group_base(tree, group)
            base_methods = base_data.get("methods")
            if isinstance(base_methods, dict):
                base_params = base_methods.get("params")
                if isinstance(base_params, dict) and param_key in base_params:
                    return base_params[param_key]
                if method_name in base_methods and isinstance(base_methods[method_name], dict):
                    if param_key in base_methods[method_name]:
                        return base_methods[method_name][param_key]

        # 3. Agent YAML: in/config/agent/<agent>.yaml
        if tree and agent_name:
            agent_params = tree.agent_hyperparameters(agent_name)
            if param_key in agent_params:
                return agent_params[param_key]

        # 4. Model YAML: in/config/model/<model>.yaml
        if tree and model_name:
            model_params = tree.model_parameters(model_name)
            if param_key in model_params:
                return model_params[param_key]

        # 5. Generic fallback
        known = {
            "lr": 0.0003,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "epochs": 25,
            "eval_every_n_epochs": 5,
            "weight_decay": 0.0001,
        }
        if param_key in known:
            return known[param_key]

        if "lr" in param_key:
            return 0.0003
        if "coef" in param_key or "alpha" in param_key:
            return 0.01
        return 0.0

    def _resolve_default_subcomponent_param_value(self, method_name: str, subcomp_name: str, param_key: str) -> Any:
        methods = self.raw_data.get("methods", {})
        m_spec = methods.get(method_name, {}) if isinstance(methods, dict) else {}
        tree = config_tree()
        sub_dict = {}
        if isinstance(m_spec.get("model"), dict):
            for arch_dict in m_spec["model"].values():
                if isinstance(arch_dict, dict) and subcomp_name in arch_dict:
                    sub_dict = arch_dict[subcomp_name] if isinstance(arch_dict[subcomp_name], dict) else {}
                    break
        if not sub_dict and isinstance(m_spec.get(subcomp_name), dict):
            sub_dict = m_spec[subcomp_name]

        sub_type = sub_dict.get("type") or sub_dict.get("architecture") or sub_dict.get("reasoner")
        if not sub_type:
            if isinstance(m_spec.get(subcomp_name), str):
                sub_type = m_spec[subcomp_name]
            elif tree and model_name and model_name in tree.models:
                slot_dict = tree.models[model_name].parameters.get(subcomp_name)
                if isinstance(slot_dict, dict):
                    sub_type = slot_dict.get("type") or slot_dict.get("architecture") or subcomp_name
                    if param_key in slot_dict:
                        return slot_dict[param_key]
                elif isinstance(slot_dict, str):
                    sub_type = slot_dict

        if tree and sub_type and sub_type in tree.models:
            model_params = tree.model_parameters(sub_type)
            if param_key in model_params:
                return model_params[param_key]

        fallbacks = {
            "hidden_sizes": [256, 256],
            "activation": "relu",
            "ecm_dthr": 0.03,
            "rules": "default",
            "logic_lr": 0.001,
            "valuation": "default",
            "fyd": True,
            "fyd_top_k": 3,
            "mode": "neural",
            "blend_function": "softmax",
        }
        return fallbacks.get(param_key, 0.0)

    def _add_method_subcomponent(self, method_name: str, subcomp_name: str):
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        tree = config_tree()
        model_val = m_spec.get("model")
        model_name = next(iter(model_val)) if isinstance(model_val, dict) else str(model_val or "dnn")

        tree = config_tree()
        subcomps = tree.model_subcomponents(model_name) if tree else {}
        sub_default = dict(subcomps.get(subcomp_name, {}))
        if not sub_default:
            sub_default = {"type": "dnn"}

        if isinstance(m_spec.get("model"), dict):
            m_spec["model"].setdefault(model_name, {})[subcomp_name] = sub_default
        else:
            m_spec[subcomp_name] = sub_default

        self._mark_dirty()
        self._render_boxes()

    def _remove_method_subcomponent(self, method_name: str, subcomp_name: str):
        methods = self.raw_data.get("methods", {})
        m_spec = methods.get(method_name, {})
        if isinstance(m_spec.get("model"), dict):
            for arch_dict in m_spec["model"].values():
                if isinstance(arch_dict, dict):
                    arch_dict.pop(subcomp_name, None)
        m_spec.pop(subcomp_name, None)
        self._mark_dirty()
        self._render_boxes()

    def _on_method_subcomponent_type_changed(self, method_name: str, subcomp_name: str, new_type: str):
        if self._block_updates or not new_type:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})

        container = None
        if isinstance(m_spec.get("model"), dict):
            for arch_dict in m_spec["model"].values():
                if isinstance(arch_dict, dict) and subcomp_name in arch_dict:
                    container = arch_dict
                    break
        if container is None:
            if subcomp_name in m_spec:
                container = m_spec
            elif isinstance(m_spec.get("model"), dict):
                model_name = next(iter(m_spec["model"]))
                container = m_spec["model"].setdefault(model_name, {})
            else:
                container = m_spec

        cur_val = container.get(subcomp_name)
        if isinstance(cur_val, str):
            container[subcomp_name] = new_type
            old_submodel = next((k for k in cur_val if (self.tree and k in self.tree.models)), None)
            if old_submodel:
                params = cur_val.pop(old_submodel)
                container[subcomp_name] = {new_type: params if isinstance(params, dict) else {}}
            elif "type" in cur_val:
                cur_val["type"] = new_type
            else:
                container[subcomp_name] = new_type
        else:
            container[subcomp_name] = new_type

        self._mark_dirty()
        self._render_boxes()

    def _on_method_subcomponent_param_edited(self, method_name: str, subcomp_name: str, key: str, value: Any):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        target = None
        if isinstance(m_spec.get("model"), dict):
            for arch_dict in m_spec["model"].values():
                if isinstance(arch_dict, dict) and subcomp_name in arch_dict:
                    if isinstance(arch_dict[subcomp_name], dict):
                        target = arch_dict[subcomp_name]
                    elif isinstance(arch_dict[subcomp_name], str):
                        arch_dict[subcomp_name] = {"type": arch_dict[subcomp_name]}
                        target = arch_dict[subcomp_name]
                    break
        if target is None:
            if subcomp_name not in m_spec:
                m_spec[subcomp_name] = {}
            elif isinstance(m_spec[subcomp_name], str):
                m_spec[subcomp_name] = {"type": m_spec[subcomp_name]}
            target = m_spec[subcomp_name]

        if self.tree:
            for sub_name in self.tree.models:
                if sub_name in target and isinstance(target[sub_name], dict):
                    target[sub_name][key] = _parse_yaml_value(value)
                    self._mark_dirty()
                    return

        target[key] = _parse_yaml_value(value)
        self._mark_dirty()

    def _remove_method_subcomponent_param(self, method_name: str, subcomp_name: str, key: str):
        methods = self.raw_data.get("methods", {})
        m_spec = methods.get(method_name, {})
        target = None
        if isinstance(m_spec.get("model"), dict):
            for arch_dict in m_spec["model"].values():
                if isinstance(arch_dict, dict) and subcomp_name in arch_dict and isinstance(arch_dict[subcomp_name], dict):
                    target = arch_dict[subcomp_name]
                    break
        if target is None and subcomp_name in m_spec and isinstance(m_spec[subcomp_name], dict):
            target = m_spec[subcomp_name]
        if target is not None:
            if self.tree:
                for sub_name in self.tree.models:
                    if sub_name in target and isinstance(target[sub_name], dict):
                        target[sub_name].pop(key, None)
                        self._mark_dirty()
                        self._render_boxes()
                        return
            target.pop(key, None)
            self._mark_dirty()
            self._render_boxes()

    def _add_method_subcomponent_param(self, method_name: str, subcomp_name: str, key: str, value: Any = None):
        if not key:
            return
        if value is None or value == "":
            value = self._resolve_default_subcomponent_param_value(method_name, subcomp_name, key)
        self._on_method_subcomponent_param_edited(method_name, subcomp_name, key, value)
        self._render_boxes()

    def _remove_method_param(self, method_name, param_key):
        if "methods" in self.raw_data and method_name in self.raw_data["methods"]:
            m_spec = self.raw_data["methods"][method_name]
            if isinstance(m_spec.get("model"), dict):
                model_name = next(iter(m_spec["model"]))
                if isinstance(m_spec["model"].get(model_name), dict):
                    m_spec["model"][model_name].pop(param_key, None)
            m_spec.pop(param_key, None)
            self._mark_dirty()
            self._render_boxes()

    def _add_method_param(self, method_name, param_key, value=None):
        if not param_key:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        model_name = None
        if isinstance(m_spec.get("model"), str):
            model_name = m_spec["model"]
        elif isinstance(m_spec.get("model"), dict):
            model_name = next(iter(m_spec["model"]), None)

        if self.tree and model_name and param_key in self.tree.model_slots(model_name):
            self._add_method_subcomponent(method_name, param_key)
            return

        if value is None or value == "":
            value = self._resolve_default_param_value(method_name, param_key)
        parsed_val = _parse_yaml_value(value)
        if isinstance(m_spec.get("model"), dict) and model_name:
            if self.tree and model_name in self.tree.models and param_key in self.tree.model_parameters(model_name):
                m_spec["model"].setdefault(model_name, {})[param_key] = parsed_val
                self._mark_dirty()
                self._render_boxes()
                return
        m_spec[param_key] = parsed_val
        self._mark_dirty()
        self._render_boxes()

    def _add_experiment_param(self, key: str, default_val: Any = None):
        """Add an experiment-level parameter override to the configuration."""
        if not key:
            return
        if default_val is not None:
            self.raw_data[key] = default_val
        elif key == "total_timesteps":
            self.raw_data[key] = int(self._get_resolved_defaults().get("total_timesteps") or 10000)
        elif key == "seed":
            res_seed = self._get_resolved_defaults().get("seed")
            self.raw_data[key] = int(res_seed if res_seed is not None else 42)
        elif key == "eval_episodes":
            self.raw_data[key] = int(self._get_resolved_defaults().get("eval_episodes") or 100)
        elif key == "eval_freq_steps":
            self.raw_data[key] = int(self._get_resolved_defaults().get("eval_freq_steps") or 10000)
        elif key in ("tensorboard", "save_dataset", "recover", "no_plot"):
            self.raw_data[key] = True
        else:
            self.raw_data[key] = ""
        self._mark_dirty()
        self._render_boxes()

    def _prompt_add_custom_experiment_param(self):
        """Prompt user to add an arbitrary top-level experiment parameter."""
        key, ok = QInputDialog.getText(
            self,
            "Add Experiment Parameter",
            "Enter parameter name (e.g. eval_freq_steps, num_envs):"
        )
        if not ok or not key.strip():
            return
        key = key.strip()
        val_str, ok2 = QInputDialog.getText(
            self,
            "Initial Value",
            f"Enter initial value for '{key}':",
            text="1000" if key.endswith(("_steps", "_episodes", "_count")) else ""
        )
        if not ok2:
            return
        typed_val = _parse_yaml_value(val_str)
        self.raw_data[key] = typed_val
        self._mark_dirty()
        self._render_boxes()

    def flush_pending_save(self):
        """Write a pending auto-save now (before anything reads the file from disk, or on close)."""
        self._autosave_timer.stop()
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

    def _typing_in_text_field(self) -> bool:
        """True while the user is typing into one of this viewer's text boxes."""
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QPlainTextEdit, QTextEdit)) and self.isAncestorOf(focus)

    def _mark_dirty(self):
        self.is_dirty = True
        if getattr(self, "auto_save", False) and self._typing_in_text_field():
            self._autosave_timer.start()  # restarts on each keystroke; saves once typing pauses
        elif getattr(self, "auto_save", False):
            self._autosave_timer.stop()  # this save covers any pending text edit
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

    #: The only value each rollout field may take under a paradigm that forbids it.
    _FORBIDDEN_FIELD_VALUES = {"eval_episodes": 0}

    def _pin_fields_the_paradigm_forbids(self, rules):
        """Write legal values for the fields the chosen paradigm disables.

        in/config/config.yaml defaults eval_episodes to 100 for the online case.
        A config that simply omits it inherits that and is rejected before
        training starts, which is why the shipped offline and supervised bases pin it.
        Marking the field "not used" in the panel is not enough on its own:
        the inherited value still reaches the backend.

        Switching back to a paradigm that allows a field clears the pin, but only
        while it still holds the value set here, so a deliberate choice survives.
        """
        for name, forced in self._FORBIDDEN_FIELD_VALUES.items():
            if not rules.field_enabled(name):
                self.raw_data[name] = forced
            elif self.raw_data.get(name) == forced:
                self.raw_data.pop(name, None)

    def _reconcile_group_base(self):
        """Re-pin a group base against whatever paradigm it currently declares.

        _on_paradigm_changed covers picking a paradigm, but not a base that
        already had one - an older group, or one where only the environment was
        changed. Those keep inheriting eval_episodes: 100 and stay unrunnable,
        so the base is reconciled whenever it is saved.

        Only _base.yaml: the pin belongs on the file that binds the paradigm,
        and experiments inherit it.
        """
        if not self.current_path or self.current_path.name != "_base.yaml":
            return
        paradigm = self.raw_data.get("paradigm")
        tree = config_tree()
        if tree and isinstance(paradigm, str) and paradigm in tree.paradigms:
            self._pin_fields_the_paradigm_forbids(tree.paradigms[paradigm])

    def _drop_empty_methods(self):
        """Remove a methods block with nothing in it.

        Preserves methods if it has non-empty universal params or concrete methods.
        """
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict):
            return
        if "params" in methods and not methods["params"]:
            methods.pop("params", None)
        if not methods:
            self.raw_data.pop("methods", None)

    def save_to_disk(self, show_error: bool = False):
        """Save current YAML dictionary back to disk."""
        if not self.current_path:
            return False
        try:
            self._clean_redundant_params()
            self._reconcile_group_base()
            self._drop_empty_methods()
            yaml_str = yaml.safe_dump(self.raw_data, sort_keys=False)
            self.current_path.write_text(self._preamble + yaml_str, encoding="utf-8")
            self.is_dirty = False
            self._update_dirty_ui()
            if self.current_rel_path:
                self.dirty_state_changed.emit(self.current_rel_path, False)
            invalidate_config_tree()
            self.save_requested.emit()
            return True
        except Exception as exc:
            if show_error:
                QMessageBox.critical(self, "Save Error", f"Could not save file to disk:\n{exc}")
            return False

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

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
        for group in ("paradigm", "env", "site"):
            value = self._get_explicit_env() if group == "env" else data.get(group)
            if isinstance(value, str) and value and value != self.INHERIT:
                overrides.append(f"{group}={value}")

        if "seed" in data:
            overrides.append(f"seed={data['seed']}")
        if "total_timesteps" in data:
            overrides.append(f"total_timesteps={data['total_timesteps']}")
        if "epochs" in data:
            overrides.append(f"epochs={data['epochs']}")
        if "eval_every_n_epochs" in data:
            overrides.append(f"eval_every_n_epochs={data['eval_every_n_epochs']}")
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
