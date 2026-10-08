"""Organized, tabbed Settings view with side-tab navigation and ASCII Theta overlay."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .about import AsciiTheta, AnimationSettingsDialog
from .hotkeys import parse_action_key
from .theme import current_theme, theme_color
from .widgets import ToggleSlider, label

if TYPE_CHECKING:
    from .app import Window

ICON_DIR = Path(__file__).resolve().parent / "icons"


def get_settings_icon(name: str, size: int = 20) -> QIcon:
    """Render icons/<name>.svg with current theme's text and accent colors."""
    p = Path(name)
    if p.exists() and p.is_file():
        source_path = p
    elif (ICON_DIR / f"{name}.svg").exists():
        source_path = ICON_DIR / f"{name}.svg"
    elif (ICON_DIR / name).exists():
        source_path = ICON_DIR / name
    else:
        return QIcon()

    source = source_path.read_text(encoding="utf-8")
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    icon = QIcon()

    colors = {
        (QIcon.Mode.Normal, QIcon.State.Off): "muted",
        (QIcon.Mode.Active, QIcon.State.Off): "text",
        (QIcon.Mode.Normal, QIcon.State.On): "accent",
        (QIcon.Mode.Active, QIcon.State.On): "accent",
    }
    for (mode, state), role in colors.items():
        rendered_src = source.replace("currentColor", theme_color(role)).encode("utf-8")
        renderer = QSvgRenderer(QByteArray(rendered_src))
        pixmap = QPixmap(round(size * ratio), round(size * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        icon.addPixmap(pixmap, mode, state)
    return icon


TAB_SPECS = [
    {
        "id": "appearance",
        "title": "Appearance",
        "subtitle": "Themes, sidebar panel navigation & display options",
        "icon": "appearance",
    },
    {
        "id": "core_plugins",
        "title": "Core Plugins",
        "subtitle": "Built-in core extensions bundled natively into Theta-IDE",
        "icon": "plugins_core",
    },
    {
        "id": "community_plugins",
        "title": "Community Plugins",
        "subtitle": "Installed plugins, extensions, and Community Hub",
        "icon": "plugins_community",
    },
    {
        "id": "hotkeys",
        "title": "Hotkeys",
        "subtitle": "Action-key leader navigation & keyboard shortcuts",
        "icon": "hotkeys",
    },
    {
        "id": "backend",
        "title": "Backend API",
        "subtitle": "FastAPI daemon connection and server status",
        "icon": "backend",
    },
    {
        "id": "storage",
        "title": "Workspace & Storage",
        "subtitle": "Data roots, UI layout reset, and settings.toml",
        "icon": "storage",
    },
]
 
DEFAULT_PANES = [
    "settings",
    "components",
    "config",
    "monitor",
    "results",
    "plots",
    "tensorboard",
    "queue",
    "terminal",
    "console",
]

AVAILABLE_PANES = [
    ("settings", "Settings & About"),
    ("components", "Components"),
    ("config", "Experiment Builder"),
    ("monitor", "Training Monitor"),
    ("results", "Results Browser"),
    ("plots", "Plot Viewer"),
    ("tensorboard", "TensorBoard"),
    ("queue", "Job Queue"),
    ("terminal", "Terminal"),
    ("console", "Console"),
    ("none", "— Unassigned / Disabled —"),
]

ACTION_KEY_PRESETS = [
    ("ctrl+b", "Ctrl + B / Caps Lock (tmux default)"),
    ("caps_lock", "Caps Lock (fast modal key)"),
    ("ctrl+space", "Ctrl + Space"),
    ("alt+space", "Alt + Space"),
    ("ctrl+a", "Ctrl + A (Screen default)"),
    ("ctrl+x", "Ctrl + X (Emacs default)"),
    ("alt", "Alt / Option (hold chord)"),
    ("ctrl", "Control (hold chord)"),
    ("meta", "Command / Meta (⌘ / Win)"),
    ("shift", "Shift (hold chord)"),
]

DEFAULT_MENU_SHORTCUTS = [
    ("launch_training", "Launch training", "F5"),
    ("stop_run", "Stop training", "Shift+F5"),
    ("new_experiment", "New experiment", "Ctrl+N"),
    ("export_config", "Export draft YAML…", "Ctrl+Shift+S"),
    ("save_config", "Save configuration", "Ctrl+S"),
    ("add_to_queue", "Add to queue", "Ctrl+Shift+Q"),
    ("toggle_queue", "Start or pause queue", "Ctrl+Shift+R"),
    ("start_demo", "Start simulated demo", "Ctrl+F5"),
    ("quit", "Quit", "Ctrl+Q"),
]


class SettingsSideTabButton(QToolButton):
    """Button in the left settings navigation bar."""

    def __init__(self, tab_id: str, title: str, icon_name: str, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.title_text = title
        self.icon_name = icon_name
        self.setObjectName("settingsSideTab")
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText(f"  {title}")
        self.setToolTip(f"Configure {title}")
        self.setIconSize(QSize(20, 20))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_icon()

    def refresh_icon(self):
        icon = get_settings_icon(self.icon_name, 20)
        if not icon.isNull():
            self.setIcon(icon)


class AsciiThetaSplash(QWidget):
    """Base welcome view: prominent rotating ASCII Theta sculpture and branding."""

    def __init__(self, window: Window, parent=None):
        super().__init__(parent)
        self.window = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        # Header branding
        header_row = QHBoxLayout()
        title_lbl = label("ThetaIDE", "brand")
        title_lbl.setStyleSheet("font-size: 26px; font-weight: 700;")
        header_row.addWidget(title_lbl)
        header_row.addStretch()

        self.anim_settings_btn = QPushButton("Animation Settings")
        self.anim_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.anim_settings_btn.setToolTip("Configure ASCII Theta animation speed, size, thickness, and 3D effects")
        self.anim_settings_btn.clicked.connect(self.open_animation_settings)
        header_row.addWidget(self.anim_settings_btn)
        self.anim_toggle_btn = self.anim_settings_btn  # Backward-compatible alias
        layout.addLayout(header_row)

        # 3D ASCII Sculpture
        self.ascii_sculpture = AsciiTheta(self, settings_manager=getattr(window, "settings_manager", None))
        layout.addWidget(self.ascii_sculpture, 1)

    def open_animation_settings(self):
        dialog = AnimationSettingsDialog(
            target=self.ascii_sculpture,
            settings_manager=getattr(self.window, "settings_manager", None),
            parent=self,
        )
        dialog.exec()


TAB_SIZES = {
    "appearance": (840, 680),
    "core_plugins": (780, 540),
    "community_plugins": (860, 640),
    "hotkeys": (940, 760),
    "backend": (760, 460),
    "storage": (780, 480),
}


class DynamicStackedWidget(QStackedWidget):
    """QStackedWidget that constrains horizontal minimumSizeHint to 0 to prevent child sizeHint blowup in QScrollArea."""

    def minimumSizeHint(self) -> QSize:
        curr = self.currentWidget()
        h = curr.minimumSizeHint().height() if curr is not None else 0
        return QSize(0, h)

    def sizeHint(self) -> QSize:
        curr = self.currentWidget()
        h = curr.sizeHint().height() if curr is not None else 0
        return QSize(0, h)


class SettingsDetailWindow(QFrame):
    """Settings page overlay displaying the active tab's configuration over the ASCII Theta."""

    closeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsOverlayLayer")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.current_tab_id = "appearance"
        self.target_size = (840, 680)

        # Top-left layout: Page is docked flush with the sidebar & top, leaving margins to the right & underneath
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Settings Page Frame
        self.card = QFrame(self)
        self.card.setObjectName("settingsDetailCard")

        self.shadow = QGraphicsDropShadowEffect(self.card)
        self.shadow.setBlurRadius(24)
        self.shadow.setColor(QColor(0, 0, 0, 140))
        self.shadow.setOffset(4, 4)
        self.card.setGraphicsEffect(self.shadow)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(14)

        # Settings Window Header
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(28, 28)
        self.icon_label.setScaledContents(True)
        header_row.addWidget(self.icon_label)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        self.title_label = label("Settings", "heading")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        title_layout.addWidget(self.title_label)

        self.subtitle_label = label("", "muted")
        self.subtitle_label.setStyleSheet("font-size: 12px;")
        title_layout.addWidget(self.subtitle_label)
        header_row.addLayout(title_layout, 1)

        # Prominent Close button
        self.btn_close = QToolButton()
        self.btn_close.setObjectName("settingsCloseButton")
        self.btn_close.setText("Close")
        self.btn_close.setToolTip("Close settings (Return to ASCII Theta)")
        self.btn_close.setFixedHeight(30)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.closeRequested.emit)
        header_row.addWidget(self.btn_close)

        card_layout.addLayout(header_row)

        # Subtle separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("settingsSeparator")
        card_layout.addWidget(sep)

        # Content Area: Scrollable stacked pages (vertical only, no horizontal scrollbars)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setObjectName("settingsScrollArea")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.pages_stack = DynamicStackedWidget()
        self.scroll_area.setWidget(self.pages_stack)
        card_layout.addWidget(self.scroll_area, 1)

        outer_layout.addWidget(self.card)

    def set_header(self, title: str, subtitle: str, icon_name: str):
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        icon = get_settings_icon(icon_name, 28)
        if not icon.isNull():
            pix = icon.pixmap(28, 28)
            self.icon_label.setPixmap(pix)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)

    def apply_tab_size(self, tab_id: str):
        self.current_tab_id = tab_id
        self.target_size = TAB_SIZES.get(tab_id, (840, 680))
        self._update_card_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_card_geometry()

    def _update_card_geometry(self):
        if not hasattr(self, "card"):
            return
        tw, th = getattr(self, "target_size", (840, 680))
        avail_w = self.width()
        avail_h = self.height()
        if avail_w > 0 and avail_h > 0:
            # Leave margin to the right and underneath, clamping gracefully on small viewports
            cw = min(tw, max(360, avail_w - 60))
            ch = min(th, max(280, avail_h - 60))
            self.card.setFixedSize(cw, ch)

    def mousePressEvent(self, event):
        if hasattr(self, "card") and not self.card.geometry().contains(event.pos()):
            self.closeRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        # Translucent overlay layer leaves right and bottom margins clear so ASCII art is visible
        super().paintEvent(event)


def normalize_hotkey_combo(text: str) -> str:
    """Normalize user input (e.g. '1', 'Action + 1', 't', 'Action + T') to canonical 'Action + <KEY>' or empty."""
    cleaned = text.strip()
    if not cleaned or cleaned.lower() in ("none", "unassigned", "-", "disabled", "clear", ""):
        return ""
    lower = cleaned.lower()
    if lower.startswith("action"):
        cleaned = cleaned[6:]
    elif lower.startswith("leader"):
        cleaned = cleaned[6:]
    # Strip any leading/trailing spaces and plus signs (e.g. " + 1" -> "1", "+ C" -> "C")
    cleaned = cleaned.strip(" +").strip()
    cleaned = cleaned.upper()
    if not cleaned:
        return ""
    return f"Action + {cleaned}"


class HotkeyComboEdit(QLineEdit):
    """Interactive hotkey input field for pane navigation chords.

    Accepts keyboard input directly (e.g. pressing '1' sets 'Action + 1',
    pressing 'T' sets 'Action + T', Backspace/Delete clears).
    """

    comboChanged = pyqtSignal(str)

    def __init__(self, canonical_combo: str = "", parent=None):
        super().__init__(parent)
        self._canonical_combo = canonical_combo
        self.setText(canonical_combo)
        self.setPlaceholderText("Unassigned (click to set)")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px; font-weight: 600;")
        self.editingFinished.connect(self._on_editing_finished)

    def get_canonical_combo(self) -> str:
        return self._canonical_combo

    def set_canonical_combo(self, combo: str):
        self._canonical_combo = combo
        self.blockSignals(True)
        self.setText(combo)
        self.blockSignals(False)

    def keyPressEvent(self, event):
        key = event.key()
        # Ignore pure modifiers alone
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            super().keyPressEvent(event)
            return

        # Clear on Backspace or Delete
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.set_canonical_combo("")
            self.comboChanged.emit("")
            return

        # Cancel / exit focus on Escape
        if key == Qt.Key.Key_Escape:
            self.clearFocus()
            return

        # Check for digit (0-9)
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            digit = str(key - Qt.Key.Key_0)
            combo = f"Action + {digit}"
            self.set_canonical_combo(combo)
            self.comboChanged.emit(combo)
            return

        # Check for letter/printable text
        text = event.text().strip().upper()
        if text and len(text) == 1 and (text.isalnum() or text in "+-_/"):
            combo = f"Action + {text}"
            self.set_canonical_combo(combo)
            self.comboChanged.emit(combo)
            return

        super().keyPressEvent(event)

    def _on_editing_finished(self):
        normalized = normalize_hotkey_combo(self.text())
        if normalized != self._canonical_combo:
            self.set_canonical_combo(normalized)
            self.comboChanged.emit(normalized)


class SettingsView(QWidget):
    """Complete, organized settings view with left side-tab and overlay on ASCII Theta."""

    def __init__(self, window: Window, parent=None):
        super().__init__(parent)
        self.window = window
        self.active_tab_id: str | None = None
        self.tab_buttons: Dict[str, SettingsSideTabButton] = {}
        self.pane_hotkey_edits: Dict[str, HotkeyComboEdit] = {}
        self.pane_conflict_labels: Dict[str, QLabel] = {}
        self.menu_shortcut_edits: Dict[str, QKeySequenceEdit] = {}
        self.slider_term: ToggleSlider | None = None
        self.conflict_banner: QFrame | None = None
        self.conflict_banner_label: QLabel | None = None

        # Main horizontal layout: Side-tab on the left, Content stack on the right
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 1. Left Side-Tab ───────────────────────────────────────────────────
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("settingsSidebar")
        self.sidebar_frame.setFixedWidth(220)

        sb_layout = QVBoxLayout(self.sidebar_frame)
        sb_layout.setContentsMargins(10, 16, 10, 14)
        sb_layout.setSpacing(6)

        # Sidebar title
        sb_title = label("Settings", "heading")
        sb_title.setStyleSheet("font-size: 16px; font-weight: 700; padding-left: 6px; margin-bottom: 6px;")
        sb_layout.addWidget(sb_title)

        # Navigation tabs list
        for spec in TAB_SPECS:
            tid = spec["id"]
            btn = SettingsSideTabButton(tid, spec["title"], spec["icon"], self.sidebar_frame)
            btn.clicked.connect(lambda _, t=tid: self.on_tab_button_clicked(t))
            self.tab_buttons[tid] = btn
            sb_layout.addWidget(btn)

        sb_layout.addStretch()

        layout.addWidget(self.sidebar_frame)

        # ── 2. Right Content Area: Stacked ASCII Splash (0) & Settings (1) ────
        self.content_stack = QStackedWidget(self)
        self.content_stack.setObjectName("settingsContentStack")
        self.content_stack.layout().setStackingMode(QStackedLayout.StackingMode.StackAll)

        # Page 0: ASCII Sculpture Splash view (default view on enter)
        self.ascii_splash = AsciiThetaSplash(self.window, self.content_stack)
        self.content_stack.addWidget(self.ascii_splash)

        # Bind Window references so window.settings_ascii, window.anim_settings_btn, and window.anim_toggle_btn work
        self.window.settings_ascii = self.ascii_splash.ascii_sculpture
        self.window.anim_settings_btn = self.ascii_splash.anim_settings_btn
        self.window.anim_toggle_btn = self.ascii_splash.anim_toggle_btn

        # Page 1: Settings Detail Window (appears on top when a tab is clicked)
        self.settings_window = SettingsDetailWindow(self.content_stack)
        self.settings_window.closeRequested.connect(self.hide_overlay)
        self.content_stack.addWidget(self.settings_window)

        # Start on Page 0 (ASCII Theta) with overlay hidden
        self.settings_window.setVisible(False)
        self.content_stack.setCurrentIndex(0)

        layout.addWidget(self.content_stack, 1)

        # ── 3. Build Pages inside the Settings Window ─────────────────────────
        self.page_widgets: Dict[str, QWidget] = {}
        self.build_pages()

        # Apply initial styling
        self.refresh_styles()

        # Keep hotkeys & settings in sync with live settings changes
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.changed.connect(self.sync_from_settings)

    # ── Page Builders ────────────────────────────────────────────────────────

    def build_pages(self):
        """Construct the settings pages inside the settings window stack."""
        stack = self.settings_window.pages_stack

        # Page 0: Appearance
        p_appearance = self._build_appearance_page()
        stack.addWidget(p_appearance)
        self.page_widgets["appearance"] = p_appearance

        # Page 1: Core Plugins
        p_core = self._build_core_plugins_page()
        stack.addWidget(p_core)
        self.page_widgets["core_plugins"] = p_core

        # Page 2: Community Plugins
        p_comm = self._build_community_plugins_page()
        stack.addWidget(p_comm)
        self.page_widgets["community_plugins"] = p_comm

        # Page 3: Hotkeys
        p_hotkeys = self._build_hotkeys_page()
        stack.addWidget(p_hotkeys)
        self.page_widgets["hotkeys"] = p_hotkeys

        # Page 4: Backend API
        p_backend = self._build_backend_page()
        stack.addWidget(p_backend)
        self.page_widgets["backend"] = p_backend

        # Page 5: Workspace & Storage
        p_storage = self._build_storage_page()
        stack.addWidget(p_storage)
        self.page_widgets["storage"] = p_storage

    def _build_appearance_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        # Card 1: Active Theme
        theme_card = QFrame()
        theme_card.setObjectName("card")
        tc_layout = QVBoxLayout(theme_card)
        tc_layout.setContentsMargins(16, 14, 16, 14)
        tc_layout.setSpacing(10)

        tc_layout.addWidget(label("Color Theme & Palette", "cardTitle"))
        tc_sub = label("Select an active color palette or create custom theme roles.", "muted")
        tc_sub.setWordWrap(True)
        tc_layout.addWidget(tc_sub)

        theme_row = QHBoxLayout()
        theme_row.addWidget(label("Active Theme:", "muted"))
        self.window.settings_theme_select = QComboBox()
        for name in self.window.theme_manager.themes():
            self.window.settings_theme_select.addItem(name)
        self.window.settings_theme_select.setCurrentText(self.window.theme_manager.active["name"])
        self.window.settings_theme_select.currentTextChanged.connect(self.window.settings_theme_selected)
        self.window.settings_theme_select.setMaximumWidth(320)
        theme_row.addWidget(self.window.settings_theme_select, 1)

        btn_builder = QPushButton("Customize palette…")
        btn_builder.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_builder.clicked.connect(self.window.show_theme_builder)
        theme_row.addWidget(btn_builder)
        theme_row.addStretch()
        tc_layout.addLayout(theme_row)
        layout.addWidget(theme_card)

        # Card 2: Sidebar Panels
        sidebar_card = QFrame()
        sidebar_card.setObjectName("card")
        sb_layout = QVBoxLayout(sidebar_card)
        sb_layout.setContentsMargins(16, 14, 16, 14)
        sb_layout.setSpacing(10)

        sb_layout.addWidget(label("Sidebar Panels Visibility", "cardTitle"))
        sb_sub = label("Choose which navigation panels are visible in the left sidebar.", "muted")
        sb_sub.setWordWrap(True)
        sb_layout.addWidget(sb_sub)

        panes_grid = QGridLayout()
        panes_grid.setHorizontalSpacing(24)
        panes_grid.setVerticalSpacing(8)

        pane_metadata = [
            ("components", "Components"),
            ("config", "Experiment Builder"),
            ("workflows", "Workflows"),
            ("monitor", "Training Monitor"),
            ("results", "Results Browser"),
            ("plots", "Plot Viewer"),
            ("tensorboard", "TensorBoard"),
            ("queue", "Job Queue"),
            ("terminal", "Terminal"),
            ("console", "Console"),
        ]

        self.window.pane_sliders = {}
        for i, (pid, name) in enumerate(pane_metadata):
            col = 0 if i < 5 else 1
            row_idx = i if i < 5 else i - 5

            item_row = QHBoxLayout()
            item_row.setSpacing(8)
            title_lbl = label(name)
            title_lbl.setWordWrap(True)
            title_lbl.setStyleSheet("font-weight: 500; font-size: 12px;")
            item_row.addWidget(title_lbl, 1)

            slider = ToggleSlider(checked=self.window.tabs.is_tab_visible(pid))
            slider.setToolTip(f"Show or hide {name} in the sidebar")
            slider.setAccessibleName(f"Toggle {name} visibility in sidebar")
            slider.toggled.connect(lambda chk, p=pid: self.window.on_pane_slider_toggled(p, chk))
            self.window.pane_sliders[pid] = slider
            item_row.addWidget(slider)

            panes_grid.addLayout(item_row, row_idx, col)

        sb_layout.addLayout(panes_grid)

        auto_hide_row = QHBoxLayout()
        auto_hide_info = QVBoxLayout()
        auto_hide_info.setSpacing(1)
        auto_hide_title = label("Auto-hide sidebar")
        auto_hide_title.setStyleSheet("font-weight: 500; font-size: 12px;")
        auto_hide_info.addWidget(auto_hide_title)
        auto_hide_sub = label("Tuck the sidebar into a thin left edge; hover the edge to slide it out.", "muted")
        auto_hide_sub.setWordWrap(True)
        auto_hide_info.addWidget(auto_hide_sub)
        auto_hide_row.addLayout(auto_hide_info, 1)
        self.window.auto_hide_slider = ToggleSlider(
            checked=bool(self.window.settings_manager.get("sidebar", "auto_hide", default=False)))
        self.window.auto_hide_slider.setToolTip("Hide the sidebar until you hover the left edge")
        self.window.auto_hide_slider.setAccessibleName("Toggle sidebar auto-hide")
        self.window.auto_hide_slider.toggled.connect(lambda on: self.window.tabs.set_auto_hide(on))
        auto_hide_row.addWidget(self.window.auto_hide_slider)
        sb_layout.addLayout(auto_hide_row)

        sb_btn_row = QHBoxLayout()
        btn_reset_sidebar = QPushButton("Restore default sidebar")
        btn_reset_sidebar.setToolTip("Show all panels and restore original sidebar order")
        btn_reset_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_sidebar.clicked.connect(self.window.reset_sidebar_layout)
        sb_btn_row.addWidget(btn_reset_sidebar)
        sb_btn_row.addStretch()
        sb_layout.addLayout(sb_btn_row)

        layout.addWidget(sidebar_card)

        # Card 3: 3-D ASCII Sculpture Setting
        sculpture_card = QFrame()
        sculpture_card.setObjectName("card")
        sc_layout = QVBoxLayout(sculpture_card)
        sc_layout.setContentsMargins(16, 14, 16, 14)
        sc_layout.setSpacing(8)

        sc_layout.addWidget(label("3D ASCII Sculpture Animation", "cardTitle"))
        sc_sub = label("Configure the software-rendered rotating ASCII Theta sculpture.", "muted")
        sc_sub.setWordWrap(True)
        sc_layout.addWidget(sc_sub)

        row_sc = QHBoxLayout()
        row_sc.addWidget(label("Enable sculpture animation on overview", "muted"), 1)
        anim_slider = ToggleSlider(checked=self.window.settings_manager.ascii_animation)
        anim_slider.toggled.connect(self._on_ascii_setting_toggled)
        row_sc.addWidget(anim_slider)
        sc_layout.addLayout(row_sc)

        btn_row = QHBoxLayout()
        btn_anim_config = QPushButton("Animation Settings…")
        btn_anim_config.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_anim_config.setToolTip("Configure speed, size, thickness, and 3D projection")
        btn_anim_config.clicked.connect(self._open_animation_settings_dialog)
        btn_row.addWidget(btn_anim_config)
        btn_row.addStretch()
        sc_layout.addLayout(btn_row)

        layout.addWidget(sculpture_card)
        layout.addStretch()
        return container

    def _open_animation_settings_dialog(self):
        target = getattr(self.window, "settings_ascii", None) or self.ascii_splash.ascii_sculpture
        dialog = AnimationSettingsDialog(
            target=target,
            settings_manager=getattr(self.window, "settings_manager", None),
            parent=self,
        )
        dialog.exec()

    def _on_ascii_setting_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("appearance", "ascii_animation", checked)
        if hasattr(self.window, "settings_ascii"):
            self.window.settings_ascii.set_paused(not checked)
        if hasattr(self.window, "anim_toggle_btn") and hasattr(self.window.anim_toggle_btn, "isCheckable") and self.window.anim_toggle_btn.isCheckable():
            self.window.anim_toggle_btn.setChecked(not checked)
            self.window.anim_toggle_btn.setText("Resume animation" if not checked else "Pause animation")

    def _build_core_plugins_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("card")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(18, 16, 18, 16)
        c_layout.setSpacing(12)

        c_layout.addWidget(label("Built-in Core Plugins", "cardTitle"))
        sub = label("Core extensions bundled natively into Theta-IDE.", "muted")
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        # Status badge pill
        status_box = QHBoxLayout()
        engine_badge = label("● Core Engine v1.0  ·  Ready", "badge")
        engine_badge.setStyleSheet("font-weight: 600; padding: 4px 10px;")
        status_box.addWidget(engine_badge)
        status_box.addStretch()
        c_layout.addLayout(status_box)

        desc_lbl = label(
            "Foundational pipeline runners, experiment builder, and telemetry engines\n"
            "operate directly via the Theta-IDE core framework.",
            "muted",
        )
        desc_lbl.setWordWrap(True)
        c_layout.addWidget(desc_lbl)

        # Dynamic core plugins grid populated by window.refresh_plugins_ui()
        self.window.core_plugins_grid = QVBoxLayout()
        self.window.core_plugins_grid.setSpacing(8)
        c_layout.addLayout(self.window.core_plugins_grid)

        if hasattr(self.window, "refresh_plugins_ui"):
            self.window.refresh_plugins_ui()

        layout.addWidget(card)
        layout.addStretch()
        return container

    def _build_community_plugins_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        plugins_card = QFrame()
        plugins_card.setObjectName("card")
        pc_layout = QVBoxLayout(plugins_card)
        pc_layout.setContentsMargins(18, 16, 18, 16)
        pc_layout.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.addWidget(label("Community Plugins & Extensions", "cardTitle"), 1)

        btn_browse_hub = QPushButton("Browse Community Hub…")
        btn_browse_hub.setToolTip("Explore and install community plugins, RL methods, and models")
        btn_browse_hub.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse_hub.clicked.connect(lambda: self.window.open_hub("plugin"))
        header_row.addWidget(btn_browse_hub)
        pc_layout.addLayout(header_row)

        sub = label("Install, configure, or remove community-created plugins.", "muted")
        sub.setWordWrap(True)
        pc_layout.addWidget(sub)

        # Dynamic plugins grid populated by window.refresh_plugins_ui()
        self.window.plugins_grid = QVBoxLayout()
        self.window.plugins_grid.setSpacing(8)
        self.window.plugin_sliders = {}
        pc_layout.addLayout(self.window.plugins_grid)
        self.window.refresh_plugins_ui()

        layout.addWidget(plugins_card)

        # ── Card 2: Theta Hub Preferences ──────────────────────────────────
        hub_card = QFrame()
        hub_card.setObjectName("card")
        hc_layout = QVBoxLayout(hub_card)
        hc_layout.setContentsMargins(18, 16, 18, 16)
        hc_layout.setSpacing(12)

        hc_layout.addWidget(label("Theta Hub Preferences", "cardTitle"))
        hc_sub = label("Configure default behavior when managing community components.", "muted")
        hc_sub.setWordWrap(True)
        hc_layout.addWidget(hc_sub)

        row = QHBoxLayout()
        lbl_uninstall = label("When uninstalling components:")
        row.addWidget(lbl_uninstall, 1)

        combo_uninstall = QComboBox()
        combo_uninstall.addItem("Ask every time", "ask")
        combo_uninstall.addItem("Keep configuration files", "keep")
        combo_uninstall.addItem("Remove configuration files", "remove")

        current_pref = getattr(self.window.settings_manager, "hub_uninstall_configs", "ask")
        idx = combo_uninstall.findData(current_pref)
        if idx >= 0:
            combo_uninstall.setCurrentIndex(idx)

        combo_uninstall.currentIndexChanged.connect(
            lambda i: self.window.settings_manager.set("hub", "uninstall_configs", combo_uninstall.itemData(i))
        )
        row.addWidget(combo_uninstall)
        hc_layout.addLayout(row)

        layout.addWidget(hub_card)
        layout.addStretch()
        return container

    def _build_hotkeys_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        # ── Card 1: Action Key & Terminal Precedence ──────────────────────────
        action_card = QFrame()
        action_card.setObjectName("card")
        ak_layout = QVBoxLayout(action_card)
        ak_layout.setContentsMargins(18, 16, 18, 16)
        ak_layout.setSpacing(12)

        ak_layout.addWidget(label("Action Key & Terminal Precedence", "cardTitle"))
        sub = label("Select the primary leader key used to trigger navigation chords, and manage terminal precedence.", "muted")
        ak_layout.addWidget(sub)

        # 1. Action key selector row
        ak_row = QHBoxLayout()
        ak_lbl = label("Primary Action Key (Leader):", "muted")
        ak_lbl.setFixedWidth(190)
        ak_row.addWidget(ak_lbl)

        self.window.settings_action_key_combo = QComboBox()
        for k, title in ACTION_KEY_PRESETS:
            self.window.settings_action_key_combo.addItem(title, k)
        cur_action_key = self.window.settings_manager.action_key.lower()
        idx = self.window.settings_action_key_combo.findData(cur_action_key)
        if idx >= 0:
            self.window.settings_action_key_combo.setCurrentIndex(idx)
        else:
            custom_title = f"{cur_action_key.replace('_', '+').upper()} (Custom)"
            self.window.settings_action_key_combo.addItem(custom_title, cur_action_key)
            self.window.settings_action_key_combo.setCurrentText(custom_title)
        self.window.settings_action_key_combo.currentIndexChanged.connect(self._on_action_preset_changed)
        self.window.settings_action_key_combo.setMaximumWidth(320)
        ak_row.addWidget(self.window.settings_action_key_combo, 1)
        ak_row.addStretch()
        ak_layout.addLayout(ak_row)

        # 2. Terminal precedence toggle
        row_term = QHBoxLayout()
        term_vbox = QVBoxLayout()
        term_vbox.setSpacing(2)
        term_title = label("Terminal Precedence")
        term_title.setStyleSheet("font-weight: 500;")
        term_vbox.addWidget(term_title)
        term_sub = label("Pass action keys directly to terminal when terminal pane is focused.", "muted")
        term_vbox.addWidget(term_sub)
        row_term.addLayout(term_vbox, 1)

        self.slider_term = ToggleSlider(checked=self.window.settings_manager.hotkeys_terminal_precedence)
        self.slider_term.toggled.connect(self._on_terminal_precedence_toggled)
        row_term.addWidget(self.slider_term)
        ak_layout.addLayout(row_term)

        layout.addWidget(action_card)

        # ── Card 2: Interactive Pane Navigation Hotkey Menu ───────────────────
        pane_card = QFrame()
        pane_card.setObjectName("card")
        pc_layout = QVBoxLayout(pane_card)
        pc_layout.setContentsMargins(18, 16, 18, 16)
        pc_layout.setSpacing(12)

        pc_layout.addWidget(label("Pane Navigation Hotkey Menu", "cardTitle"))
        pc_sub = label(
            "Assign hotkeys (e.g. Action + 1, Action + T) to IDE panels. "
            "Click an input and press a key to record, or Backspace to clear. "
            "Hotkeys persist across panel reordering, sidebar toggling, and plugin states.",
            "muted",
        )
        pc_layout.addWidget(pc_sub)

        # Conflict Warning Banner (hidden by default; shown on shortcut collision)
        self.conflict_banner = QFrame()
        self.conflict_banner.setObjectName("hotkeyConflictBanner")
        self.conflict_banner.setStyleSheet(
            "QFrame#hotkeyConflictBanner {"
            "  background-color: rgba(251, 73, 52, 0.12);"
            "  border: 1.5px solid #fb4934;"
            "  border-radius: 6px;"
            "  padding: 10px 14px;"
            "}"
        )
        banner_layout = QHBoxLayout(self.conflict_banner)
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner_layout.setSpacing(10)

        self.conflict_banner_label = label("", "danger")
        self.conflict_banner_label.setWordWrap(True)
        self.conflict_banner_label.setStyleSheet("color: #ff9999; font-size: 12px; line-height: 1.4;")
        banner_layout.addWidget(self.conflict_banner_label, 1)

        self.conflict_banner.setVisible(False)
        pc_layout.addWidget(self.conflict_banner)

        # Pane Rows List
        self.pane_hotkey_edits = {}
        self.pane_conflict_labels = {}

        current_bindings = self.window.settings_manager.hotkey_bindings
        configured_panes = self.window.settings_manager.hotkey_panes
        all_panes = self._get_all_tracked_panes()

        rows_layout = QVBoxLayout()
        rows_layout.setSpacing(8)

        for pane_info in all_panes:
            pid = pane_info["id"]
            title = pane_info["title"]
            status_text = pane_info["status"]
            status_type = pane_info["status_type"]

            row_frame = QFrame()
            row_frame.setObjectName("hotkeyRowFrame")
            row_frame.setStyleSheet(
                "QFrame#hotkeyRowFrame {"
                "  background: rgba(255, 255, 255, 0.03);"
                "  border: 1px solid rgba(255, 255, 255, 0.07);"
                "  border-radius: 6px;"
                "}"
                "QFrame#hotkeyRowFrame:hover {"
                "  background: rgba(255, 255, 255, 0.05);"
                "  border-color: rgba(255, 255, 255, 0.14);"
                "}"
            )
            rf_layout = QVBoxLayout(row_frame)
            rf_layout.setContentsMargins(12, 10, 12, 10)
            rf_layout.setSpacing(4)

            top_row = QHBoxLayout()
            top_row.setSpacing(10)

            # Left: Title and ID
            title_box = QVBoxLayout()
            title_box.setSpacing(2)
            t_lbl = label(title)
            t_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
            title_box.addWidget(t_lbl)

            sub_lbl = label(f"ID: {pid}", "muted")
            sub_lbl.setStyleSheet("font-size: 11px; font-family: 'Consolas', monospace; opacity: 0.75;")
            title_box.addWidget(sub_lbl)
            top_row.addLayout(title_box, 1)

            # Center: Status pill badge
            status_badge = label(status_text, "badge")
            if status_type == "success":
                status_badge.setStyleSheet("background: rgba(166, 218, 149, 0.15); color: #a6da95; border: 1px solid rgba(166, 218, 149, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            elif status_type == "warning":
                status_badge.setStyleSheet("background: rgba(238, 212, 159, 0.15); color: #eed49f; border: 1px solid rgba(238, 212, 159, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            elif status_type == "accent":
                status_badge.setStyleSheet("background: rgba(138, 173, 244, 0.15); color: #8aadf4; border: 1px solid rgba(138, 173, 244, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            else:
                status_badge.setStyleSheet("background: rgba(255, 255, 255, 0.08); color: #a5adcb; border: 1px solid rgba(255, 255, 255, 0.12); font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px;")
            top_row.addWidget(status_badge)

            # Right: HotkeyComboEdit + Clear button
            init_combo = current_bindings.get(pid) or ""
            if not init_combo and pid in configured_panes:
                idx = configured_panes.index(pid)
                init_combo = f"Action + {idx}"

            combo_edit = HotkeyComboEdit(canonical_combo=init_combo)
            combo_edit.setFixedWidth(160)
            combo_edit.setToolTip("Click to record key (e.g. 1, T), or Backspace to clear")
            combo_edit.comboChanged.connect(lambda c, p=pid: self._on_pane_hotkey_changed(p, c))
            self.pane_hotkey_edits[pid] = combo_edit
            top_row.addWidget(combo_edit)

            btn_clear = QToolButton()
            btn_clear.setText("✕")
            btn_clear.setToolTip(f"Clear shortcut for {title}")
            btn_clear.setFixedSize(28, 28)
            btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_clear.setStyleSheet("QToolButton { border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 4px; color: #a5adcb; } QToolButton:hover { border-color: #ed8796; color: #ed8796; }")
            btn_clear.clicked.connect(lambda _, e=combo_edit: (e.set_canonical_combo(""), e.comboChanged.emit("")))
            top_row.addWidget(btn_clear)

            rf_layout.addLayout(top_row)

            # Per-row conflict label
            conflict_lbl = label("", "danger")
            conflict_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #fb4934; margin-top: 2px;")
            conflict_lbl.setVisible(False)
            self.pane_conflict_labels[pid] = conflict_lbl
            rf_layout.addWidget(conflict_lbl)

            rows_layout.addWidget(row_frame)

        pc_layout.addLayout(rows_layout)

        # Pane reset row
        pane_btn_row = QHBoxLayout()
        btn_reset_panes = QPushButton("Restore default pane shortcuts")
        btn_reset_panes.setToolTip("Reset all panels to default Action + [0–9] mapping")
        btn_reset_panes.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_panes.clicked.connect(self._reset_pane_mappings_to_default)
        pane_btn_row.addWidget(btn_reset_panes)
        pane_btn_row.addStretch()
        pc_layout.addLayout(pane_btn_row)

        layout.addWidget(pane_card)

        # ── Card 3: Global Menu & Application Shortcuts ───────────────────────
        menu_card = QFrame()
        menu_card.setObjectName("card")
        mc_layout = QVBoxLayout(menu_card)
        mc_layout.setContentsMargins(18, 16, 18, 16)
        mc_layout.setSpacing(12)

        mc_layout.addWidget(label("Application Menu Shortcuts", "cardTitle"))
        mc_sub = label(
            "Global shortcuts accessible anywhere in Theta-IDE. Click any shortcut to record a new key combination:",
            "muted",
        )
        mc_layout.addWidget(mc_sub)

        m_grid = QGridLayout()
        m_grid.setHorizontalSpacing(16)
        m_grid.setVerticalSpacing(10)

        self.menu_shortcut_edits = {}
        for i, (aid, title, default_sc) in enumerate(DEFAULT_MENU_SHORTCUTS):
            col_offset = 0 if i < 5 else 2
            row = i if i < 5 else i - 5

            t_lbl = label(title)
            t_lbl.setStyleSheet("font-weight: 500; font-size: 12px;")

            current_sc = self.window.settings_manager.get("shortcuts", aid, default=default_sc) or default_sc
            seq_edit = QKeySequenceEdit(QKeySequence(current_sc))
            seq_edit.setProperty("_action_id", aid)
            seq_edit.setProperty("_default_sc", default_sc)
            seq_edit.keySequenceChanged.connect(lambda seq, a=aid: self._on_menu_shortcut_changed(a, seq))
            self.menu_shortcut_edits[aid] = seq_edit

            m_grid.addWidget(t_lbl, row, col_offset)
            m_grid.addWidget(seq_edit, row, col_offset + 1)

        mc_layout.addLayout(m_grid)

        # Menu reset row
        menu_btn_row = QHBoxLayout()
        btn_reset_menu = QPushButton("Restore default menu shortcuts")
        btn_reset_menu.setToolTip("Reset all application menu shortcuts to their factory defaults")
        btn_reset_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_menu.clicked.connect(self._reset_menu_shortcuts_to_default)
        menu_btn_row.addWidget(btn_reset_menu)
        menu_btn_row.addStretch()
        mc_layout.addLayout(menu_btn_row)

        layout.addWidget(menu_card)
        layout.addStretch()

        # Run initial conflict check
        self._validate_hotkey_conflicts()

        return container

    def _get_pane_title(self, pane_id: str) -> str:
        """Resolve a human-readable title for a pane ID."""
        for pid, title in AVAILABLE_PANES:
            if pid == pane_id:
                return title
        if hasattr(self.window, "tabs") and hasattr(self.window.tabs, "tabs") and pane_id in self.window.tabs.tabs:
            t = self.window.tabs.tabs[pane_id].get("text")
            if t:
                return t
        if hasattr(self.window, "plugin_manager") and hasattr(self.window.plugin_manager, "plugins"):
            for p in self.window.plugin_manager.plugins.values():
                if getattr(p, "id", None) == pane_id or getattr(p, "panel_id", None) == pane_id:
                    return getattr(p, "name", pane_id.replace("_", " ").title())
        return pane_id.replace("_", " ").title()

    def _get_all_tracked_panes(self) -> List[dict]:
        """Aggregate all panels across built-ins, open tabs, sidebar config, plugins, and saved bindings."""
        sm = self.window.settings_manager if hasattr(self.window, "settings_manager") else None
        tracked = []
        seen = set()

        # 1. Canonical built-in panels
        for pid, title in AVAILABLE_PANES:
            if pid == "none" or pid in seen:
                continue
            seen.add(pid)
            tracked.append({
                "id": pid,
                "title": title,
                "is_builtin": True,
            })

        # 2. Open / dynamic tabs
        if hasattr(self.window, "tabs"):
            tab_order = getattr(self.window.tabs, "tab_order", [])
            for pid in tab_order:
                if pid not in seen and pid != "none":
                    seen.add(pid)
                    tracked.append({
                        "id": pid,
                        "title": self._get_pane_title(pid),
                        "is_builtin": False,
                    })

        # 3. Sidebar order panels from settings
        if sm:
            sidebar_order = sm.get("sidebar", "order", default=[])
            if isinstance(sidebar_order, list):
                for pid in sidebar_order:
                    if pid not in seen and pid != "none":
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

        # 4. Plugins (active and inactive)
        if hasattr(self.window, "plugin_manager") and hasattr(self.window.plugin_manager, "plugins"):
            for plugin_id, plugin in self.window.plugin_manager.plugins.items():
                pid = getattr(plugin, "panel_id", None) or plugin_id
                if pid not in seen and pid != "none":
                    seen.add(pid)
                    tracked.append({
                        "id": pid,
                        "title": getattr(plugin, "name", pid.replace("_", " ").title()),
                        "is_builtin": False,
                    })

        # 5. Any previously bound pane IDs from settings.toml
        if sm:
            saved_bindings = sm.get("hotkeys", "bindings", default={})
            if isinstance(saved_bindings, dict):
                for pid in saved_bindings.keys():
                    if pid not in seen and pid != "none":
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

            saved_panes = sm.get("hotkeys", "panes", default=[])
            if isinstance(saved_panes, list):
                for pid in saved_panes:
                    if pid not in seen and pid != "none":
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

        # Decorate each pane with current status (Active, Hidden, Inactive Plugin, etc.)
        for pane in tracked:
            pid = pane["id"]
            if pid == "settings":
                pane["status"] = "Global Panel"
                pane["status_type"] = "accent"
            elif hasattr(self.window, "tabs") and hasattr(self.window.tabs, "has_tab") and self.window.tabs.has_tab(pid):
                pane["status"] = "Active"
                pane["status_type"] = "success"
            elif sm:
                sidebar_visible = sm.get("sidebar", "visible", default=[])
                enabled_plugins = sm.get("plugins", "enabled", default=[])
                if isinstance(sidebar_visible, list) and pid not in sidebar_visible and pid in sm.get("sidebar", "order", default=[]):
                    pane["status"] = "Hidden in sidebar"
                    pane["status_type"] = "muted"
                elif isinstance(enabled_plugins, list) and pid not in enabled_plugins and hasattr(self.window, "plugin_manager"):
                    pane["status"] = "Inactive plugin"
                    pane["status_type"] = "warning"
                else:
                    pane["status"] = "Active"
                    pane["status_type"] = "success"
            else:
                pane["status"] = "Active"
                pane["status_type"] = "success"

        return tracked

    def _validate_hotkey_conflicts(self) -> bool:
        """Scan all pane hotkey assignments for duplicates, update warning banner and per-row alerts.

        Returns True if any conflicts are detected, False otherwise.
        """
        combo_map: Dict[str, List[tuple[str, str]]] = {}
        for pid, edit in self.pane_hotkey_edits.items():
            raw_text = edit.text().strip()
            combo = normalize_hotkey_combo(raw_text)
            if combo:
                title = self._get_pane_title(pid)
                combo_map.setdefault(combo, []).append((pid, title))

        conflicts = {c: p_list for c, p_list in combo_map.items() if len(p_list) > 1}
        has_conflicts = len(conflicts) > 0

        # Update the top conflict warning banner
        if hasattr(self, "conflict_banner") and self.conflict_banner:
            if has_conflicts:
                conflict_details = []
                for combo, p_list in sorted(conflicts.items()):
                    names = " and ".join(t for _, t in p_list)
                    conflict_details.append(f"• <b>{combo}</b> is assigned to <b>{names}</b>")
                self.conflict_banner_label.setText(
                    "<b>Hotkey conflict:</b> Multiple panels share the same shortcut:<br>"
                    + "<br>".join(conflict_details)
                    + "<br><span style='font-size: 11px; opacity: 0.8;'>Conflicting shortcuts will only navigate to the first matching panel.</span>"
                )
                self.conflict_banner.setVisible(True)
            else:
                self.conflict_banner.setVisible(False)

        # Update per-row styling and inline warning labels
        conflicting_pids = {pid for p_list in conflicts.values() for pid, _ in p_list}
        for pid, edit in self.pane_hotkey_edits.items():
            warn_lbl = self.pane_conflict_labels.get(pid)
            if pid in conflicting_pids:
                raw_text = edit.text().strip()
                combo = normalize_hotkey_combo(raw_text)
                other_titles = [t for p, t in combo_map.get(combo, []) if p != pid]
                other_str = ", ".join(other_titles)

                edit.setStyleSheet(
                    "font-family: 'Consolas', monospace; font-size: 12px; font-weight: 700; "
                    "border: 1.5px solid #fb4934; background: rgba(251, 73, 52, 0.15); "
                    "color: #ff9999; border-radius: 4px; padding: 4px 8px;"
                )
                if warn_lbl:
                    warn_lbl.setText(f"Conflict: also assigned to {other_str}")
                    warn_lbl.setVisible(True)
            else:
                edit.setStyleSheet(
                    "font-family: 'Consolas', monospace; font-size: 12px; font-weight: 600; "
                    "border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 4px; padding: 4px 8px;"
                )
                if warn_lbl:
                    warn_lbl.setText("")
                    warn_lbl.setVisible(False)

        return has_conflicts

    def _on_action_preset_changed(self, index: int):
        if not hasattr(self.window, "settings_action_key_combo"):
            return
        val = self.window.settings_action_key_combo.itemData(index)
        if val and hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("hotkeys", "action_key", val)
            if hasattr(self.window, "statusBar") and self.window.statusBar():
                self.window.statusBar().showMessage(f"Action Key set to {val.replace('_', '+').upper()}", 3000)

    def _on_terminal_precedence_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("hotkeys", "terminal_precedence", checked)

    def _on_pane_hotkey_changed(self, pane_id: str, combo_str: str):
        normalized = normalize_hotkey_combo(combo_str)
        sm = getattr(self.window, "settings_manager", None)
        if sm:
            bindings = dict(sm.hotkey_bindings)
            if normalized:
                bindings[pane_id] = normalized
            elif pane_id in bindings:
                del bindings[pane_id]
            sm.set("hotkeys", "bindings", bindings)

            # Update legacy hotkey_panes for digits 0-9
            panes = list(sm.hotkey_panes)
            while len(panes) < 10:
                panes.append("none")

            for d in range(10):
                target_combo = f"Action + {d}".upper()
                assigned_pid = next((p for p, c in bindings.items() if c.upper() == target_combo), None)
                if assigned_pid:
                    panes[d] = assigned_pid
                elif panes[d] == pane_id and not normalized:
                    panes[d] = "none"
            sm.set("hotkeys", "panes", panes)

        # Validate conflicts across all inputs
        self._validate_hotkey_conflicts()

        if hasattr(self.window, "statusBar") and self.window.statusBar():
            title = self._get_pane_title(pane_id)
            if normalized:
                self.window.statusBar().showMessage(f"Shortcut for {title} set to {normalized}", 3000)
            else:
                self.window.statusBar().showMessage(f"Shortcut for {title} cleared", 3000)

    def _reset_pane_mappings_to_default(self):
        sm = getattr(self.window, "settings_manager", None)
        default_bindings = {DEFAULT_PANES[i]: f"Action + {i}" for i in range(len(DEFAULT_PANES))}
        if sm:
            sm.set("hotkeys", "bindings", default_bindings)
            sm.set("hotkeys", "panes", list(DEFAULT_PANES))

        for pid, edit in self.pane_hotkey_edits.items():
            expected = default_bindings.get(pid, "")
            edit.set_canonical_combo(expected)

        self._validate_hotkey_conflicts()

        if hasattr(self.window, "statusBar") and self.window.statusBar():
            self.window.statusBar().showMessage("Restored default pane navigation shortcuts.", 3000)

    def _on_menu_shortcut_changed(self, action_id: str, sequence: QKeySequence):
        seq_str = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("shortcuts", action_id, seq_str)
        if hasattr(self.window, "menu_actions") and action_id in self.window.menu_actions:
            self.window.menu_actions[action_id].setShortcut(sequence)
        if hasattr(self.window, "statusBar") and self.window.statusBar():
            self.window.statusBar().showMessage(f"Shortcut for {action_id} set to {seq_str or 'None'}", 3000)

    def _reset_menu_shortcuts_to_default(self):
        for aid, _, default_sc in DEFAULT_MENU_SHORTCUTS:
            if aid in self.menu_shortcut_edits:
                seq = QKeySequence(default_sc)
                self.menu_shortcut_edits[aid].blockSignals(True)
                self.menu_shortcut_edits[aid].setKeySequence(seq)
                self.menu_shortcut_edits[aid].blockSignals(False)
            if hasattr(self.window, "settings_manager"):
                self.window.settings_manager.set("shortcuts", aid, default_sc)
            if hasattr(self.window, "menu_actions") and aid in self.window.menu_actions:
                self.window.menu_actions[aid].setShortcut(QKeySequence(default_sc))
        if hasattr(self.window, "statusBar") and self.window.statusBar():
            self.window.statusBar().showMessage("Restored default application menu shortcuts.", 3000)

    def sync_from_settings(self):
        """Keep UI in sync with settings_manager without triggering feedback loops."""
        sm = getattr(self.window, "settings_manager", None)
        if not sm:
            return

        # 1. Terminal precedence
        if hasattr(self, "slider_term") and self.slider_term:
            self.slider_term.blockSignals(True)
            self.slider_term.setChecked(sm.hotkeys_terminal_precedence)
            self.slider_term.blockSignals(False)

        # 2. Action key combo
        if hasattr(self.window, "settings_action_key_combo") and self.window.settings_action_key_combo:
            combo = self.window.settings_action_key_combo
            cur_key = sm.action_key.lower()
            idx = combo.findData(cur_key)
            combo.blockSignals(True)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                custom_title = f"{cur_key.replace('_', '+').upper()} (Custom)"
                combo.addItem(custom_title, cur_key)
                combo.setCurrentIndex(combo.count() - 1)
            combo.blockSignals(False)

        # 3. Pane navigation hotkey edits
        if hasattr(self, "pane_hotkey_edits") and self.pane_hotkey_edits:
            bindings = sm.hotkey_bindings
            for pid, edit in self.pane_hotkey_edits.items():
                expected = bindings.get(pid, "")
                if edit.get_canonical_combo() != expected:
                    edit.set_canonical_combo(expected)
            self._validate_hotkey_conflicts()

        # 4. Menu shortcut edits
        if hasattr(self, "menu_shortcut_edits") and self.menu_shortcut_edits:
            for aid, edit in self.menu_shortcut_edits.items():
                default_sc = edit.property("_default_sc") or ""
                sc = sm.get("shortcuts", aid, default=default_sc) or default_sc
                seq = QKeySequence(sc)
                if edit.keySequence() != seq:
                    edit.blockSignals(True)
                    edit.setKeySequence(seq)
                    edit.blockSignals(False)

    def _build_backend_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        backend_card = QFrame()
        backend_card.setObjectName("card")
        bc_layout = QVBoxLayout(backend_card)
        bc_layout.setContentsMargins(18, 16, 18, 16)
        bc_layout.setSpacing(12)

        bc_layout.addWidget(label("FastAPI Backend Service", "cardTitle"))
        bc_sub = label("The backend daemon manages training runs, sweeps, and job orchestration.", "muted")
        bc_layout.addWidget(bc_sub)

        # Status row
        status_box = QHBoxLayout()
        status_box.setSpacing(10)
        self.window.settings_backend_status = label("Status: Checking connection…", "muted")
        self.window.settings_backend_status.setStyleSheet("font-size: 12px; font-weight: 600;")
        status_box.addWidget(self.window.settings_backend_status, 1)

        btn_test = QPushButton("Test connection")
        btn_test.setToolTip("Ping the FastAPI daemon health check endpoint")
        btn_test.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_test.clicked.connect(self.window.test_backend_connection)
        status_box.addWidget(btn_test)

        btn_docs = QPushButton("Swagger docs")
        btn_docs.setToolTip("Open interactive API documentation in your web browser")
        btn_docs.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_docs.clicked.connect(self.window.open_swagger_docs)
        status_box.addWidget(btn_docs)

        bc_layout.addLayout(status_box)

        # URL row
        url_row = QHBoxLayout()
        url_row.setSpacing(10)
        url_lbl = label("API URL:", "muted")
        url_lbl.setStyleSheet("font-weight: 500;")
        url_row.addWidget(url_lbl)

        self.window.settings_backend_url = QLineEdit(self.window.backend.base_url)
        self.window.settings_backend_url.setReadOnly(True)
        self.window.settings_backend_url.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self.window.settings_backend_url.setMaximumWidth(320)
        url_row.addWidget(self.window.settings_backend_url, 1)
        url_row.addStretch()
        bc_layout.addLayout(url_row)

        # CLI tip
        tip_lbl = label(
            "Daemon command:  uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000",
            "muted",
        )
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-style: italic;")
        bc_layout.addWidget(tip_lbl)

        layout.addWidget(backend_card)
        layout.addStretch()
        return container

    def _build_storage_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        storage_card = QFrame()
        storage_card.setObjectName("card")
        sc_layout = QVBoxLayout(storage_card)
        sc_layout.setContentsMargins(18, 16, 18, 16)
        sc_layout.setSpacing(12)

        sc_layout.addWidget(label("Workspace & Storage", "cardTitle"))
        sc_sub = label("Manage local experiment databases, filesystems, and layout preferences.", "muted")
        sc_sub.setWordWrap(True)
        sc_layout.addWidget(sc_sub)

        # Storage info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        root_path = getattr(self.window.store, "root", "Unknown")
        row_root = QHBoxLayout()
        lbl_root = label("Data root:", "muted")
        lbl_root.setFixedWidth(90)
        row_root.addWidget(lbl_root)
        val_root = label(str(root_path))
        val_root.setWordWrap(True)
        val_root.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        val_root.setToolTip(str(root_path))
        row_root.addWidget(val_root, 1)
        info_layout.addLayout(row_root)

        row_runs = QHBoxLayout()
        lbl_runs = label("Database:", "muted")
        lbl_runs.setFixedWidth(90)
        row_runs.addWidget(lbl_runs)
        self.window.settings_runs_count_label = label(
            f"Total run records:  {len(getattr(self.window, 'runs', []))} runs", "muted"
        )
        self.window.settings_runs_count_label.setStyleSheet("font-weight: 500; font-size: 12px;")
        row_runs.addWidget(self.window.settings_runs_count_label, 1)
        info_layout.addLayout(row_runs)

        sc_layout.addLayout(info_layout)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_open_settings = QPushButton("Open settings.toml")
        btn_open_settings.setToolTip(
            f"Open settings file in your default editor\n{self.window.settings_manager.workspace_settings_path}"
        )
        btn_open_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_settings.clicked.connect(self.window.open_settings_file)
        btn_row.addWidget(btn_open_settings)

        btn_reset_layout = QPushButton("Reset UI layout")
        btn_reset_layout.setToolTip("Restore default pane sizes and layout")
        btn_reset_layout.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_layout.clicked.connect(
            lambda: (
                self.window.restoreState(self.window.default_layout),
                self.window.statusBar().showMessage("Restored default UI layout.", 4000),
            )
        )
        btn_row.addWidget(btn_reset_layout)
        btn_row.addStretch()

        sc_layout.addLayout(btn_row)

        hint = label("Changes to settings.toml are automatically hot-reloaded across the IDE.", "muted")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; font-style: italic;")
        sc_layout.addWidget(hint)

        layout.addWidget(storage_card)
        layout.addStretch()
        return container

    # ── Interaction & View Control ───────────────────────────────────────────

    def showEvent(self, event):
        """When entering settings: display the ASCII Theta until user clicks a tab."""
        super().showEvent(event)
        if self.active_tab_id is None:
            self.hide_overlay()

    def on_tab_button_clicked(self, tab_id: str):
        """Clicking a tab displays that tab's settings."""
        if self.active_tab_id == tab_id and self.content_stack.currentIndex() == 1:
            # Clicking currently active tab toggles back to ASCII Theta
            self.hide_overlay()
            return

        self.show_tab(tab_id)

    def show_tab(self, tab_id: str):
        """Activate the given tab and show its settings."""
        if tab_id not in self.page_widgets:
            return

        self.active_tab_id = tab_id

        # Update button checked states
        for tid, btn in self.tab_buttons.items():
            btn.setChecked(tid == tab_id)

        # Find tab spec for titles & icons
        spec = next((s for s in TAB_SPECS if s["id"] == tab_id), None)
        title = spec["title"] if spec else tab_id.title()
        subtitle = spec["subtitle"] if spec else ""
        icon_name = spec["icon"] if spec else "settings"

        # Update settings header and resize card for active tab
        self.settings_window.set_header(title, subtitle, icon_name)
        self.settings_window.apply_tab_size(tab_id)

        # Switch stacked widget to target page
        target_page = self.page_widgets[tab_id]
        self.settings_window.pages_stack.setCurrentWidget(target_page)

        # Switch main content stack to settings window (Page 1) so it is overlayed on top
        self.settings_window.setVisible(True)
        self.content_stack.setCurrentIndex(1)
        self.settings_window.raise_()

    def hide_overlay(self):
        """Hide settings and return to full ASCII Theta view."""
        self.active_tab_id = None
        for btn in self.tab_buttons.values():
            btn.setChecked(False)
        self.settings_window.setVisible(False)
        self.content_stack.setCurrentIndex(0)
        self.ascii_splash.raise_()

    def reset_to_ascii(self):
        """Explicitly reset settings view to the clean ASCII Theta."""
        self.hide_overlay()

    def refresh_styles(self):
        """Theme refresh: update custom styles, icons, and colors."""
        panel_col = theme_color("panel")
        surface_col = theme_color("surface")
        border_col = theme_color("border")
        accent_col = theme_color("accent")
        text_col = theme_color("text")
        muted_col = theme_color("muted")

        # Sidebar styling
        self.sidebar_frame.setStyleSheet(f"""
            QFrame#settingsSidebar {{
                background-color: {panel_col};
                border-right: 1px solid {border_col};
            }}
            QToolButton#settingsSideTab {{
                text-align: left;
                padding-left: 12px;
                border: 1px solid transparent;
                border-radius: 6px;
                background-color: transparent;
                color: {muted_col};
                font-size: 12px;
                font-weight: 500;
            }}
            QToolButton#settingsSideTab:hover {{
                background-color: {surface_col};
                color: {text_col};
            }}
            QToolButton#settingsSideTab:checked {{
                background-color: {surface_col};
                border: 1px solid {accent_col};
                color: {accent_col};
                font-weight: 600;
            }}
        """)

        # Settings detail window styling
        self.settings_window.setStyleSheet(f"""
            QFrame#settingsOverlayLayer {{
                background-color: transparent;
                border: none;
            }}
            QFrame#settingsDetailCard {{
                background-color: {panel_col};
                border-top: none;
                border-left: none;
                border-right: 1px solid {border_col};
                border-bottom: 1px solid {border_col};
                border-bottom-right-radius: 12px;
            }}
            QFrame#card {{
                background-color: {surface_col};
                border: 1px solid {border_col};
                border-radius: 8px;
            }}
            QFrame#placeholderCard {{
                background-color: {surface_col};
                border: 1px dashed {border_col};
                border-radius: 8px;
            }}
            QFrame#settingsSeparator {{
                color: {border_col};
                background-color: {border_col};
                max-height: 1px;
            }}
            QToolButton#settingsCloseButton {{
                background-color: {surface_col};
                border: 1px solid {border_col};
                border-radius: 6px;
                padding: 4px 12px;
                color: {text_col};
                font-size: 12px;
                font-weight: 600;
            }}
            QToolButton#settingsCloseButton:hover {{
                background-color: rgba(251, 73, 52, 0.2);
                border-color: #fb4934;
                color: #fb4934;
            }}
            QScrollArea#settingsScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea#settingsScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                height: 0px;
                border: none;
            }}
            QKeySequenceEdit {{
                background-color: {surface_col};
                border: 1px solid {border_col};
                border-radius: 4px;
                padding: 4px 8px;
                color: {text_col};
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: 600;
            }}
            QKeySequenceEdit:focus {{
                border: 1px solid {accent_col};
            }}
        """)

        if hasattr(self.settings_window, "shadow") and self.settings_window.shadow:
            self.settings_window.shadow.setColor(QColor(0, 0, 0, 160))

        # Refresh icons on buttons
        for btn in self.tab_buttons.values():
            btn.refresh_icon()

        if self.active_tab_id:
            spec = next((s for s in TAB_SPECS if s["id"] == self.active_tab_id), None)
            if spec:
                self.settings_window.set_header(spec["title"], spec["subtitle"], spec["icon"])

        # Update ASCII sculpture
        if hasattr(self.ascii_splash, "ascii_sculpture"):
            self.ascii_splash.ascii_sculpture.update()
