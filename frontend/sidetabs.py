"""Vertical tab bar with themed SVG icons: a drop-in for the subset of QTabWidget the window uses.

Tabs can be grouped into named bins: drag one tab onto the middle of another (or onto a bin's header) to
group them. A bin's header expands or collapses it on click and renames it on double-click. Dragging a tab
by the top or bottom edge of another item reorders instead, and dropping it outside a bin takes it out.

The bar's structure is `entries`, the top-level items in order: tab ids and bin ids. Each bin keeps its own
`members`. `tab_order`, the flat order of every tab that the rest of the app reads, is derived from it.
"""
import functools
from pathlib import Path

from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLineEdit, QMenu, QSizePolicy, QStackedWidget,
                             QToolButton, QVBoxLayout, QWidget)

from .theme import theme_color

ICON_DIR = Path(__file__).resolve().parent / "icons"
ICON_SIZE = 20
AUTO_HIDE_EDGE = 6  # px strip left visible while the bar is tucked away
AUTO_HIDE_DELAY_MS = 350
TAB_MIME = "application/x-thetatab-id"
BIN_MIME = "application/x-thetabin-id"
MERGE_ZONE = 0.28  # dropping in the top or bottom 28% of an item reorders; the middle groups
BIN_NAME_LIMIT = 40


def svg_icon(name, colors, size=ICON_SIZE):
    """Render icons/<name>.svg once per state, replacing currentColor with that state's theme role color.

    colors maps (QIcon.Mode, QIcon.State) to a theme role such as "muted" or "accent".
    """
    # Bundled icons win. A bare name like "plots" must not pick up a same-named file in the
    # working directory (the repo root has a "plots" symlink, a plain text file on Windows).
    p = Path(name)
    if (ICON_DIR / f"{name}.svg").is_file():
        source_path = ICON_DIR / f"{name}.svg"
    elif (ICON_DIR / name).is_file():
        source_path = ICON_DIR / name
    elif (p.is_absolute() or p.suffix.lower() == ".svg") and p.is_file():
        source_path = p  # explicit path, e.g. a plugin's own icon
    else:
        return QIcon()
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    # Resolve roles to colors here so the cache key changes with the theme.
    resolved = tuple((mode, state, theme_color(role)) for (mode, state), role in colors.items())
    # A copy: QIcon is implicitly shared, so a caller that modifies its icon detaches from the cache.
    return QIcon(_render_svg_icon(source_path.read_bytes(), resolved, size, ratio))


@functools.lru_cache(maxsize=1024)
def _render_svg_icon(source: bytes, colors: tuple, size: int, ratio: float) -> QIcon:
    """Render one icon. Cached: the UI asks for the same ~40 icons ~160 times while starting up,
    and again on every theme switch. Keyed on the file's bytes, so an edited icon is re-rendered."""
    text = source.decode("utf-8")
    icon = QIcon()
    for mode, state, color in colors:
        renderer = QSvgRenderer(QByteArray(text.replace("currentColor", color).encode("utf-8")))
        pixmap = QPixmap(round(size * ratio), round(size * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        icon.addPixmap(pixmap, mode, state)
    return icon


TAB_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "muted",
    (QIcon.Mode.Active, QIcon.State.Off): "text",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
}


class DraggableTabButton(QToolButton):
    """Sidebar button that can be clicked to activate or dragged to reorder (or group, for tabs)."""

    mime_type = TAB_MIME

    def __init__(self, tab_id, side_tabs, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.side_tabs = side_tabs
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
        self.setDown(False)  # the drag swallows the release, which would otherwise leave the button pressed
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(self.mime_type, QByteArray(self.tab_id.encode("utf-8")))
        drag.setMimeData(mime)

        pixmap = self.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.7)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        bin_id = self.side_tabs.bin_of(self.tab_id)
        if bin_id:
            menu.addAction(f"Remove from {self.side_tabs.bins[bin_id]['name']}",
                           lambda: self.side_tabs.remove_from_bin(self.tab_id))
        menu.addAction("Add to new bin", lambda: self.side_tabs.new_bin([self.tab_id], rename=True))
        menu.exec(event.globalPos())


class BinHeader(DraggableTabButton):
    """A bin's header: click to expand or collapse, double-click (or right-click) to rename, drag to move."""

    mime_type = BIN_MIME

    def __init__(self, bin_id, side_tabs, parent=None):
        super().__init__(bin_id, side_tabs, parent)
        self.setObjectName("sideTabBin")
        self.setCheckable(True)  # checked while it hides the open panel (collapsed with an active member)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIconSize(QSize(10, 10))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.clicked.connect(self._on_clicked)
        self.editor = None

    @property
    def bin_id(self):
        return self.tab_id

    def _on_clicked(self):
        self.side_tabs.toggle_bin(self.bin_id)

    def mouseDoubleClickEvent(self, event):
        # The two clicks of a double-click toggled the bin twice, so it ends up as it was.
        self.start_rename()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_text()

    def refresh_text(self):
        entry = self.side_tabs.bins.get(self.bin_id)
        if not entry:
            return
        room = max(12, self.width() - self.iconSize().width() - 12)
        self.setText(self.fontMetrics().elidedText(entry["name"], Qt.TextElideMode.ElideRight, room))
        count = len(entry["members"])
        state = "click to collapse" if entry["expanded"] else "click to expand"
        self.setToolTip(f"{entry['name']} ({count} panel{'s' if count != 1 else ''}): {state}, "
                        f"double-click to rename")
        self.setAccessibleName(f"{entry['name']} bin")

    def contextMenuEvent(self, event):
        entry = self.side_tabs.bins[self.bin_id]
        menu = QMenu(self)
        menu.addAction("Rename", self.start_rename)
        menu.addAction("Collapse" if entry["expanded"] else "Expand", lambda: self.side_tabs.toggle_bin(self.bin_id))
        menu.addSeparator()
        menu.addAction("Ungroup", lambda: self.side_tabs.ungroup(self.bin_id))
        menu.exec(event.globalPos())

    # ── inline rename ────────────────────────────────────────────────────────

    def start_rename(self):
        if self.editor is not None or self.bin_id not in self.side_tabs.bins:
            return
        bar = self.parentWidget()
        editor = _BinNameEditor(self, bar)
        editor.setText(self.side_tabs.bins[self.bin_id]["name"])
        editor.setGeometry(self.geometry().adjusted(0, -1, 0, 1))
        editor.show()
        editor.selectAll()
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        self.editor = editor

    def finish_rename(self, commit):
        editor, self.editor = self.editor, None
        if editor is None:
            return
        if commit:
            self.side_tabs.rename_bin(self.bin_id, editor.text())
        editor.hide()
        editor.deleteLater()


class _BinNameEditor(QLineEdit):
    """Inline editor over a bin header: Enter or focus loss keeps the name, Escape cancels."""

    def __init__(self, header, parent):
        super().__init__(parent)
        self.header = header
        self.setObjectName("sideTabBinEditor")
        self.setMaxLength(BIN_NAME_LIMIT)
        self._done = False
        self.returnPressed.connect(lambda: self._finish(True))

    def _finish(self, commit):
        if not self._done:
            self._done = True
            self.header.finish_rename(commit)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._finish(True)


class SideTabBar(QWidget):
    """Container widget that accepts tab and bin drops, draws bins' trays and the drop indicator."""

    def __init__(self, side_tabs, parent=None):
        super().__init__(parent)
        self.side_tabs = side_tabs
        self.setObjectName("sideTabs")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setAcceptDrops(True)
        self._drop_slot = None
        self._merge_target = None  # ("tab" | "bin", id) highlighted as the group target

    @staticmethod
    def _dragged(event):
        mime = event.mimeData()
        for kind, fmt in (("tab", TAB_MIME), ("bin", BIN_MIME)):
            if mime.hasFormat(fmt):
                return kind, bytes(mime.data(fmt)).decode("utf-8")
        return None

    def dragEnterEvent(self, event):
        if self._dragged(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        dragged = self._dragged(event)
        if not dragged:
            event.ignore()
            return
        target = self._drop_target(event.position().y(), dragged)
        slot, merge = (target[1], None) if target[0] == "slot" else (None, target[1:])
        if slot != self._drop_slot or merge != self._merge_target:
            self._drop_slot, self._merge_target = slot, merge
            self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_slot = self._merge_target = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        dragged = self._dragged(event)
        if not dragged:
            event.ignore()
            return
        slot, merge = self._drop_slot, self._merge_target
        self._drop_slot = self._merge_target = None
        self.update()
        kind, item_id = dragged
        if merge is not None and kind == "tab":
            target_kind, target_id = merge
            if target_kind == "bin":
                self.side_tabs.add_to_bin(target_id, item_id)
            else:
                self.side_tabs.group_tabs(target_id, item_id, rename=True)
        elif slot is not None:
            if kind == "tab":
                self.side_tabs.move_tab_to_visible_slot(item_id, slot)
            else:
                self.side_tabs.move_bin_to_visible_slot(item_id, slot)
        event.acceptProposedAction()

    def _drop_target(self, y, dragged):
        """("merge", kind, id) when over the middle of a groupable item, else ("slot", visible index)."""
        items = self.side_tabs.visible_items()
        kind, item_id = dragged
        if kind == "tab":
            for item_kind, other_id, widget in items:
                if item_kind == "tab" and other_id == item_id:
                    continue
                rect = widget.geometry()
                if rect.top() <= y <= rect.bottom() and rect.height() > 0:
                    fraction = (y - rect.top()) / rect.height()
                    if MERGE_ZONE < fraction < 1 - MERGE_ZONE:
                        own_bin = self.side_tabs.bin_of(item_id)
                        if own_bin and (other_id == own_bin or self.side_tabs.bin_of(other_id) == own_bin):
                            break  # already in this bin: reorder instead
                        return ("merge", item_kind, other_id)
                    break
        for i, (_kind, _id, widget) in enumerate(items):
            if y < widget.geometry().center().y():
                return ("slot", i)
        return ("slot", len(items))

    def _calculate_drop_slot(self, y):
        return self._drop_target(y, ("bin", ""))[1]

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Bins' trays behind their header and members
        for bin_id in self.side_tabs.entries:
            rect = self.side_tabs.bin_rect(bin_id)
            if rect is None:
                continue
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), 7, 7)
            painter.fillPath(path, QColor(theme_color("base")))
            painter.setPen(QPen(QColor(theme_color("border")), 1))
            painter.drawPath(path)

        accent = QColor(theme_color("accent"))
        if self._merge_target is not None:
            widget = self.side_tabs.item_widget(*self._merge_target)
            if widget is not None and widget.isVisible():
                painter.setPen(QPen(accent, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(QRectF(widget.geometry()).adjusted(1, 1, -1, -1), 6, 6)
        elif self._drop_slot is not None:
            widgets = [widget for _kind, _id, widget in self.side_tabs.visible_items()]
            if widgets:
                slot = self._drop_slot
                if slot == 0:
                    y = float(widgets[0].geometry().top() - 1)
                elif slot >= len(widgets):
                    y = float(widgets[-1].geometry().bottom() + 2)
                else:
                    y = (widgets[slot - 1].geometry().bottom() + widgets[slot].geometry().top()) / 2.0
                painter.setPen(QPen(accent, 2))
                painter.drawLine(6, round(y), self.width() - 6, round(y))
                painter.setBrush(accent)
                painter.drawEllipse(QRectF(3, y - 2.5, 5, 5))
                painter.drawEllipse(QRectF(self.width() - 8, y - 2.5, 5, 5))
        painter.end()


class SideTabs(QWidget):
    """Icon-over-label buttons stacked on the left, with the pages in a QStackedWidget beside them."""
    currentChanged = pyqtSignal(int)
    logoClicked = pyqtSignal()
    tabOrderChanged = pyqtSignal(list)  # also emitted when bins change, so the layout is saved
    tabVisibilityChanged = pyqtSignal(str, bool)
    tabsChanged = pyqtSignal()  # a tab was added or removed (plugins add theirs after the window is built)
    autoHideChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.bar = SideTabBar(self)
        self.bar_layout = QVBoxLayout(self.bar)
        # Inset the buttons so the rounded selection sits inside the bar, clear of its 1 px right border.
        self.bar_layout.setContentsMargins(5, 6, 6, 6)
        self.bar_layout.setSpacing(3)
        self.bar_layout.addStretch()

        self.logo_button = QToolButton()
        self.logo_button.setObjectName("sideTabLogo")
        self.logo_button.setCheckable(True)
        self.logo_button.setAutoRaise(True)
        self.logo_button.setToolTip("Settings & About")
        self.logo_button.setAccessibleName("ThetaIDE Settings & About")
        self.logo_button.setIconSize(QSize(36, 36))
        self.logo_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.logo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logo_button.clicked.connect(self._on_logo_clicked)
        self.set_logo_icon()
        self.bar_layout.addWidget(self.logo_button)

        # Auto-hide: the bar leaves the layout and floats over the pages; this strip reveals it on hover
        self.auto_hide = False
        self.edge = QWidget(self)
        self.edge.setObjectName("sideTabsEdge")
        self.edge.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.edge.setFixedWidth(AUTO_HIDE_EDGE)
        self.edge.setToolTip("Hover to show the sidebar")
        self.edge.hide()
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(AUTO_HIDE_DELAY_MS)
        self._hide_timer.timeout.connect(self._tuck_bar)
        self.edge.installEventFilter(self)
        self.bar.installEventFilter(self)

        self.stack = QStackedWidget()
        layout.addWidget(self.edge)
        layout.addWidget(self.bar)
        layout.addWidget(self.stack, 1)

        self.tabs = {}
        self.tab_order = []
        self.entries = []  # top level, in order: tab ids and bin ids
        self.bins = {}  # bin id -> {"name", "members", "expanded", "header"}
        self._bin_serial = 0
        self.settings_widget = None
        self.settings_index = -1

    @property
    def buttons(self):
        return [self.tabs[tid]["button"] for tid in self.tab_order]

    @property
    def icon_names(self):
        return [self.tabs[tid]["icon"] for tid in self.tab_order]

    def set_logo_icon(self):
        """Corner logo: the bracket-theta mark, tinted with the current theme's accent."""
        source_path = ICON_DIR / "theta_bracket_mark.svg"
        if not source_path.exists():
            return
        ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
        source = source_path.read_text(encoding="utf-8").replace("currentColor", theme_color("accent"))
        renderer = QSvgRenderer(QByteArray(source.encode("utf-8")))
        pixmap = QPixmap(round(36 * ratio), round(36 * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        self.logo_button.setIcon(QIcon(pixmap))

    def addTab(self, widget, text, icon=None, short=None, tab_id=None):
        """icon names an SVG in frontend/icons/; short is the label shown under it (text becomes the tooltip)."""
        if tab_id is None:
            tab_id = icon or (short or text).lower().replace(" ", "_")
        index = self.stack.addWidget(widget)
        button = DraggableTabButton(tab_id, self)
        button.setObjectName("sideTab")
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setText(short or text)
        button.setToolTip(text)
        button.setAccessibleName(text)
        button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda: self.setCurrentWidget(widget))

        self.tabs[tab_id] = {
            "widget": widget,
            "text": text,
            "short": short or text,
            "icon": icon,
            "button": button,
            "visible": True,
        }
        self.entries.append(tab_id)
        if icon:
            button.setIcon(svg_icon(icon, TAB_ICON_COLORS))
        self._rebuild(emit=False)
        if len(self.tab_order) == 1:
            button.setChecked(True)
        self.tabsChanged.emit()
        return index

    def removeTab(self, tab_id):
        """Remove a tab by tab_id and detach its button and widget."""
        if tab_id not in self.tabs:
            return
        entry = self.tabs[tab_id]

        # If active widget is this one, switch to another visible tab
        if self.currentWidget() is entry["widget"]:
            switched = False
            for tid in self.tab_order:
                if tid != tab_id and self.tabs[tid]["visible"]:
                    self.setCurrentWidget(self.tabs[tid]["widget"])
                    switched = True
                    break
            if not switched and self.settings_widget:
                self.setCurrentWidget(self.settings_widget)

        self._detach(tab_id)
        self.tabs.pop(tab_id)
        self.bar_layout.removeWidget(entry["button"])
        entry["button"].deleteLater()
        self.stack.removeWidget(entry["widget"])
        self._rebuild()
        self.tabsChanged.emit()

    # ── structure ────────────────────────────────────────────────────────────

    def bin_of(self, tab_id):
        """The bin a tab is in, or None."""
        return next((bin_id for bin_id, b in self.bins.items() if tab_id in b["members"]), None)

    def _top_level_of(self, tab_id):
        return self.bin_of(tab_id) or tab_id

    def _flatten(self):
        order = []
        for entry in self.entries:
            order.extend(self.bins[entry]["members"] if entry in self.bins else [entry])
        return order

    def _display(self):
        """Every widget in display order: (kind, id, widget, shown)."""
        out = []
        for entry in self.entries:
            if entry in self.bins:
                b = self.bins[entry]
                any_visible = any(self.tabs[t]["visible"] for t in b["members"])
                out.append(("bin", entry, b["header"], any_visible))
                for t in b["members"]:
                    out.append(("tab", t, self.tabs[t]["button"], b["expanded"] and self.tabs[t]["visible"]))
            else:
                out.append(("tab", entry, self.tabs[entry]["button"], self.tabs[entry]["visible"]))
        return out

    def visible_items(self):
        """(kind, id, widget) for each item shown in the bar, top to bottom."""
        return [(kind, item_id, widget) for kind, item_id, widget, shown in self._display() if shown]

    def item_widget(self, kind, item_id):
        if kind == "bin":
            return self.bins.get(item_id, {}).get("header")
        return self.tabs.get(item_id, {}).get("button")

    def bin_rect(self, bin_id):
        """The tray behind a shown bin: its header and expanded members, in bar coordinates."""
        b = self.bins.get(bin_id)
        if not b or not b["header"].isVisible():
            return None
        rect = b["header"].geometry()
        if b["expanded"]:
            for t in b["members"]:
                button = self.tabs[t]["button"]
                if button.isVisible():
                    rect = rect.united(button.geometry())
        return rect.adjusted(-3, -1, 3, 1)

    def _rebuild(self, emit=True):
        """Lay the widgets out from `entries`, apply visibility, refresh bins' headers; save if `emit`."""
        before = list(self.tab_order)
        self.tab_order = self._flatten()
        for i, (_kind, _id, widget, shown) in enumerate(self._display()):
            self.bar_layout.removeWidget(widget)
            self.bar_layout.insertWidget(i, widget)
            widget.setVisible(shown)
        for bin_id, b in self.bins.items():
            b["header"].setIcon(svg_icon("bin_expanded" if b["expanded"] else "bin_collapsed",
                                         TAB_ICON_COLORS, size=10))
            b["header"].refresh_text()
        self._sync_bin_highlight()
        self.bar.update()
        if emit:
            self.tabOrderChanged.emit(list(self.tab_order))
        return before != self.tab_order

    def _detach(self, tab_id):
        """Take a tab out of wherever it sits; a bin left empty goes away."""
        if tab_id in self.entries:
            self.entries.remove(tab_id)
            return
        bin_id = self.bin_of(tab_id)
        if bin_id:
            self.bins[bin_id]["members"].remove(tab_id)
            if not self.bins[bin_id]["members"]:
                self._delete_bin(bin_id)

    def _delete_bin(self, bin_id):
        b = self.bins.pop(bin_id)
        if bin_id in self.entries:
            self.entries.remove(bin_id)
        self.bar_layout.removeWidget(b["header"])
        b["header"].hide()
        b["header"].deleteLater()

    def _make_bin(self, name, members, expanded=True):
        self._bin_serial += 1
        bin_id = f"bin-{self._bin_serial}"
        while bin_id in self.bins:
            self._bin_serial += 1
            bin_id = f"bin-{self._bin_serial}"
        header = BinHeader(bin_id, self, self.bar)
        self.bins[bin_id] = {"name": name, "members": list(members), "expanded": bool(expanded), "header": header}
        return bin_id

    def _default_bin_name(self):
        names = {b["name"] for b in self.bins.values()}
        n = 1
        while f"Bin {n}" in names:
            n += 1
        return f"Bin {n}"

    # ── bins: user actions ───────────────────────────────────────────────────

    def new_bin(self, tab_ids, name=None, rename=False):
        """Group tabs into a new bin where the first of them is. Returns the bin id."""
        tab_ids = [t for t in dict.fromkeys(tab_ids) if t in self.tabs]
        if not tab_ids:
            return None
        # A placeholder marks the spot: detaching tabs (and bins they empty) shifts everything else.
        marker = object()
        self.entries.insert(self.entries.index(self._top_level_of(tab_ids[0])), marker)
        for t in tab_ids:
            self._detach(t)
        bin_id = self._make_bin(name or self._default_bin_name(), tab_ids)
        self.entries[self.entries.index(marker)] = bin_id
        self._rebuild()
        if rename:
            QTimer.singleShot(0, self.bins[bin_id]["header"].start_rename)
        return bin_id

    def group_tabs(self, target_id, dragged_id, rename=False):
        """`dragged_id` was dropped on `target_id`: join the target's bin, or start a bin with both."""
        if target_id == dragged_id or target_id not in self.tabs or dragged_id not in self.tabs:
            return None
        bin_id = self.bin_of(target_id)
        if bin_id:
            self.add_to_bin(bin_id, dragged_id, self.bins[bin_id]["members"].index(target_id) + 1)
            return bin_id
        return self.new_bin([target_id, dragged_id], rename=rename)

    def add_to_bin(self, bin_id, tab_id, index=None):
        if bin_id not in self.bins or tab_id not in self.tabs:
            return
        members = self.bins[bin_id]["members"]
        if tab_id in members:
            old = members.index(tab_id)
            members.remove(tab_id)
            if index is not None and index > old:
                index -= 1
        else:
            self._detach(tab_id)
            if bin_id not in self.bins:
                return
        members.insert(len(members) if index is None else max(0, min(index, len(members))), tab_id)
        self._rebuild()

    def remove_from_bin(self, tab_id):
        """Move a tab out of its bin, to just after the bin."""
        bin_id = self.bin_of(tab_id)
        if not bin_id:
            return
        index = self.entries.index(bin_id) + 1
        self._detach(tab_id)
        if bin_id not in self.entries:
            index -= 1
        self.entries.insert(index, tab_id)
        self._rebuild()

    def ungroup(self, bin_id):
        """Dissolve a bin, leaving its tabs where it was."""
        if bin_id not in self.bins:
            return
        index = self.entries.index(bin_id)
        members = list(self.bins[bin_id]["members"])
        self._delete_bin(bin_id)
        self.entries[index:index] = members
        self._rebuild()

    def rename_bin(self, bin_id, name):
        name = " ".join(str(name).split())[:BIN_NAME_LIMIT]
        if bin_id in self.bins and name and name != self.bins[bin_id]["name"]:
            self.bins[bin_id]["name"] = name
            self._rebuild()

    def set_bin_expanded(self, bin_id, expanded):
        if bin_id in self.bins and self.bins[bin_id]["expanded"] != bool(expanded):
            self.bins[bin_id]["expanded"] = bool(expanded)
            self._rebuild()

    def toggle_bin(self, bin_id):
        if bin_id in self.bins:
            self.set_bin_expanded(bin_id, not self.bins[bin_id]["expanded"])

    def _sync_bin_highlight(self):
        """A collapsed bin hiding the open panel shows as selected."""
        current = self.stack.currentWidget()
        for b in self.bins.values():
            hides_current = not b["expanded"] and any(self.tabs[t]["widget"] is current for t in b["members"])
            b["header"].setChecked(hides_current)

    # ── bins: saving and restoring ───────────────────────────────────────────

    def bins_state(self):
        """Bins in bar order, for settings: [{"name", "members", "expanded"}]."""
        return [{"name": self.bins[e]["name"], "members": list(self.bins[e]["members"]),
                 "expanded": self.bins[e]["expanded"]} for e in self.entries if e in self.bins]

    def set_bins(self, specs):
        """Replace the bins with saved ones; tabs that no longer exist are skipped, empty bins dropped."""
        self.clear_bins(emit=False)
        for spec in specs or []:
            if not isinstance(spec, dict):
                continue
            members = [t for t in dict.fromkeys(spec.get("members") or [])
                       if isinstance(t, str) and t in self.entries]
            if not members:
                continue
            index = min(self.entries.index(t) for t in members)
            for t in members:
                self.entries.remove(t)
            name = " ".join(str(spec.get("name") or "").split())[:BIN_NAME_LIMIT] or self._default_bin_name()
            bin_id = self._make_bin(name, members, spec.get("expanded", True))
            self.entries.insert(index, bin_id)
        self._rebuild(emit=False)

    def clear_bins(self, emit=True):
        for bin_id in [e for e in self.entries if e in self.bins]:
            index = self.entries.index(bin_id)
            members = list(self.bins[bin_id]["members"])
            self._delete_bin(bin_id)
            self.entries[index:index] = members
        self._rebuild(emit=emit)

    # ── order ────────────────────────────────────────────────────────────────

    def move_tab_to_visible_slot(self, tab_id, slot):
        """Move a tab to a gap between the shown items. A gap inside an open bin puts it in that bin."""
        if tab_id not in self.tabs or not self.tabs[tab_id]["visible"]:
            return
        items = self.visible_items()
        before = (list(self.tab_order), self.bins_state())
        prev = items[slot - 1] if 0 < slot <= len(items) else None
        nxt = items[slot] if 0 <= slot < len(items) else None
        if nxt and nxt[0] == "tab" and nxt[1] == tab_id or prev and prev[0] == "tab" and prev[1] == tab_id:
            return  # dropped right next to itself

        # Inside an open bin: between its header and a member, or between two of its members
        nxt_bin = self.bin_of(nxt[1]) if nxt and nxt[0] == "tab" else None
        prev_bin = (prev[1] if prev and prev[0] == "bin" else self.bin_of(prev[1]) if prev else None)
        if nxt_bin and prev_bin == nxt_bin:
            self.add_to_bin(nxt_bin, tab_id, self.bins[nxt_bin]["members"].index(nxt[1]))
            return
        if prev and prev[0] == "bin" and self.bins[prev[1]]["expanded"] and not nxt_bin:
            self.add_to_bin(prev[1], tab_id, 0)  # just under the header of an open bin
            return

        # Top level: before the next item's top-level entry, else after the previous one's
        self._detach(tab_id)
        if nxt is not None:
            anchor = nxt[1] if nxt[0] == "bin" else self._top_level_of(nxt[1])
            index = self.entries.index(anchor) if anchor in self.entries else len(self.entries)
        elif prev is not None:
            anchor = prev[1] if prev[0] == "bin" else self._top_level_of(prev[1])
            index = self.entries.index(anchor) + 1 if anchor in self.entries else len(self.entries)
        else:
            index = len(self.entries)
        self.entries.insert(index, tab_id)
        self._rebuild(emit=False)
        if (list(self.tab_order), self.bins_state()) != before:
            self.tabOrderChanged.emit(list(self.tab_order))

    def move_bin_to_visible_slot(self, bin_id, slot):
        """Move a whole bin to a gap between the shown items. A gap inside another open bin means just
        after that bin: bins don't nest."""
        if bin_id not in self.bins:
            return
        items = self.visible_items()
        own = [i for i, (kind, item_id, _w) in enumerate(items)
               if (kind == "bin" and item_id == bin_id) or (kind == "tab" and self.bin_of(item_id) == bin_id)]
        if own and own[0] <= slot <= own[-1] + 1:
            return  # dropped onto or right around itself
        nxt = items[slot] if 0 <= slot < len(items) else None
        marker = object()
        if nxt is None:
            self.entries.append(marker)
        elif nxt[0] == "bin":
            self.entries.insert(self.entries.index(nxt[1]), marker)
        else:
            other = self.bin_of(nxt[1])
            if other is None:
                self.entries.insert(self.entries.index(nxt[1]), marker)
            else:  # inside another open bin
                self.entries.insert(self.entries.index(other) + 1, marker)
        self.entries.remove(bin_id)
        self.entries[self.entries.index(marker)] = bin_id
        self._rebuild()

    def set_tab_visible(self, tab_id, visible):
        if tab_id not in self.tabs:
            return
        entry = self.tabs[tab_id]
        if entry["visible"] == visible:
            return
        entry["visible"] = visible

        if not visible and self.currentWidget() is entry["widget"]:
            for tid in self.tab_order:
                if self.tabs[tid]["visible"]:
                    self.setCurrentWidget(self.tabs[tid]["widget"])
                    break
        self._rebuild(emit=False)
        self.tabVisibilityChanged.emit(tab_id, visible)

    def is_tab_visible(self, tab_id):
        return self.tabs.get(tab_id, {}).get("visible", True)

    def apply_tab_order(self, order):
        """Reorder tabs to `order` (unknown ids ignored, missing tabs kept at the end); bins stay together,
        placed where their first member falls."""
        valid = [tid for tid in dict.fromkeys(order) if tid in self.tabs]
        valid += [tid for tid in self.tab_order if tid not in valid]
        rank = {tid: i for i, tid in enumerate(valid)}
        for b in self.bins.values():
            b["members"].sort(key=lambda t: rank[t])
        entries, placed = [], set()
        for tid in valid:
            bin_id = self.bin_of(tid)
            if bin_id is None:
                entries.append(tid)
            elif bin_id not in placed:
                entries.append(bin_id)
                placed.add(bin_id)
        self.entries = entries
        self._rebuild(emit=False)

    def refresh_icons(self):
        """Re-render icons in the current theme's colors."""
        for item in self.tabs.values():
            if item["icon"]:
                item["button"].setIcon(svg_icon(item["icon"], TAB_ICON_COLORS))
        for b in self.bins.values():
            b["header"].setIcon(svg_icon("bin_expanded" if b["expanded"] else "bin_collapsed",
                                         TAB_ICON_COLORS, size=10))
        self.set_logo_icon()
        self.bar.update()

    # ── Auto-hide ────────────────────────────────────────────────────────────

    def set_auto_hide(self, enabled):
        """Collapse the bar to a thin edge that slides it back out on hover, or dock it again."""
        enabled = bool(enabled)
        if enabled == self.auto_hide:
            return
        self.auto_hide = enabled
        self._hide_timer.stop()
        if enabled:
            self.layout().removeWidget(self.bar)
            self.bar.hide()
            self.edge.show()
        else:
            self.edge.hide()
            self.layout().insertWidget(1, self.bar)
            self.bar.show()
        self.autoHideChanged.emit(enabled)

    def _reveal_bar(self):
        self._hide_timer.stop()
        width = max(self.bar.minimumWidth(), self.bar.sizeHint().width())
        self.bar.setGeometry(0, 0, width, self.height())
        self.bar.show()
        self.bar.raise_()

    def _tuck_bar(self):
        # Stay open while the pointer is still over the bar (e.g. coming back from a tooltip)
        if self.auto_hide and not self.bar.rect().contains(self.bar.mapFromGlobal(QCursor.pos())):
            self.bar.hide()

    def eventFilter(self, obj, event):
        if self.auto_hide:
            kind = event.type()
            if obj is self.edge and kind == QEvent.Type.Enter:
                self._reveal_bar()
            elif obj is self.bar and kind == QEvent.Type.Enter:
                self._hide_timer.stop()
            elif obj is self.bar and kind == QEvent.Type.Leave:
                self._hide_timer.start()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_hide and self.bar.isVisible():
            self.bar.setGeometry(0, 0, self.bar.width(), self.height())

    # ── QTabWidget-compatible API ────────────────────────────────────────────

    def count(self):
        return self.stack.count()

    def currentIndex(self):
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.stack.currentWidget()

    def set_settings_widget(self, widget):
        self.settings_widget = widget
        self.settings_index = self.stack.addWidget(widget)
        return self.settings_index

    def _on_logo_clicked(self):
        self.logoClicked.emit()
        if self.settings_widget is not None:
            self.setCurrentWidget(self.settings_widget)

    def widget(self, index):
        return self.stack.widget(index)

    def indexOf(self, widget):
        return self.stack.indexOf(widget)

    def setCurrentIndex(self, index):
        if not 0 <= index < self.count():
            return
        self.setCurrentWidget(self.stack.widget(index))

    def setCurrentWidget(self, widget):
        if widget is self.settings_widget:
            for item in self.tabs.values():
                item["button"].setChecked(False)
            self.logo_button.setChecked(True)
        else:
            self.logo_button.setChecked(False)
            for item in self.tabs.values():
                item["button"].setChecked(item["widget"] is widget)
        if widget is not None and widget != self.stack.currentWidget():
            self.stack.setCurrentWidget(widget)
            self.currentChanged.emit(self.stack.indexOf(widget))
        self._sync_bin_highlight()
