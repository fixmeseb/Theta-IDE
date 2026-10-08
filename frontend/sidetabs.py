"""Vertical tab bar with themed SVG icons: a drop-in for the subset of QTabWidget the window uses."""
from pathlib import Path

from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QSizePolicy, QStackedWidget, QToolButton,
                             QVBoxLayout, QWidget)

from .theme import theme_color

ICON_DIR = Path(__file__).resolve().parent / "icons"
ICON_SIZE = 20
AUTO_HIDE_EDGE = 6  # px strip left visible while the bar is tucked away
AUTO_HIDE_DELAY_MS = 350


def svg_icon(name, colors):
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
    source = source_path.read_text(encoding="utf-8")
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    icon = QIcon()
    for (mode, state), role in colors.items():
        renderer = QSvgRenderer(QByteArray(source.replace("currentColor", theme_color(role)).encode("utf-8")))
        pixmap = QPixmap(round(ICON_SIZE * ratio), round(ICON_SIZE * ratio))
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
    """Sidebar tab button that can be clicked to activate or dragged to reorder."""

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
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-thetatab-id", QByteArray(self.tab_id.encode("utf-8")))
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


class SideTabBar(QWidget):
    """Container widget that accepts tab drops and draws a reorder drop indicator."""

    def __init__(self, side_tabs, parent=None):
        super().__init__(parent)
        self.side_tabs = side_tabs
        self.setObjectName("sideTabs")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setAcceptDrops(True)
        self._drop_slot = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-thetatab-id"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-thetatab-id"):
            slot = self._calculate_drop_slot(event.position().y())
            if slot != self._drop_slot:
                self._drop_slot = slot
                self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_slot = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-thetatab-id"):
            tab_id = bytes(event.mimeData().data("application/x-thetatab-id")).decode("utf-8")
            slot = self._drop_slot
            self._drop_slot = None
            self.update()
            if slot is not None:
                self.side_tabs.move_tab_to_visible_slot(tab_id, slot)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _calculate_drop_slot(self, y):
        visible_buttons = [self.side_tabs.tabs[tid]["button"]
                           for tid in self.side_tabs.tab_order
                           if self.side_tabs.tabs[tid]["visible"]]
        if not visible_buttons:
            return 0
        for i, btn in enumerate(visible_buttons):
            mid_y = btn.geometry().center().y()
            if y < mid_y:
                return i
        return len(visible_buttons)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_slot is not None:
            visible_buttons = [self.side_tabs.tabs[tid]["button"]
                               for tid in self.side_tabs.tab_order
                               if self.side_tabs.tabs[tid]["visible"]]
            if not visible_buttons:
                return
            slot = self._drop_slot
            if slot == 0:
                y = float(visible_buttons[0].geometry().top() - 1)
            elif slot >= len(visible_buttons):
                y = float(visible_buttons[-1].geometry().bottom() + 2)
            else:
                prev_b = visible_buttons[slot - 1]
                next_b = visible_buttons[slot]
                y = (prev_b.geometry().bottom() + next_b.geometry().top()) / 2.0

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(theme_color("accent"))
            painter.setPen(QPen(color, 2))
            painter.drawLine(6, round(y), self.width() - 6, round(y))
            painter.setBrush(color)
            painter.drawEllipse(QRectF(3, y - 2.5, 5, 5))
            painter.drawEllipse(QRectF(self.width() - 8, y - 2.5, 5, 5))
            painter.end()


class SideTabs(QWidget):
    """Icon-over-label buttons stacked on the left, with the pages in a QStackedWidget beside them."""
    currentChanged = pyqtSignal(int)
    logoClicked = pyqtSignal()
    tabOrderChanged = pyqtSignal(list)
    tabVisibilityChanged = pyqtSignal(str, bool)
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

        # Insert button before the stretch spacer
        self.bar_layout.insertWidget(len(self.tab_order), button)
        self.tabs[tab_id] = {
            "widget": widget,
            "text": text,
            "short": short or text,
            "icon": icon,
            "button": button,
            "visible": True,
        }
        self.tab_order.append(tab_id)
        if icon:
            button.setIcon(svg_icon(icon, TAB_ICON_COLORS))
        if len(self.tab_order) == 1:
            button.setChecked(True)
        return index

    def removeTab(self, tab_id):
        """Remove a tab by tab_id and detach its button and widget."""
        if tab_id not in self.tabs:
            return
        entry = self.tabs.pop(tab_id)
        if tab_id in self.tab_order:
            self.tab_order.remove(tab_id)

        # If active widget is this one, switch to another visible tab
        if self.currentWidget() is entry["widget"]:
            switched = False
            for tid in self.tab_order:
                if self.tabs[tid]["visible"]:
                    self.setCurrentWidget(self.tabs[tid]["widget"])
                    switched = True
                    break
            if not switched and self.settings_widget:
                self.setCurrentWidget(self.settings_widget)

        self.bar_layout.removeWidget(entry["button"])
        entry["button"].deleteLater()
        self.stack.removeWidget(entry["widget"])
        self.tabOrderChanged.emit(list(self.tab_order))

    def move_tab_to_visible_slot(self, tab_id, slot):
        if tab_id not in self.tabs:
            return
        visible_ids = [tid for tid in self.tab_order if self.tabs[tid]["visible"]]
        if tab_id not in visible_ids:
            return
        cur_vis_idx = visible_ids.index(tab_id)
        if slot == cur_vis_idx or slot == cur_vis_idx + 1:
            return

        target_slot = slot - 1 if slot > cur_vis_idx else slot
        new_vis_ids = [tid for tid in visible_ids if tid != tab_id]
        new_vis_ids.insert(target_slot, tab_id)

        if target_slot == 0:
            neighbor = new_vis_ids[1]
            self.tab_order.remove(tab_id)
            idx = self.tab_order.index(neighbor)
            self.tab_order.insert(idx, tab_id)
        elif target_slot >= len(new_vis_ids) - 1:
            neighbor = new_vis_ids[-2]
            self.tab_order.remove(tab_id)
            idx = self.tab_order.index(neighbor) + 1
            self.tab_order.insert(idx, tab_id)
        else:
            neighbor = new_vis_ids[target_slot + 1]
            self.tab_order.remove(tab_id)
            idx = self.tab_order.index(neighbor)
            self.tab_order.insert(idx, tab_id)

        for i, tid in enumerate(self.tab_order):
            btn = self.tabs[tid]["button"]
            self.bar_layout.removeWidget(btn)
            self.bar_layout.insertWidget(i, btn)

        self.tabOrderChanged.emit(list(self.tab_order))

    def set_tab_visible(self, tab_id, visible):
        if tab_id not in self.tabs:
            return
        entry = self.tabs[tab_id]
        if entry["visible"] == visible:
            return
        entry["visible"] = visible
        entry["button"].setVisible(visible)

        if not visible and self.currentWidget() is entry["widget"]:
            for tid in self.tab_order:
                if self.tabs[tid]["visible"]:
                    self.setCurrentWidget(self.tabs[tid]["widget"])
                    break

        self.tabVisibilityChanged.emit(tab_id, visible)

    def is_tab_visible(self, tab_id):
        return self.tabs.get(tab_id, {}).get("visible", True)

    def apply_tab_order(self, order):
        valid_order = [tid for tid in order if tid in self.tabs]
        for tid in self.tab_order:
            if tid not in valid_order:
                valid_order.append(tid)
        self.tab_order = valid_order
        for i, tid in enumerate(self.tab_order):
            btn = self.tabs[tid]["button"]
            self.bar_layout.removeWidget(btn)
            self.bar_layout.insertWidget(i, btn)

    def refresh_icons(self):
        """Re-render icons in the current theme's colors."""
        for item in self.tabs.values():
            if item["icon"]:
                item["button"].setIcon(svg_icon(item["icon"], TAB_ICON_COLORS))
        self.set_logo_icon()

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
