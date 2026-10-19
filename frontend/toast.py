"""In-app toasts: short notices that slide up from the window's bottom-right corner and dismiss themselves.

    ┌──────────────────────────┐
    │ ✓  cartpole_demo finished│
    │    Completed · 2m 14s    │
    │    [View results]     ✕  │
    └──────────────────────────┘

ToastStack keeps them stacked upward from the corner (newest at the bottom) and in place as the window resizes.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QToolButton

from .theme import theme_color

TOAST_WIDTH = 300
TOAST_TIMEOUT_MS = 8000  # paused while the pointer is over the toast
MARGIN = 16
SPACING = 8
MAX_TOASTS = 4


class Toast(QFrame):
    """`action` when its button is clicked (it then closes); `closed` once it is gone, however it went."""

    action = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, mark: str, title: str, detail: str = "", action_text: str = "", color: str = "accent",
                 timeout_ms: int = TOAST_TIMEOUT_MS, parent=None):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setFixedWidth(TOAST_WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#toast {{ background: {theme_color('surface')}; border: 1px solid {theme_color('border')};"
            f" border-left: 3px solid {theme_color(color)}; border-radius: 6px; }}"
            f"#toast QLabel {{ background: transparent; }}"
            f"#toast QToolButton {{ background: transparent; border: none; color: {theme_color('muted')};"
            f" font-size: 13px; padding: 2px 4px; }}"
            f"#toast QToolButton:hover {{ color: {theme_color('text')}; }}"
        )
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 10, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)

        self.mark = QLabel(mark)
        self.mark.setStyleSheet(f"color: {theme_color(color)}; font-size: 15px; font-weight: 700;")
        grid.addWidget(self.mark, 0, 0, Qt.AlignmentFlag.AlignTop)
        self.title = QLabel(title)
        self.title.setStyleSheet(f"color: {theme_color('text')}; font-weight: 600;")
        self.title.setWordWrap(True)
        grid.addWidget(self.title, 0, 1, 1, 2)
        self.detail = QLabel(detail)
        self.detail.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px;")
        self.detail.setVisible(bool(detail))
        grid.addWidget(self.detail, 1, 1, 1, 2)

        self.action_button = QPushButton(action_text)
        self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.action_button.setVisible(bool(action_text))
        self.action_button.clicked.connect(self._on_action)
        grid.addWidget(self.action_button, 2, 1, Qt.AlignmentFlag.AlignLeft)
        self.close_button = QToolButton()
        self.close_button.setText("✕")
        self.close_button.setToolTip("Dismiss")
        self.close_button.setAutoRaise(True)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.dismiss)
        grid.addWidget(self.close_button, 2, 2, Qt.AlignmentFlag.AlignRight)
        grid.setColumnStretch(1, 1)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(timeout_ms)
        self.timer.timeout.connect(self.dismiss)
        self._dismissed = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._dismissed and self.timer.interval() > 0:
            self.timer.start()

    def enterEvent(self, event):
        self.timer.stop()  # reading it: don't let it vanish
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._dismissed and self.timer.interval() > 0:
            self.timer.start()
        super().leaveEvent(event)

    def _on_action(self):
        self.action.emit()
        self.dismiss()

    def dismiss(self):
        if self._dismissed:
            return
        self._dismissed = True
        self.timer.stop()
        self.hide()
        self.closed.emit()
        self.deleteLater()


class ToastStack(QObject):
    """The window's toasts: stacked upward from its bottom-right corner, above the status bar."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.toasts: list[Toast] = []
        window.installEventFilter(self)

    def show(self, toast: Toast) -> Toast:
        toast.setParent(self.window)
        toast.closed.connect(lambda t=toast: self._forget(t))
        self.toasts.append(toast)
        while len(self.toasts) > MAX_TOASTS:
            self.toasts[0].dismiss()  # the oldest makes room
        self.relayout()  # in place before it shows: shown at (0, 0) first, a pointer there would pause it
        toast.show()
        toast.raise_()
        return toast

    def _forget(self, toast: Toast):
        if toast in self.toasts:
            self.toasts.remove(toast)
            self.relayout()

    def relayout(self):
        bar = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        bottom = self.window.height() - MARGIN - (bar.height() if bar is not None and bar.isVisible() else 0)
        for toast in reversed(self.toasts):  # newest at the bottom
            toast.adjustSize()
            top = bottom - toast.height()
            toast.move(self.window.width() - MARGIN - toast.width(), top)
            toast.raise_()
            bottom = top - SPACING

    def eventFilter(self, obj, event):
        if obj is self.window and event.type() == QEvent.Type.Resize and self.toasts:
            self.relayout()
        return False
