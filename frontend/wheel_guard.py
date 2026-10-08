"""Stop the mouse wheel from changing input widgets the user is only scrolling past.

Combo boxes, spin boxes and sliders react to the wheel whenever the cursor is over them, so
scrolling a long form silently rewrites whatever passes under the pointer. With this filter
installed they only take the wheel once focused (clicked or tabbed into); otherwise the event
goes to the parent, so the surrounding scroll area keeps scrolling.
"""
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox, QScrollBar

GUARDED = (QComboBox, QAbstractSpinBox, QAbstractSlider)


def guarded(obj):
    return isinstance(obj, GUARDED) and not isinstance(obj, QScrollBar)


class WheelGuard(QObject):
    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Type.Polish and guarded(obj):
            # Spin boxes default to WheelFocus, which would let the wheel focus them and then edit them
            if obj.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        elif kind == QEvent.Type.Wheel and guarded(obj) and not obj.hasFocus():
            forward(obj, event)
            return True
        return False


def forward(widget, event):
    """Offer the wheel to each ancestor until one takes it; Qt won't propagate a re-sent event itself."""
    global_pos = event.globalPosition()
    parent = widget.parentWidget()
    while parent is not None:
        copy = QWheelEvent(parent.mapFromGlobal(global_pos), global_pos, event.pixelDelta(), event.angleDelta(),
                           event.buttons(), event.modifiers(), event.phase(), event.inverted())
        QApplication.sendEvent(parent, copy)
        if copy.isAccepted() or parent.isWindow():
            return
        parent = parent.parentWidget()


def install(app):
    """Install the guard on the application; keep the returned object alive with the app."""
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
