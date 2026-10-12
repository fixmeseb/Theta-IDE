import re
from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QColor, QPainter, QPen, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QFrame, QAbstractButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QTableWidgetItem, QScrollArea,
)
from .theme import theme_color
from .charts.chart import Chart, format_tick, nice_step  # noqa: F401  (moved; re-exported)


class _WheelNeedsFocus:
    """Only edit the value when the widget has been clicked into.

    Qt gives spin boxes and combo boxes WheelFocus by default, so they take
    focus from a passing wheel event and then consume it. Scrolling a long form
    silently rewrites every field the pointer crosses. Ignoring the event
    instead lets it reach the scroll area, which is what the user meant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SpinBox(_WheelNeedsFocus, QSpinBox):
    pass


class DoubleSpinBox(_WheelNeedsFocus, QDoubleSpinBox):
    pass


class ComboBox(_WheelNeedsFocus, QComboBox):
    pass


class SmoothScrollArea(QScrollArea):
    """QScrollArea that respects pixelDelta for high-precision, smooth trackpad and mouse scrolling."""

    def viewportEvent(self, event):
        if event.type() == event.Type.Wheel:
            pixel_delta = event.pixelDelta()
            if not pixel_delta.isNull():
                handled = False
                if pixel_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                    bar = self.verticalScrollBar()
                    bar.setValue(bar.value() - pixel_delta.y())
                    handled = True
                if pixel_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                    hbar = self.horizontalScrollBar()
                    hbar.setValue(hbar.value() - pixel_delta.x())
                    handled = True
                if handled:
                    event.accept()
                    return True
            angle_delta = event.angleDelta()
            if not angle_delta.isNull():
                handled = False
                if angle_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                    bar = self.verticalScrollBar()
                    step = bar.singleStep() if bar.singleStep() > 0 else 20
                    delta = int(round(angle_delta.y() / 120.0 * step * 2))
                    bar.setValue(bar.value() - delta)
                    handled = True
                if angle_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                    hbar = self.horizontalScrollBar()
                    step = hbar.singleStep() if hbar.singleStep() > 0 else 20
                    delta = int(round(angle_delta.x() / 120.0 * step * 2))
                    hbar.setValue(hbar.value() - delta)
                    handled = True
                if handled:
                    event.accept()
                    return True
        return super().viewportEvent(event)

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta()
        if not pixel_delta.isNull():
            handled = False
            if pixel_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                bar = self.verticalScrollBar()
                bar.setValue(bar.value() - pixel_delta.y())
                handled = True
            if pixel_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                hbar = self.horizontalScrollBar()
                hbar.setValue(hbar.value() - pixel_delta.x())
                handled = True
            if handled:
                event.accept()
                return
        angle_delta = event.angleDelta()
        if not angle_delta.isNull():
            handled = False
            if angle_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                bar = self.verticalScrollBar()
                step = bar.singleStep() if bar.singleStep() > 0 else 20
                delta = int(round(angle_delta.y() / 120.0 * step * 2))
                bar.setValue(bar.value() - delta)
                handled = True
            if angle_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                hbar = self.horizontalScrollBar()
                step = hbar.singleStep() if hbar.singleStep() > 0 else 20
                delta = int(round(angle_delta.x() / 120.0 * step * 2))
                hbar.setValue(hbar.value() - delta)
                handled = True
            if handled:
                event.accept()
                return
        super().wheelEvent(event)


def label(text, kind=None, wrap=False):
    """A QLabel styled by `kind` (its object name). wrap=True lets long text flow onto more lines
    instead of setting a minimum width for its whole layout."""
    result = QLabel(text)
    if kind:
        result.setObjectName(kind)
    result.setWordWrap(wrap)
    return result


class SortableItem(QTableWidgetItem):
    """Table cell that sorts by a value (number, timestamp or text) rather than by its displayed text.

    Cells without a value (shown as "—") sort last in either direction.
    """
    def __init__(self, text, key):
        super().__init__(text)
        self.key = key

    def __lt__(self, other):
        if not isinstance(other, SortableItem):
            return super().__lt__(other)
        if self.key is None and other.key is None:
            return False
        if self.key is None or other.key is None:
            table = self.tableWidget()
            descending = (table is not None
                          and table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder)
            return descending if self.key is None else not descending
        return self.key < other.key


class MetricCard(QFrame):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        self.title = label(title, "muted")
        layout.addWidget(self.title)
        self.value = label("—", "value")
        layout.addWidget(self.value)
        self.subtitle = label(subtitle, "muted")
        layout.addWidget(self.subtitle)


class YamlHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):
        for pattern, color in ((r'^\s*[\w_]+(?=:)', '#83a598'),
                               (r'\b\d+(?:\.\d+)?\b', '#d3869b'),
                               (r'"[^"\n]*"', '#b8bb26'), (r'#.*$', '#928374')):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(theme_color(color)))
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), len(match.group()), fmt)


class ToggleSlider(QAbstractButton):
    """Instant toggle slider switch with theme-aware pill track and knob."""

    def __init__(self, checked=True, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)

    def sizeHint(self):
        return QSize(40, 22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        radius = rect.height() / 2
        is_on = self.isChecked()

        track_color = QColor(theme_color("primary" if is_on else "raised"))
        border_color = QColor(theme_color("primary" if is_on else "border"))
        if not self.isEnabled():
            track_color.setAlpha(120)
            border_color.setAlpha(120)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        thumb_diameter = rect.height() - 6
        thumb_y = 3.0
        thumb_x = (rect.width() - thumb_diameter - 3.0) if is_on else 3.0
        knob_color = QColor(theme_color("base" if is_on else "muted"))
        if not self.isEnabled():
            knob_color.setAlpha(150)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diameter, thumb_diameter))
        painter.end()

