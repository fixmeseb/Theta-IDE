"""Interactive Qt graphics items for string diagrams (nodes, typed ports, and bezier wires)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from frontend.theme import theme_color
from src.app.pipeline.workflow.model import Port, PortDirection, PortType, WorkflowNode, WorkflowString

if TYPE_CHECKING:
    from .canvas import WorkflowScene


class PortItem(QGraphicsObject):
    """Circular interactive port anchor on the left or right of a NodeBoxItem."""
    clicked = pyqtSignal(object)  # self

    RADIUS = 6.0
    HOVER_RADIUS = 8.5

    def __init__(self, port: Port, parent_box: NodeBoxItem):
        super().__init__(parent_box)
        self.port = port
        self.parent_box = parent_box
        self.is_input = (port.direction == PortDirection.INPUT)
        self.setAcceptHoverEvents(True)
        self.hovered = False
        self.highlighted = False
        self.setToolTip(
            f"<b>{port.name}</b> ({port.port_type.display_name})<br>"
            f"<i>{port.description or 'No description'}</i>"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def boundingRect(self) -> QRectF:
        r = self.HOVER_RADIUS + 3.0
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        r = self.HOVER_RADIUS
        path.addEllipse(-r, -r, 2 * r, 2 * r)
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.HOVER_RADIUS if (self.hovered or self.highlighted) else self.RADIUS

        color = QColor(self.port.port_type.color)
        if self.highlighted:
            # Draw outer pulse / glow ring
            glow_pen = QPen(color, 3.0)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), r + 2.5, r + 2.5)

        # Port body
        painter.setPen(QPen(QColor(theme_color("surface")), 1.5))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), r, r)

        # Center inner dot
        painter.setPen(Qt.PenStyle.NoPen)
        inner_color = QColor(theme_color("text")) if self.hovered else QColor("#ffffff")
        painter.setBrush(QBrush(inner_color))
        painter.drawEllipse(QPointF(0, 0), 2.0, 2.0)

    def hoverEnterEvent(self, event) -> None:
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class StringItem(QGraphicsPathItem):
    """Cubic Bézier curve carrying typed artifacts between ports."""

    def __init__(self, wire: WorkflowString, src_port_item: PortItem, tgt_port_item: PortItem):
        super().__init__()
        self.wire = wire
        self.src_port_item = src_port_item
        self.tgt_port_item = tgt_port_item

        self.setZValue(-1)  # Behind node boxes
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_path()

    def update_path(self) -> None:
        p1 = self.src_port_item.scene_center()
        p2 = self.tgt_port_item.scene_center()

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()

        path = QPainterPath()
        path.moveTo(p1)

        # Standard forward flow (source to the left of target)
        if dx > 40:
            ctrl_offset = max(40.0, dx * 0.5)
            c1 = QPointF(p1.x() + ctrl_offset, p1.y())
            c2 = QPointF(p2.x() - ctrl_offset, p2.y())
            path.cubicTo(c1, c2, p2)
        else:
            # Feedback or backward loop: curve outward/downward
            loop_dist = max(70.0, abs(dy) * 0.5 + 40.0)
            c1 = QPointF(p1.x() + loop_dist, p1.y() + (50 if dy >= 0 else -50))
            c2 = QPointF(p2.x() - loop_dist, p2.y() + (50 if dy >= 0 else -50))
            path.cubicTo(c1, c2, p2)

        self.setPath(path)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        base_color = QColor(self.wire.port_type.color)

        if self.isSelected():
            # Halo effect when selected
            halo_pen = QPen(QColor(theme_color("accent")), 6.0, Qt.PenStyle.SolidLine)
            painter.setPen(halo_pen)
            painter.drawPath(self.path())

            wire_pen = QPen(QColor("#ffffff"), 3.0, Qt.PenStyle.SolidLine)
        else:
            wire_pen = QPen(base_color, 2.8, Qt.PenStyle.SolidLine)

        wire_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(wire_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())


class TempStringItem(QGraphicsPathItem):
    """Temporary interactive wire drawn while user drags from a port."""

    def __init__(self, src_port_item: PortItem):
        super().__init__()
        self.src_port_item = src_port_item
        self.target_pos = src_port_item.scene_center()
        self.is_valid = False
        self.setZValue(10)
        self.update_path()

    def set_target_pos(self, pos: QPointF, is_valid: bool = False) -> None:
        self.target_pos = pos
        self.is_valid = is_valid
        self.update_path()

    def update_path(self) -> None:
        p1 = self.src_port_item.scene_center()
        p2 = self.target_pos

        dx = p2.x() - p1.x()
        ctrl_offset = max(30.0, abs(dx) * 0.5)

        c1 = QPointF(p1.x() + (ctrl_offset if not self.src_port_item.is_input else -ctrl_offset), p1.y())
        c2 = QPointF(p2.x() - (ctrl_offset if not self.src_port_item.is_input else -ctrl_offset), p2.y())

        path = QPainterPath()
        path.moveTo(p1)
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.is_valid:
            color = QColor(self.src_port_item.port.port_type.color)
            pen = QPen(color, 3.0, Qt.PenStyle.DashLine)
        else:
            pen = QPen(QColor(theme_color("muted")), 2.0, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawPath(self.path())


class NodeBoxItem(QGraphicsObject):
    """Draggable box representing an experiment, paradigm, or transformation."""
    position_changed = pyqtSignal(object)  # self
    selected_changed = pyqtSignal(object)  # self

    WIDTH = 250.0
    HEADER_HEIGHT = 36.0
    PORT_ROW_HEIGHT = 24.0

    def __init__(self, node: WorkflowNode):
        super().__init__()
        self.node = node
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPos(node.pos_x, node.pos_y)

        self.input_items: dict[str, PortItem] = {}
        self.output_items: dict[str, PortItem] = {}

        self._build_ports()

    def _calculate_height(self) -> float:
        num_ports = max(len(self.node.inputs), len(self.node.outputs), 1)
        return self.HEADER_HEIGHT + num_ports * self.PORT_ROW_HEIGHT + 14.0

    def boundingRect(self) -> QRectF:
        h = self._calculate_height()
        return QRectF(0, 0, self.WIDTH, h)

    def _build_ports(self) -> None:
        # Build input ports along left edge (x = 0)
        y_offset = self.HEADER_HEIGHT + 16.0
        for port in self.node.inputs.values():
            item = PortItem(port, self)
            item.setPos(0, y_offset)
            self.input_items[port.name] = item
            y_offset += self.PORT_ROW_HEIGHT

        # Build output ports along right edge (x = WIDTH)
        y_offset = self.HEADER_HEIGHT + 16.0
        for port in self.node.outputs.values():
            item = PortItem(port, self)
            item.setPos(self.WIDTH, y_offset)
            self.output_items[port.name] = item
            y_offset += self.PORT_ROW_HEIGHT

    def get_port_item(self, port_name: str) -> PortItem | None:
        return self.input_items.get(port_name) or self.output_items.get(port_name)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.pos_x = self.pos().x()
            self.node.pos_y = self.pos().y()
            self.position_changed.emit(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.selected_changed.emit(self)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.boundingRect()
        h = rect.height()

        # 1. Box shadow / background card
        bg_color = QColor(theme_color("surface"))
        border_color = QColor(theme_color("accent")) if self.isSelected() else QColor(theme_color("border"))
        border_width = 2.0 if self.isSelected() else 1.2

        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        # 2. Header bar
        header_rect = QRectF(0, 0, self.WIDTH, self.HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 8.0, 8.0)
        # Flatten bottom corners of header
        header_cut = QPainterPath()
        header_cut.addRect(0, self.HEADER_HEIGHT - 6, self.WIDTH, 6)
        header_path = header_path.united(header_cut)

        header_bg = QColor(theme_color("panel"))
        painter.fillPath(header_path, QBrush(header_bg))
        painter.setPen(QPen(QColor(theme_color("border")), 1.0))
        painter.drawLine(0, int(self.HEADER_HEIGHT), int(self.WIDTH), int(self.HEADER_HEIGHT))

        # 3. Header title
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(theme_color("text")))

        # Elide label if too long
        metrics = painter.fontMetrics()
        elided_title = metrics.elidedText(self.node.label, Qt.TextElideMode.ElideRight, int(self.WIDTH - 16))
        painter.drawText(QRectF(10, 4, self.WIDTH - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_title)

        # 4. Paradigm badge
        badge_font = QFont()
        badge_font.setPointSize(7)
        badge_font.setBold(True)
        painter.setFont(badge_font)

        badge_text = f"[{self.node.paradigm}]"
        painter.setPen(QColor(theme_color("muted")))
        painter.drawText(QRectF(10, 20, self.WIDTH - 20, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, badge_text)

        # 5. Port labels
        port_font = QFont()
        port_font.setPointSize(8)
        painter.setFont(port_font)
        font_metrics = painter.fontMetrics()

        # Input labels (left)
        for name, item in self.input_items.items():
            py = item.pos().y()
            label_rect = QRectF(12, py - 10, self.WIDTH * 0.46 - 12, 20)
            painter.setPen(QColor(theme_color("muted")))
            elided = font_metrics.elidedText(name, Qt.TextElideMode.ElideRight, int(label_rect.width()))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)

        # Output labels (right)
        for name, item in self.output_items.items():
            py = item.pos().y()
            label_rect = QRectF(self.WIDTH * 0.54, py - 10, self.WIDTH * 0.46 - 12, 20)
            painter.setPen(QColor(theme_color("muted")))
            elided = font_metrics.elidedText(name, Qt.TextElideMode.ElideRight, int(label_rect.width()))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, elided)
