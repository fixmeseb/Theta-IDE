"""Interactive QGraphicsScene and QGraphicsView for the string diagram canvas."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
)

from frontend.theme import theme_color
from src.app.pipeline.workflow.model import (
    PortDirection,
    PortType,
    WorkflowGraph,
    WorkflowNode,
    WorkflowString,
)

from .diagram_items import NodeBoxItem, PortItem, StringItem, TempStringItem


class WorkflowScene(QGraphicsScene):
    """Scene managing workflow nodes, string wires, and connection dragging."""
    graph_changed = pyqtSignal()
    selection_changed_custom = pyqtSignal(object)  # NodeBoxItem | StringItem | None

    GRID_SIZE = 24.0

    def __init__(self, graph: WorkflowGraph | None = None, parent=None):
        super().__init__(parent)
        self.setSceneRect(-2000, -2000, 4000, 4000)

        self.graph = graph or WorkflowGraph(id="new_workflow", name="New Workflow")
        self.node_items: dict[str, NodeBoxItem] = {}
        self.string_items: dict[str, StringItem] = {}

        self.active_drag_port: PortItem | None = None
        self.temp_wire: TempStringItem | None = None
        self.hovered_target_port: PortItem | None = None

        self.selectionChanged.connect(self._on_scene_selection_changed)
        self.rebuild_from_graph()

    def set_graph(self, graph: WorkflowGraph) -> None:
        self.graph = graph
        self.rebuild_from_graph()
        self.graph_changed.emit()

    def rebuild_from_graph(self) -> None:
        self.clear()
        self.node_items.clear()
        self.string_items.clear()
        self.active_drag_port = None
        self.temp_wire = None
        self.hovered_target_port = None

        # 1. Add all nodes
        for node in self.graph.nodes.values():
            item = NodeBoxItem(node)
            item.position_changed.connect(self._on_node_position_changed)
            for p_item in list(item.input_items.values()) + list(item.output_items.values()):
                p_item.clicked.connect(self.on_port_clicked)
            self.addItem(item)
            self.node_items[node.id] = item

        # 2. Add all strings
        for wire in self.graph.strings:
            src_node_item = self.node_items.get(wire.source_node_id)
            tgt_node_item = self.node_items.get(wire.target_node_id)
            if src_node_item and tgt_node_item:
                src_port = src_node_item.output_items.get(wire.source_port_name)
                tgt_port = tgt_node_item.input_items.get(wire.target_port_name)
                if src_port and tgt_port:
                    s_item = StringItem(wire, src_port, tgt_port)
                    self.addItem(s_item)
                    self.string_items[wire.id] = s_item

    def add_node_to_scene(self, node: WorkflowNode) -> NodeBoxItem:
        self.graph.add_node(node)
        item = NodeBoxItem(node)
        item.position_changed.connect(self._on_node_position_changed)
        for p_item in list(item.input_items.values()) + list(item.output_items.values()):
            p_item.clicked.connect(self.on_port_clicked)
        self.addItem(item)
        self.node_items[node.id] = item
        self.graph_changed.emit()
        return item

    def remove_node_from_scene(self, node_id: str) -> None:
        item = self.node_items.get(node_id)
        if item:
            self.removeItem(item)
            del self.node_items[node_id]

        # Remove attached wires
        attached_wires = [
            wid for wid, s in self.string_items.items()
            if s.wire.source_node_id == node_id or s.wire.target_node_id == node_id
        ]
        for wid in attached_wires:
            self.removeItem(self.string_items[wid])
            del self.string_items[wid]

        self.graph.remove_node(node_id)
        self.graph_changed.emit()

    def remove_string_from_scene(self, wire_id: str) -> None:
        item = self.string_items.get(wire_id)
        if item:
            self.removeItem(item)
            del self.string_items[wire_id]
        self.graph.disconnect(wire_id)
        self.graph_changed.emit()

    def _on_node_position_changed(self, node_item: NodeBoxItem) -> None:
        # Update any strings connected to this node
        for s_item in self.string_items.values():
            if s_item.wire.source_node_id == node_item.node.id or s_item.wire.target_node_id == node_item.node.id:
                s_item.update_path()
        self.graph_changed.emit()

    def _on_scene_selection_changed(self) -> None:
        selected = self.selectedItems()
        if not selected:
            self.selection_changed_custom.emit(None)
            return

        # Prefer first selected NodeBoxItem or StringItem
        first = selected[0]
        if isinstance(first, (NodeBoxItem, StringItem)):
            self.selection_changed_custom.emit(first)
        else:
            self.selection_changed_custom.emit(None)

    def on_port_clicked(self, port_item: PortItem) -> None:
        if self.active_drag_port is None:
            # Start dragging from this port
            self.active_drag_port = port_item
            port_item.highlighted = True
            port_item.update()

            self.temp_wire = TempStringItem(port_item)
            self.addItem(self.temp_wire)
        else:
            # Clicked a second port — attempt connection
            self._finalize_connection(port_item)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.temp_wire and self.active_drag_port:
            mouse_pos = event.scenePos()

            # Find nearby port to snap to
            target_port = self._find_compatible_port_at(mouse_pos)
            if target_port != self.hovered_target_port:
                if self.hovered_target_port:
                    self.hovered_target_port.highlighted = False
                    self.hovered_target_port.update()
                self.hovered_target_port = target_port
                if self.hovered_target_port:
                    self.hovered_target_port.highlighted = True
                    self.hovered_target_port.update()

            if self.hovered_target_port:
                self.temp_wire.set_target_pos(self.hovered_target_port.scene_center(), is_valid=True)
            else:
                self.temp_wire.set_target_pos(mouse_pos, is_valid=False)

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # If right click while dragging, cancel wire
        if event.button() == Qt.MouseButton.RightButton and self.temp_wire:
            self._cancel_wire_drag()
            event.accept()
            return
        super().mousePressEvent(event)

    def _find_compatible_port_at(self, scene_pos: QPointF, snap_distance: float = 30.0) -> PortItem | None:
        if not self.active_drag_port:
            return None

        src_port = self.active_drag_port.port
        src_node_id = self.active_drag_port.parent_box.node.id

        best_port = None
        min_dist = snap_distance

        for node_item in self.node_items.values():
            # Disallow self-connecting the same node's ports
            if node_item.node.id == src_node_id:
                continue

            ports_to_check = (
                node_item.input_items.values()
                if self.active_drag_port.port.direction == PortDirection.OUTPUT
                else node_item.output_items.values()
            )

            for target_port_item in ports_to_check:
                tgt_port = target_port_item.port
                if not WorkflowGraph.is_type_compatible(src_port.port_type, tgt_port.port_type):
                    continue

                dist = math.hypot(
                    target_port_item.scene_center().x() - scene_pos.x(),
                    target_port_item.scene_center().y() - scene_pos.y()
                )
                if dist < min_dist:
                    min_dist = dist
                    best_port = target_port_item

        return best_port

    def _finalize_connection(self, target_port_item: PortItem) -> None:
        if not self.active_drag_port:
            return

        p1 = self.active_drag_port
        p2 = target_port_item

        # Must connect Output -> Input
        if p1.port.direction == PortDirection.OUTPUT and p2.port.direction == PortDirection.INPUT:
            src, tgt = p1, p2
        elif p1.port.direction == PortDirection.INPUT and p2.port.direction == PortDirection.OUTPUT:
            src, tgt = p2, p1
        else:
            self._cancel_wire_drag()
            return

        if src.parent_box.node.id == tgt.parent_box.node.id:
            self._cancel_wire_drag()
            return

        try:
            wire = self.graph.connect(
                source_node_id=src.parent_box.node.id,
                source_port_name=src.port.name,
                target_node_id=tgt.parent_box.node.id,
                target_port_name=tgt.port.name,
            )
            # Remove any existing graphical wire targeting this port
            for wid, s_item in list(self.string_items.items()):
                if s_item.wire.target_node_id == tgt.parent_box.node.id and s_item.wire.target_port_name == tgt.port.name:
                    self.removeItem(s_item)
                    del self.string_items[wid]

            s_item = StringItem(wire, src, tgt)
            self.addItem(s_item)
            self.string_items[wire.id] = s_item
            self.graph_changed.emit()
        except (ValueError, TypeError) as exc:
            pass

        self._cancel_wire_drag()

    def _cancel_wire_drag(self) -> None:
        if self.active_drag_port:
            self.active_drag_port.highlighted = False
            self.active_drag_port.update()
            self.active_drag_port = None
        if self.hovered_target_port:
            self.hovered_target_port.highlighted = False
            self.hovered_target_port.update()
            self.hovered_target_port = None
        if self.temp_wire:
            self.removeItem(self.temp_wire)
            self.temp_wire = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected = self.selectedItems()
            for item in selected:
                if isinstance(item, NodeBoxItem):
                    self.remove_node_from_scene(item.node.id)
                elif isinstance(item, StringItem):
                    self.remove_string_from_scene(item.wire.id)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Escape and self.temp_wire:
            self._cancel_wire_drag()
            event.accept()
            return
        super().keyPressEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        # Subtle dark background with fine dot grid
        painter.fillRect(rect, QColor(theme_color("base")))

        dot_pen = QPen(QColor(theme_color("raised")), 1.2)
        painter.setPen(dot_pen)

        left = int(rect.left()) - (int(rect.left()) % int(self.GRID_SIZE))
        top = int(rect.top()) - (int(rect.top()) % int(self.GRID_SIZE))

        points = []
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                points.append(QPointF(x, y))
                y += self.GRID_SIZE
            x += self.GRID_SIZE

        if points:
            painter.drawPoints(points)


class WorkflowCanvasView(QGraphicsView):
    """Viewport supporting smooth pan and zoom for the string diagram canvas."""

    def __init__(self, scene: WorkflowScene, parent=None):
        super().__init__(scene, parent)
        self.workflow_scene = scene

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event: QWheelEvent) -> None:
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current_scale = self.transform().m11()

        # Clamp zoom between 0.3x and 2.5x
        if 0.3 <= current_scale * zoom_factor <= 2.5:
            self.scale(zoom_factor, zoom_factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def zoom_fit(self) -> None:
        """Fit all nodes in view with generous margins."""
        items_rect = self.workflow_scene.itemsBoundingRect()
        if not items_rect.isEmpty():
            self.fitInView(items_rect.adjusted(-60, -60, 60, 60), Qt.AspectRatioMode.KeepAspectRatio)
