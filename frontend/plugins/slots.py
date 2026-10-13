"""UI Extension Slots for loose coupling between IDE panels and plugins."""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


class ExtensionSlot(QObject):
    """A named insertion point where plugins can contribute actions, buttons, or widgets."""
    changed = pyqtSignal()

    def __init__(self, slot_id: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.slot_id = slot_id
        # {action_id: (factory_or_widget, priority)}
        self._items: Dict[str, Tuple[Union[Callable[[], Optional[QWidget]], QWidget], int]] = {}

    def add_action(
        self,
        action_id: str,
        factory_or_widget: Union[Callable[[], Optional[QWidget]], QWidget],
        priority: int = 100,
    ) -> None:
        """Register an action widget or widget factory under this slot."""
        self._items[action_id] = (factory_or_widget, priority)
        self.changed.emit()

    def remove_action(self, action_id: str) -> None:
        """Remove an action from this slot if present."""
        if action_id in self._items:
            del self._items[action_id]
            self.changed.emit()

    def get_items(self) -> List[Tuple[str, Union[Callable[[], Optional[QWidget]], QWidget]]]:
        """Return contributed items ordered by priority (lower number = higher priority)."""
        sorted_pairs = sorted(self._items.items(), key=lambda kv: kv[1][1])
        return [(k, v[0]) for k, v in sorted_pairs]


class ExtensionSlotRegistry(QObject):
    """Registry maintaining active extension slots across Theta-IDE."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._slots: Dict[str, ExtensionSlot] = {}

    def get_slot(self, slot_id: str, create: bool = True) -> Optional[ExtensionSlot]:
        """Retrieve or lazily create an ExtensionSlot by ID."""
        if slot_id not in self._slots and create:
            self._slots[slot_id] = ExtensionSlot(slot_id, parent=self)
        return self._slots.get(slot_id)

    def clear(self) -> None:
        """Clear all registered slots."""
        self._slots.clear()


class ExtensionSlotWidget(QWidget):
    """Container QWidget that renders contributions to an ExtensionSlot dynamically."""

    def __init__(
        self,
        slot: ExtensionSlot,
        orientation: str = "horizontal",
        spacing: int = 8,
        margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.slot = slot
        self.layout_kind = orientation
        if orientation == "horizontal":
            self._layout = QHBoxLayout(self)
        else:
            self._layout = QVBoxLayout(self)
        self._layout.setSpacing(spacing)
        self._layout.setContentsMargins(*margins)
        self._rendered_widgets: Dict[str, QWidget] = {}

        self.slot.changed.connect(self._rebuild)
        self._rebuild()

    def _rebuild(self) -> None:
        """Re-instantiate widgets when slot contributions change."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._rendered_widgets.clear()

        items = self.slot.get_items()
        for action_id, factory_or_w in items:
            widget = factory_or_w() if callable(factory_or_w) else factory_or_w
            if widget:
                self._rendered_widgets[action_id] = widget
                self._layout.addWidget(widget)

        self.setVisible(bool(items))
