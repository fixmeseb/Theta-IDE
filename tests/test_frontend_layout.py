"""Tests for frontend layout enhancements: draggable/reorderable sidebar icons and pane visibility sliders."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure QApplication exists in offscreen mode for tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.widgets import ToggleSlider
    from frontend.sidetabs import SideTabs
    from frontend.app import Window
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestToggleSlider(unittest.TestCase):
    def test_initial_state_and_toggle(self):
        slider = ToggleSlider(checked=True)
        self.assertTrue(slider.isChecked())

        slider.setChecked(False)
        self.assertFalse(slider.isChecked())

        # Test click toggle
        events = []
        slider.toggled.connect(lambda val: events.append(val))
        slider.click()
        self.assertTrue(slider.isChecked())
        self.assertEqual(events, [True])

        slider.click()
        self.assertFalse(slider.isChecked())
        self.assertEqual(events, [True, False])

    def test_size_and_paint(self):
        slider = ToggleSlider(checked=True)
        self.assertEqual(slider.sizeHint(), QSize(40, 22))

        pixmap = QPixmap(slider.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        slider.render(pixmap)
        self.assertFalse(pixmap.isNull())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSideTabsDraggableAndVisibility(unittest.TestCase):
    def setUp(self):
        self.side_tabs = SideTabs()
        self.w1 = QLabel("Monitor Widget")
        self.w2 = QLabel("Results Widget")
        self.w3 = QLabel("Config Widget")
        self.w4 = QLabel("Plots Widget")

        self.side_tabs.addTab(self.w1, "Training monitor", "monitor", "Monitor", tab_id="monitor")
        self.side_tabs.addTab(self.w2, "Results browser", "results", "Results", tab_id="results")
        self.side_tabs.addTab(self.w3, "Experiment", "config", "Experiment", tab_id="config")
        self.side_tabs.addTab(self.w4, "Plot viewer", "plots", "Plots", tab_id="plots")

    def test_initial_tabs(self):
        self.assertEqual(self.side_tabs.tab_order, ["monitor", "results", "config", "plots"])
        self.assertTrue(self.side_tabs.is_tab_visible("monitor"))
        self.assertTrue(self.side_tabs.is_tab_visible("results"))
        self.assertEqual(self.side_tabs.currentWidget(), self.w1)

    def test_reorder_tabs(self):
        order_changes = []
        self.side_tabs.tabOrderChanged.connect(lambda order: order_changes.append(list(order)))

        # Move "monitor" from visible slot 0 to after "results" (slot 2)
        self.side_tabs.move_tab_to_visible_slot("monitor", 2)
        self.assertEqual(self.side_tabs.tab_order, ["results", "monitor", "config", "plots"])
        self.assertEqual(len(order_changes), 1)

        # Move "plots" to the very top (slot 0)
        self.side_tabs.move_tab_to_visible_slot("plots", 0)
        self.assertEqual(self.side_tabs.tab_order, ["plots", "results", "monitor", "config"])

    def test_set_tab_visible(self):
        vis_events = []
        self.side_tabs.tabVisibilityChanged.connect(lambda tid, vis: vis_events.append((tid, vis)))

        # Hide "results"
        self.side_tabs.set_tab_visible("results", False)
        self.assertFalse(self.side_tabs.is_tab_visible("results"))
        self.assertTrue(self.side_tabs.tabs["results"]["button"].isHidden())
        self.assertEqual(vis_events, [("results", False)])

        # Show "results"
        self.side_tabs.set_tab_visible("results", True)
        self.assertTrue(self.side_tabs.is_tab_visible("results"))
        self.assertFalse(self.side_tabs.tabs["results"]["button"].isHidden())
        self.assertEqual(vis_events, [("results", False), ("results", True)])

    def test_hide_active_tab_auto_switches(self):
        # Current widget is w1 (monitor)
        self.assertEqual(self.side_tabs.currentWidget(), self.w1)

        # Hide monitor -> should auto-switch to next visible tab (results, w2)
        self.side_tabs.set_tab_visible("monitor", False)
        self.assertEqual(self.side_tabs.currentWidget(), self.w2)

    def test_apply_tab_order(self):
        custom_order = ["plots", "config", "results", "monitor"]
        self.side_tabs.apply_tab_order(custom_order)
        self.assertEqual(self.side_tabs.tab_order, custom_order)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestWindowLayoutAndSliders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_default_tab_order(self):
        expected_order = ["components", "config", "workflows", "monitor", "results", "plots", "tensorboard", "queue", "terminal", "console"]
        self.assertEqual(self.window.tabs.tab_order, expected_order)

    def test_pane_sliders_registered(self):
        expected_panes = ["components", "config", "workflows", "monitor", "results", "plots", "tensorboard", "queue", "terminal", "console"]
        for pane_id in expected_panes:
            self.assertIn(pane_id, self.window.pane_sliders)
            slider = self.window.pane_sliders[pane_id]
            self.assertIsInstance(slider, ToggleSlider)
            # Sliders start in step with the default sidebar visibility in settings.toml
            self.assertEqual(slider.isChecked(), pane_id in self.window.settings_manager.sidebar_visible)

    def test_toggle_slider_hides_pane(self):
        # Plots is hidden by default, so show it first
        self.window.pane_sliders["plots"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("plots"))

        # Toggle plots off
        self.window.pane_sliders["plots"].setChecked(False)
        self.assertFalse(self.window.tabs.is_tab_visible("plots"))

        # Toggle plots back on
        self.window.pane_sliders["plots"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("plots"))

    def test_cannot_hide_all_panes(self):
        # Turn off all panes except one
        panes = ["components", "config", "workflows", "monitor", "results", "plots", "tensorboard", "queue", "terminal", "console"]
        for p in panes[:-1]:
            self.window.pane_sliders[p].setChecked(False)

        # The last remaining pane should be disabled so it cannot be turned off
        last_slider = self.window.pane_sliders[panes[-1]]
        self.assertTrue(last_slider.isChecked())
        self.assertFalse(last_slider.isEnabled())

        # Attempt to toggle off the last pane
        self.window.on_pane_slider_toggled(panes[-1], False)
        self.assertTrue(last_slider.isChecked())

    def test_reset_sidebar_layout(self):
        # Alter order and hide some panes
        self.window.tabs.move_tab_to_visible_slot("plots", 0)
        self.window.pane_sliders["monitor"].setChecked(False)

        # Reset
        self.window.reset_sidebar_layout()
        default_order = ["components", "config", "workflows", "monitor", "results", "plots", "tensorboard", "queue", "terminal", "console"]
        self.assertEqual(self.window.tabs.tab_order, default_order)
        for p in default_order:
            self.assertTrue(self.window.tabs.is_tab_visible(p))
            self.assertTrue(self.window.pane_sliders[p].isChecked())

    def test_save_and_load_layout(self):
        # Reorder and hide a pane
        self.window.tabs.move_tab_to_visible_slot("terminal", 0)
        self.window.pane_sliders["console"].setChecked(False)
        self.window.save_layout()

        self.assertEqual(self.window.settings_manager.sidebar_order[0], "terminal")
        self.assertNotIn("console", self.window.settings_manager.sidebar_visible)

        # Create a new Window pointing to the same data directory
        window2 = Window(self.data_dir)
        try:
            self.assertEqual(window2.tabs.tab_order[0], "terminal")
            self.assertFalse(window2.tabs.is_tab_visible("console"))
            self.assertFalse(window2.pane_sliders["console"].isChecked())
        finally:
            window2.close()


if __name__ == "__main__":
    unittest.main()
