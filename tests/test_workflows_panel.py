"""Unit tests for the WorkflowsPanel and interactive canvas."""

import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.workflows.workflows_panel import WorkflowsPanel
    from frontend.workflows.canvas import WorkflowScene
    from frontend.workflows.diagram_items import NodeBoxItem, StringItem
    from src.app.pipeline.workflow.presets import BUILTIN_PRESETS
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestWorkflowsPanel(unittest.TestCase):
    def setUp(self):
        self.panel = WorkflowsPanel()

    def tearDown(self):
        self.panel.deleteLater()

    def test_initial_state_loads_preset(self):
        # Default preset loaded is online_vs_offline
        scene = self.panel.scene
        self.assertGreaterEqual(len(scene.node_items), 2)
        self.assertGreaterEqual(len(scene.string_items), 1)

    def test_switch_presets(self):
        for pid in BUILTIN_PRESETS:
            self.panel.load_preset(pid)
            self.assertEqual(self.panel.scene.graph.id, BUILTIN_PRESETS[pid]().id)
            self.assertGreaterEqual(len(self.panel.scene.node_items), 2)
            self.assertGreaterEqual(len(self.panel.scene.string_items), 1)

    def test_add_node_from_palette(self):
        initial_count = len(self.panel.scene.node_items)
        self.panel._spawn_from_palette("online_rl")
        self.assertEqual(len(self.panel.scene.node_items), initial_count + 1)

    def test_validate_diagram(self):
        self.panel.load_preset("transfer_learning")
        issues = self.panel.scene.graph.validate()
        self.assertEqual(len(issues), 0)

    def test_clear_workflow(self):
        self.panel.clear_workflow()
        self.assertEqual(len(self.panel.scene.node_items), 0)
        self.assertEqual(len(self.panel.scene.string_items), 0)
