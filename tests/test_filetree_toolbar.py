"""Unit tests for the ConfigTreeWidget header toolbar:
- 2-row layout with left and right groups per row
- 4 creation buttons positioned on row 2 left
- Draggable buttons that can be reordered within groups and moved between groups
- Persistence and resetting to default layout
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.config_tree import ConfigTreeWidget, DraggableToolButton, TreeHeaderWidget


class TestFileTreeToolbar(unittest.TestCase):
    def setUp(self):
        QSettings("ThetaIDE", "ThetaIDE").clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "experiment" / "cartpole" / "exp1.yaml").write_text("experiment_id: exp1\n", encoding="utf-8")
        self.tree_exp = ConfigTreeWidget(root_dir=self.root, mode="experiments")

    def tearDown(self):
        self.tree_exp.close()
        self.temp_dir.cleanup()
        QSettings("ThetaIDE", "ThetaIDE").clear()

    def test_default_two_row_layout(self):
        """Verify the 4 creation buttons are placed on row 2 left, and other buttons on row 1."""
        header_widget = self.tree_exp.header_widget
        self.assertIsInstance(header_widget, TreeHeaderWidget)
        groups = header_widget.header_layout.groups

        # Row 1 Left: reload, collapse_all, expand_all, search
        r1_left_ids = [b.button_id for b in groups["row1_left"]]
        self.assertEqual(r1_left_ids, ["reload", "collapse_all", "expand_all", "search"])

        # Row 1 Right: launch, export, raw_yaml
        r1_right_ids = [b.button_id for b in groups["row1_right"]]
        self.assertEqual(r1_right_ids, ["launch", "export", "raw_yaml"])

        # Row 2 Left: the 4 creation buttons
        r2_left_ids = [b.button_id for b in groups["row2_left"]]
        self.assertEqual(r2_left_ids, ["new_folder", "new_subfolder", "new_file", "copy"])

        # Row 2 Right: empty initially
        r2_right_ids = [b.button_id for b in groups["row2_right"]]
        self.assertEqual(r2_right_ids, [])

    def test_buttons_are_draggable(self):
        """Verify toolbar buttons are DraggableToolButton instances."""
        for b in [
            self.tree_exp.btn_new_group,
            self.tree_exp.btn_new_subfolder,
            self.tree_exp.btn_new_experiment,
            self.tree_exp.btn_duplicate,
            self.tree_exp.btn_refresh,
            self.tree_exp.btn_launch,
        ]:
            self.assertIsInstance(b, DraggableToolButton)

    def test_reorder_buttons_within_group(self):
        """Verify moving a button within the same group reorders it."""
        hw = self.tree_exp.header_widget
        # Move 'copy' (index 3) to the beginning (index 0) of row2_left
        hw.move_button("copy", "row2_left", 0)

        r2_left_ids = [b.button_id for b in hw.header_layout.groups["row2_left"]]
        self.assertEqual(r2_left_ids, ["copy", "new_folder", "new_subfolder", "new_file"])

    def test_move_button_between_groups(self):
        """Verify moving a button between different groups."""
        hw = self.tree_exp.header_widget
        # Move 'copy' from row2_left to row1_right at index 0
        hw.move_button("copy", "row1_right", 0)

        r2_left_ids = [b.button_id for b in hw.header_layout.groups["row2_left"]]
        self.assertNotIn("copy", r2_left_ids)
        self.assertEqual(r2_left_ids, ["new_folder", "new_subfolder", "new_file"])

        r1_right_ids = [b.button_id for b in hw.header_layout.groups["row1_right"]]
        self.assertIn("copy", r1_right_ids)
        self.assertEqual(r1_right_ids, ["copy", "launch", "export", "raw_yaml"])

    def test_move_button_to_empty_group(self):
        """Verify moving a button into row2_right (initially empty)."""
        hw = self.tree_exp.header_widget
        hw.move_button("new_file", "row2_right", 0)

        r2_right_ids = [b.button_id for b in hw.header_layout.groups["row2_right"]]
        self.assertEqual(r2_right_ids, ["new_file"])

        r2_left_ids = [b.button_id for b in hw.header_layout.groups["row2_left"]]
        self.assertNotIn("new_file", r2_left_ids)

    def test_reset_toolbar_layout(self):
        """Verify resetting restores the default layout."""
        hw = self.tree_exp.header_widget
        hw.move_button("copy", "row1_right", 0)
        hw.move_button("new_file", "row2_right", 0)

        self.tree_exp.reset_toolbar_layout()

        groups = hw.header_layout.groups
        self.assertEqual([b.button_id for b in groups["row2_left"]], ["new_folder", "new_subfolder", "new_file", "copy"])
        self.assertEqual([b.button_id for b in groups["row1_right"]], ["launch", "export", "raw_yaml"])
        self.assertEqual([b.button_id for b in groups["row2_right"]], [])

    def test_components_mode_default_layout(self):
        """Verify components mode has creation buttons on row 2 left."""
        tree_comp = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            hw = tree_comp.header_widget
            groups = hw.header_layout.groups
            self.assertEqual([b.button_id for b in groups["row1_left"]], ["reload", "collapse_all", "expand_all", "search"])
            self.assertEqual([b.button_id for b in groups["row1_right"]], ["plus", "hub", "raw_yaml"])
            self.assertEqual([b.button_id for b in groups["row2_left"]], ["copy", "deep_copy"])
            self.assertEqual([b.button_id for b in groups["row2_right"]], [])
        finally:
            tree_comp.close()


if __name__ == "__main__":
    unittest.main()
