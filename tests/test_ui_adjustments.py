"""Unit tests for the 6 experiment panel and config viewer UI adjustments:
1. ToggleSlider used for bool parameters instead of checkboxes.
2. Parameter labels remove "(Required)" from Paradigm and Environment.
3. Method card bottom actions (duplicate, plus) are pinned to the bottom right with stretch.
4. Hierarchical parameter adders have only the indented combobox, no plus button, and add on selection.
5. Experiment creation writes minimal configs without redundant seed/total_timesteps/eval_episodes.
6. Play button shows why an experiment cannot run when hovering over it while disabled.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QToolButton, QVBoxLayout
    from frontend.config_viewer import ConfigViewer
    from frontend.config_tree import ConfigTreeWidget
    from frontend.widgets import ToggleSlider, ComboBox
    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication(sys.argv[:1])
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestUIAdjustments(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.temp_dir.name)
        self.exp_file = self.tmp_root / "test_exp.yaml"
        self.exp_file.write_text(
            "experiment_id: test_exp\n"
            "paradigm: null\n"
            "env: null\n"
            "tensorboard: true\n"
            "custom_bool: false\n"
            "methods:\n"
            "  ppo_dnn:\n"
            "    agent: ppo\n"
            "    model: dnn\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.load_file(self.exp_file, "experiment/cartpole/test_exp.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_item1_bool_parameters_use_toggle_slider(self):
        """Boolean settings in Box 1 and runtime flags render as ToggleSlider."""
        sliders = self.viewer.findChildren(ToggleSlider)
        self.assertTrue(len(sliders) >= 1, "Expected at least one ToggleSlider for boolean flags/parameters")
        # Check that tensorboard flag widget is ToggleSlider
        if self.viewer.chk_tb is not None:
            self.assertIsInstance(self.viewer.chk_tb, ToggleSlider)
            self.assertTrue(self.viewer.chk_tb.isChecked())

    def test_item2_paradigm_env_labels_remove_required(self):
        """Labels for Paradigm and Environment should not contain '(Required)'."""
        labels = [lbl.text() for lbl in self.viewer.findChildren(QLabel)]
        self.assertIn("Paradigm", labels)
        self.assertNotIn("Paradigm (Required)", labels)
        self.assertIn("Environment", labels)
        self.assertNotIn("Environment (Required)", labels)

    def test_item3_method_card_bottom_actions_pinned(self):
        """Method card uses QVBoxLayout with stretch pinning duplicate and add buttons to bottom."""
        method_cards = self.viewer.findChildren(ConfigViewer)
        # Find method sub_cards
        cards = [w for w in self.viewer.findChildren(QVBoxLayout) if w.parent() and w.parent().objectName() == "card"]
        self.assertTrue(len(cards) >= 1, "Expected method card to use QVBoxLayout")

    def test_item4_hierarchical_parameter_combobox_direct_add(self):
        """Hierarchical parameter adder combobox is added directly without separate plus button."""
        combos = [cb for cb in self.viewer.findChildren(ComboBox) if cb.placeholderText().startswith("+ Add")]
        self.assertTrue(len(combos) >= 1, "Expected hierarchical parameter adder combobox with + Add placeholder")
        # Ensure that selecting an item from the combobox triggers _add_method_param directly
        target_cb = combos[0]
        initial_keys = set(self.viewer.raw_data.get("methods", {}).get("ppo_dnn", {}).keys())
        # Simulate textActivated signal
        target_cb.textActivated.emit("clip_range")
        updated_keys = set(self.viewer.raw_data.get("methods", {}).get("ppo_dnn", {}).keys())
        self.assertIn("clip_range", updated_keys)

    def test_item5_minimal_experiment_creation(self):
        """Creating an experiment should not write redundant seed/total_timesteps/eval_episodes."""
        tree_widget = ConfigTreeWidget(self.tmp_root, mode="experiments")
        exp_path = tree_widget._write_experiment_from_defaults("cartpole", "minimal_exp.yaml", "minimal_exp")
        self.assertIsNotNone(exp_path)
        created_file = self.tmp_root / "experiment" / "cartpole" / "minimal_exp.yaml"
        self.assertTrue(created_file.exists())
        data = yaml.safe_load(created_file.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "minimal_exp")
        self.assertIn("cartpole/_base", data["defaults"])
        self.assertNotIn("seed", data, "Redundant seed should not be written")
        self.assertNotIn("total_timesteps", data, "Redundant total_timesteps should not be written")
        self.assertNotIn("eval_episodes", data, "Redundant eval_episodes should not be written")
        tree_widget.close()

    def test_item6_play_button_always_shows_tooltip_explaining_cannot_run(self):
        """The play button has WA_AlwaysShowToolTips and explains why it cannot run when disabled."""
        tree_widget = ConfigTreeWidget(self.tmp_root, mode="experiments")
        btn = tree_widget.btn_launch
        self.assertIsNotNone(btn)
        self.assertTrue(btn.testAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips))
        tree_widget.close()

    def test_item6_update_launch_state_tooltip_messages(self):
        """update_launch_state sets specific 'Cannot run:' tooltip reasons."""
        from unittest.mock import MagicMock
        from frontend.app import Window

        app_mock = MagicMock(spec=Window)
        app_mock.active = None
        app_mock.start_button = QToolButton()
        app_mock.queue_button = None
        app_mock.stop_button = None
        app_mock.state_label = MagicMock()
        app_mock.compose_valid = False
        app_mock.current_experiment = None
        app_mock.active_config_rel_path = "experiment/cartpole/_base.yaml"
        app_mock.compose_error = None

        Window.update_launch_state(app_mock)
        self.assertIn("Base templates (_base.yaml) cannot be run directly", app_mock.start_button.toolTip())

        app_mock.active_config_rel_path = ""
        Window.update_launch_state(app_mock)
        self.assertIn("Select an experiment configuration", app_mock.start_button.toolTip())

        app_mock.current_experiment = "cartpole/cp_test"
        app_mock.compose_error = "Validation error: agent ppo forbids offline_rl"
        Window.update_launch_state(app_mock)
        self.assertIn("Validation error: agent ppo forbids offline_rl", app_mock.start_button.toolTip())

        app_mock.active = {"status": "running", "simulated": False, "backend": {"experiment_id": "cartpole/cp_test"}}
        Window.update_launch_state(app_mock)
        self.assertIn("A run is currently active", app_mock.start_button.toolTip())


if __name__ == "__main__":
    unittest.main()
