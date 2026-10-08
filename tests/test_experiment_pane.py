"""Unit tests for the overhauled Experiment Pane in Theta-IDE:
- ConfigTreeWidget mirroring in/config/
- Boxed ConfigViewer with real-time updates and saving
- On-demand Hydra YAML preview (hidden by default)
- Experiment creation and duplication
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6 import QtWebEngineWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QTreeWidget
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.components_panel import ComponentsPanel
    from frontend.config_tree import ConfigTreeWidget
    from frontend.config_viewer import ConfigBox, ConfigViewer
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigTree(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Create mock in/config structure
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "agent").mkdir(parents=True)
        (self.root / "env").mkdir(parents=True)

        (self.root / "experiment" / "cartpole" / "exp1.yaml").write_text(
            "experiment_id: exp1\nseed: 42\ntotal_timesteps: 5000\n", encoding="utf-8"
        )
        (self.root / "agent" / "ppo.yaml").write_text(
            "lr: 0.0003\nbatch_size: 64\n", encoding="utf-8"
        )
        (self.root / "config.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")

        self.tree_widget = ConfigTreeWidget(root_dir=self.root)

    def tearDown(self):
        self.tree_widget.close()
        self.temp_dir.cleanup()

    def test_tree_populates_directories_and_files(self):
        # Top-level should have experiment, agent, env, and config.yaml
        top_count = self.tree_widget.tree.topLevelItemCount()
        self.assertGreaterEqual(top_count, 4)

        top_texts = [self.tree_widget.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertTrue(any("experiment" in t for t in top_texts))
        self.assertTrue(any("agent" in t for t in top_texts))
        self.assertTrue(any("config" in t for t in top_texts))

    def test_experiment_expanded_by_default(self):
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "experiment" in item.text(0):
                self.assertTrue(item.isExpanded())

    def test_select_file_emits_signal(self):
        selected = []
        self.tree_widget.file_selected.connect(lambda path, rel: selected.append((path, rel)))

        ok = self.tree_widget.select_file("experiment/cartpole/exp1.yaml")
        self.assertTrue(ok)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1], "experiment/cartpole/exp1.yaml")

    def test_filter_tree(self):
        self.tree_widget.filter_tree("exp1")
        # exp1 item should be visible
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "agent" in item.text(0):
                self.assertTrue(item.isHidden())

        self.tree_widget.filter_tree("")
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            self.assertFalse(item.isHidden())

    def test_experiments_mode_shows_groups_directly_without_experiment_folder(self):
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            top_count = exp_tree.tree.topLevelItemCount()
            self.assertEqual(top_count, 1)
            item = exp_tree.tree.topLevelItem(0)
            self.assertEqual(item.text(0), "cartpole")
            top_texts = [exp_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
            self.assertFalse(any(t == "experiment" for t in top_texts))
            self.assertFalse(any("agent" in t for t in top_texts))
            self.assertFalse(any("env" in t for t in top_texts))
            self.assertFalse(any(t == "config" for t in top_texts))
        finally:
            exp_tree.close()

    def test_base_yaml_displayed_as_group_defaults_with_gear_icon(self):
        (self.root / "experiment" / "cartpole" / "_base.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            cartpole_item = exp_tree.tree.topLevelItem(0)
            self.assertEqual(cartpole_item.text(0), "cartpole")
            child_texts = [cartpole_item.child(i).text(0) for i in range(cartpole_item.childCount())]
            self.assertIn("group defaults", child_texts)
            self.assertFalse(any("_base" in t for t in child_texts))

            defaults_item = next(
                cartpole_item.child(i)
                for i in range(cartpole_item.childCount())
                if cartpole_item.child(i).text(0) == "group defaults"
            )
            self.assertFalse(defaults_item.icon(0).isNull(), "group defaults must have a gear icon")
        finally:
            exp_tree.close()

    def test_components_mode_excludes_experiment_directory(self):
        comp_tree = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            top_count = comp_tree.tree.topLevelItemCount()
            top_texts = [comp_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
            self.assertTrue(any("agent" in t for t in top_texts))
            self.assertTrue(any("env" in t for t in top_texts))
            self.assertTrue(any(t == "config" for t in top_texts))
            self.assertFalse(any("experiment" in t for t in top_texts))
        finally:
            comp_tree.close()

    def test_yaml_files_have_no_icon_and_no_yaml_extension_in_tree(self):
        """YAML files must not display the .yaml extension and must not have a file icon."""
        def find_file_items(parent):
            items = []
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                child = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "file":
                    items.append(child)
                items.extend(find_file_items(child))
            return items

        file_items = find_file_items(self.tree_widget.tree)
        self.assertGreater(len(file_items), 0)
        for item in file_items:
            # No .yaml or .yml extension in displayed text
            self.assertFalse(item.text(0).endswith(".yaml"))
            self.assertFalse(item.text(0).endswith(".yml"))
            # No icon / emoji next to yaml files
            self.assertTrue(item.icon(0).isNull())

    def test_directory_nodes_retain_folder_icons(self):
        """Directory nodes in the tree should still display folder icons."""
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get("type") == "dir":
                self.assertFalse(item.icon(0).isNull())

    def test_collapse_and_expand_all(self):
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            # Initially, experiment is expanded
            self.assertTrue(exp_tree.tree.topLevelItem(0).isExpanded())

            # Click collapse all
            exp_tree.btn_collapse.click()
            self.assertFalse(exp_tree.tree.topLevelItem(0).isExpanded())

            # Click expand all
            exp_tree.btn_expand.click()
            self.assertTrue(exp_tree.tree.topLevelItem(0).isExpanded())
        finally:
            exp_tree.close()

    def test_populate_preserves_expanded_folders_on_install_and_uninstall(self):
        """Folders in the tree do not collapse when new components/configs are installed or uninstalled."""
        comp_tree = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            # Locate agent folder item and expand it
            agent_item = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if "agent" in item.text(0):
                    agent_item = item
                    item.setExpanded(True)
                    break
            self.assertIsNotNone(agent_item)
            self.assertTrue(agent_item.isExpanded())

            # Simulate install into agent folder
            (self.root / "agent" / "new_algo.yaml").write_text("algo: new\n", encoding="utf-8")
            comp_tree.populate(ensure_expanded="agent")

            # Check that agent folder did NOT collapse
            found_agent = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if "agent" in item.text(0):
                    found_agent = item
                    break
            self.assertIsNotNone(found_agent)
            self.assertTrue(found_agent.isExpanded(), "Folder must not collapse when something is installed into it")

            # Simulate uninstall from agent folder
            (self.root / "agent" / "new_algo.yaml").unlink()
            comp_tree.populate(ensure_expanded="agent")

            # Check that agent folder did NOT collapse
            found_agent_after = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if "agent" in item.text(0):
                    found_agent_after = item
                    break
            self.assertIsNotNone(found_agent_after)
            self.assertTrue(found_agent_after.isExpanded(), "Folder must not collapse when something is uninstalled from it")
        finally:
            comp_tree.close()


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigViewer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test_exp.yaml"
        self.file_path.write_text(
            "experiment_id: test_exp\n"
            "seed: 123\n"
            "total_timesteps: 25000\n"
            "methods:\n"
            "  ppo:\n"
            "    agent: ppo\n"
            "    model: dnn\n"
            "    lr: 0.0005\n"
            "    batch_size: 128\n"
            "    gamma: 0.98\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.load_file(self.file_path, "experiment/test_exp.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_loads_experiment_values_into_boxes(self):
        self.assertEqual(self.viewer.txt_exp_id.text(), "test_exp")
        self.assertEqual(self.viewer.spin_seed.value(), 123)
        self.assertEqual(self.viewer.spin_timesteps.value(), 25000)

        # Boxes should be created as styled cards
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 3)

    def test_edits_mark_dirty_and_remember_values(self):
        events = []
        self.viewer.config_changed.connect(lambda: events.append(True))

        self.assertFalse(self.viewer.is_dirty)
        self.viewer.spin_seed.setValue(999)

        self.assertTrue(self.viewer.is_dirty)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.viewer.raw_data["seed"], 999)

    def test_save_to_disk(self):
        self.viewer.spin_timesteps.setValue(50000)
        self.assertTrue(self.viewer.is_dirty)

        saved = self.viewer.save_to_disk()
        self.assertTrue(saved)
        self.assertFalse(self.viewer.is_dirty)

        # Verify disk contents
        data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(data["total_timesteps"], 50000)

    def test_get_overrides(self):
        overrides = self.viewer.get_overrides()
        self.assertIn("++experiment_id='test_exp'", overrides)
        self.assertIn("seed=123", overrides)
        self.assertIn("total_timesteps=25000", overrides)

    def test_boxes_do_not_have_titles_or_subtitles(self):
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 3)
        for card in cards:
            self.assertIsNone(card.title_label)
            self.assertIsNone(card.subtitle_label)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestComponentsPanel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "agent").mkdir(parents=True)
        (self.root / "env").mkdir(parents=True)

        (self.root / "agent" / "ppo.yaml").write_text("lr: 0.0003\nbatch_size: 64\n", encoding="utf-8")
        (self.root / "env" / "cartpole.yaml").write_text("name: cartpole\nreward_shaping: false\n", encoding="utf-8")
        (self.root / "config.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")

        self.panel = ComponentsPanel()
        self.panel.components_tree.root_dir = self.root
        self.panel.components_tree.populate()

    def tearDown(self):
        self.panel.close()
        self.temp_dir.cleanup()

    def test_components_panel_tree_only_has_components(self):
        top_count = self.panel.components_tree.tree.topLevelItemCount()
        top_texts = [self.panel.components_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertTrue(any("agent" in t for t in top_texts))
        self.assertFalse(any("experiment" in t for t in top_texts))

    def test_select_component_loads_into_viewer_and_raw_editor(self):
        ok = self.panel.components_tree.select_file("agent/ppo.yaml")
        self.assertTrue(ok)
        self.assertEqual(self.panel.viewer.file_title.text(), "ppo.yaml")
        self.assertIn("lr", self.panel.viewer.raw_data)
        cards = self.panel.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 1)
        for card in cards:
            self.assertIsNone(card.title_label)
            self.assertIsNone(card.subtitle_label)

    def test_toggle_raw_yaml_view(self):
        self.assertTrue(self.panel.raw_panel.isHidden())
        self.panel.toggle_raw_preview(True)
        self.assertFalse(self.panel.raw_panel.isHidden())
        self.panel.toggle_raw_preview(False)
        self.assertTrue(self.panel.raw_panel.isHidden())

    def test_component_hub_replaces_parameters_panels_and_restores_on_selection(self):
        # Initially on parameters view
        self.assertFalse(self.panel.is_hub_active)
        self.assertFalse(self.panel.btn_hub.isChecked())
        self.assertFalse(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

        # Click Component Hub button in top right
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)
        self.assertTrue(self.panel.btn_hub.isChecked())
        self.assertTrue(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Click on a component again in the tree
        self.panel.components_tree.select_file("env/cartpole.yaml")
        # Brought back to parameters view
        self.assertFalse(self.panel.is_hub_active)
        self.assertFalse(self.panel.btn_hub.isChecked())
        self.assertFalse(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)
        self.assertEqual(self.panel.viewer.file_title.text(), "cartpole.yaml")

    def test_component_hub_toggle_and_close_button(self):
        # Open hub
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)

        # Click btn_hub again to toggle back
        self.panel.btn_hub.click()
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

        # Open hub again
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)

        # Click Back to Parameters button in Hub footer
        self.panel.hub_view.btn_close.click()
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

    def test_reload_components_preserves_hub_view_on_install_and_uninstall(self):
        # Open hub
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Simulate install: new component created and reload_components called
        (self.root / "agent" / "new_dqn.yaml").write_text("gamma: 0.99\n", encoding="utf-8")
        self.panel.reload_components(target_rel_path="agent/new_dqn.yaml", ensure_expanded="agent")
        # Must still be in the Hub!
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Simulate uninstall: component deleted and reload_components called
        (self.root / "agent" / "new_dqn.yaml").unlink()
        self.panel.reload_components(target_rel_path=None, ensure_expanded="agent")
        # Must still be in the Hub!
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Only clicking a component in the tree brings you back to params
        self.panel.components_tree.select_file("agent/ppo.yaml")
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestExperimentPaneWindowIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_hydra_yaml_preview_hidden_by_default(self):
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_toggle_hydra_yaml_preview(self):
        # Click toggle to open
        self.window.toggle_yaml_preview(True)
        self.assertFalse(self.window.preview_panel.isHidden())
        self.assertTrue(self.window.btn_toggle_yaml.isChecked())

        # Click toggle to close
        self.window.toggle_yaml_preview(False)
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_selecting_tree_item_loads_into_viewer(self):
        # Select quick_test.yaml
        ok = self.window.config_tree.select_file("experiment/cartpole/quick_test.yaml")
        if ok:
            self.assertEqual(self.window.current_experiment, "cartpole/quick_test")
            self.assertEqual(self.window.config_viewer.file_title.text(), "quick_test.yaml")

    def test_experiment_pane_only_has_experiments(self):
        self.assertEqual(self.window.config_tree.mode, "experiments")
        top_texts = [self.window.config_tree.tree.topLevelItem(i).text(0) for i in range(self.window.config_tree.tree.topLevelItemCount())]
        self.assertFalse(any(t == "experiment" for t in top_texts))
        self.assertFalse(any(t in ("agent", "env", "model", "paradigms", "site") for t in top_texts))
        self.assertTrue(any(t in ("cartpole", "mimic") for t in top_texts))

    def test_components_pane_integrated_in_window(self):
        self.assertIsNotNone(self.window.components_panel)
        self.assertEqual(self.window.components_panel.components_tree.mode, "components")
        self.assertIn("components", self.window.tabs.tabs)
        top_count = self.window.components_panel.components_tree.tree.topLevelItemCount()
        top_texts = [self.window.components_panel.components_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertFalse(any("experiment" in t for t in top_texts))


if __name__ == "__main__":
    unittest.main()
