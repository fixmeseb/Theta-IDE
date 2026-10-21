"""Unit and integration tests for Theta Hub component lifecycle and file tree synchronization.

Verifies:
1. File Tree Deletion Interception (Option 1): Component discovery and metadata-driven uninstallation.
2. Component configs are hidden on uninstall and restored on reinstall.
3. Dialog instantiation and choice contracts.
"""
import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication, QLabel
app = QApplication.instance() or QApplication([])

from frontend.hub.installer import HubInstaller
from frontend.hub.models import HubComponent
from frontend.hub.dialog import UninstallComponentDialog
from frontend.config_tree import ComponentDeleteDialog


class TestHubLifecycleSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="theta_test_sync_"))
        self.workspace_dir = self.temp_dir / "workspace"
        self.data_dir = self.temp_dir / "data"
        self.workspace_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)

        self.installer = HubInstaller(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
        )

        # Create a mock installed component
        self.comp_dir = self.workspace_dir / "in" / "envs" / "mock_game"
        self.comp_dir.mkdir(parents=True)
        self.config_dir = self.workspace_dir / "in" / "config" / "env" / "mock_game"
        self.config_dir.mkdir(parents=True)

        self.cfg1 = self.config_dir / "level1.yaml"
        self.cfg1.write_text("name: mock_game/level1\n", encoding="utf-8")
        self.cfg2 = self.config_dir / "level2.yaml"
        self.cfg2.write_text("name: mock_game/level2\n", encoding="utf-8")

        self.meta_file = self.comp_dir / ".theta_component.json"
        self.meta_file.write_text(json.dumps({
            "id": "mock-game",
            "name": "Mock Game Suite",
            "kind": "env",
            "version": "1.0.0",
            "config_paths": [
                "in/config/env/mock_game/level1.yaml",
                "in/config/env/mock_game/level2.yaml",
            ]
        }), encoding="utf-8")

        self.component = HubComponent(
            id="mock-game",
            name="Mock Game Suite",
            kind="env",
            version="1.0.0",
            target_path="in/envs/mock_game",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_has_configs_detection(self):
        """has_configs correctly detects presence of config files on disk."""
        self.assertTrue(self.installer.has_configs(self.component))

        # Remove config files
        self.cfg1.unlink()
        self.cfg2.unlink()
        self.assertFalse(self.installer.has_configs(self.component))

    def test_find_component_for_source_path(self):
        """find_component_for_path detects component by source directory."""
        meta = self.installer.find_component_for_path(self.comp_dir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")
        self.assertEqual(meta["name"], "Mock Game Suite")

    def test_find_component_for_config_dir_path(self):
        """find_component_for_path detects component by config directory (Option 1)."""
        meta = self.installer.find_component_for_path(self.config_dir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")

    def test_find_component_for_config_file_path(self):
        """find_component_for_path detects component by individual config file."""
        meta = self.installer.find_component_for_path(self.cfg1)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")

    def test_find_component_returns_none_for_unrelated_path(self):
        """find_component_for_path returns None for regular user folders."""
        other_dir = self.workspace_dir / "in" / "config" / "experiment" / "my_exp"
        other_dir.mkdir(parents=True)
        meta = self.installer.find_component_for_path(other_dir)
        self.assertIsNone(meta)

    def test_uninstall_hides_configs_and_reinstall_restores_them(self):
        """Uninstall hides configs outside the active tree and reinstall restores them."""
        ok = self.installer.uninstall(self.component)
        self.assertTrue(ok)

        # Source code is removed
        self.assertFalse(self.comp_dir.exists())

        # Active configs are hidden, while user data remains in IDE-local storage.
        self.assertFalse(self.cfg1.exists())
        self.assertFalse(self.cfg2.exists())
        self.assertFalse(self.config_dir.exists())
        retained1 = self.data_dir / "retained_configs" / "mock-game" / "in/config/env/mock_game/level1.yaml"
        retained2 = self.data_dir / "retained_configs" / "mock-game" / "in/config/env/mock_game/level2.yaml"
        self.assertTrue(retained1.is_file())
        self.assertTrue(retained2.is_file())

        restored_paths = []
        self.installer._restore_retained_configs("mock-game", restored_paths, {})
        self.assertEqual(set(restored_paths), {
            "in/config/env/mock_game/level1.yaml",
            "in/config/env/mock_game/level2.yaml",
        })
        self.assertTrue(self.cfg1.is_file())
        self.assertTrue(self.cfg2.is_file())

    def test_uninstall_removing_configs(self):
        """Uninstall removes configs from the active tree."""
        ok = self.installer.uninstall(self.component)
        self.assertTrue(ok)

        # Source code is removed
        self.assertFalse(self.comp_dir.exists())

        # Configs and empty parent folder are cleaned up!
        self.assertFalse(self.cfg1.exists())
        self.assertFalse(self.cfg2.exists())
        self.assertFalse(self.config_dir.exists())

    def test_uninstall_by_metadata(self):
        """uninstall_by_metadata cleanly uninstalls when triggered from file tree (Option 1)."""
        meta = self.installer.find_component_for_path(self.config_dir)
        self.assertIsNotNone(meta)

        ok = self.installer.uninstall_by_metadata(meta)
        self.assertTrue(ok)
        self.assertFalse(self.comp_dir.exists())
        self.assertFalse(self.config_dir.exists())

    def test_dialog_instantiation(self):
        """Delete and uninstall dialogs initialize with the new lifecycle text."""
        del_dlg = ComponentDeleteDialog("in/config/env/mock_game", "Mock Game Suite")
        self.assertEqual(del_dlg.choice, "cancel")

        uninst_dlg = UninstallComponentDialog("Mock Game Suite")
        self.assertIn("restored", " ".join(label.text() for label in uninst_dlg.findChildren(QLabel)))


if __name__ == "__main__":
    unittest.main()
