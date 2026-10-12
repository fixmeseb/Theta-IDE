"""Unit tests for nested subfolder CRUD in Theta-IDE."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.config_model import ConfigTree, Experiment
from frontend.config_tree import ConfigTreeWidget


class TestFolderCRUD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        # Setup standard directory structure
        self.exp_root = self.root / "experiment"
        self.group = self.exp_root / "cartpole"
        self.group.mkdir(parents=True)

        self.base = self.group / "_base.yaml"
        self.base.write_text("paradigm: online_rl\nenv: cartpole\n", encoding="utf-8")

        self.exp1 = self.group / "quick_test.yaml"
        self.exp1.write_text("experiment_id: quick_test\nseed: 42\nbudget: {total_timesteps: 1000}\n", encoding="utf-8")

        self.widget = ConfigTreeWidget(root_dir=self.root, mode="experiments")

    def test_create_subfolder(self):
        rel, err = self.widget.create_dir("experiment/cartpole", "ablations")
        self.assertIsNone(err)
        self.assertEqual(rel, "experiment/cartpole/ablations")
        self.assertTrue((self.group / "ablations").is_dir())

    def test_create_subfolder_rejects_empty_or_invalid(self):
        rel, err = self.widget.create_dir("experiment/cartpole", "")
        self.assertIsNone(rel)
        self.assertIn("empty", err)

        rel, err = self.widget.create_dir("experiment/cartpole", "invalid/name")
        self.assertIsNone(rel)
        self.assertIn("slashes", err)

    def test_create_subfolder_rejects_existing(self):
        (self.group / "existing").mkdir()
        rel, err = self.widget.create_dir("experiment/cartpole", "existing")
        self.assertIsNone(rel)
        self.assertIn("already exists", err)

    def test_rename_subfolder(self):
        (self.group / "old_folder").mkdir()
        renamed_signals = []
        self.widget.dir_renamed.connect(lambda o, n: renamed_signals.append((o, n)))

        new_rel, err = self.widget.rename_dir("experiment/cartpole/old_folder", "new_folder")
        self.assertIsNone(err)
        self.assertEqual(new_rel, "experiment/cartpole/new_folder")
        self.assertFalse((self.group / "old_folder").exists())
        self.assertTrue((self.group / "new_folder").exists())
        self.assertEqual(renamed_signals, [("experiment/cartpole/old_folder", "experiment/cartpole/new_folder")])

    def test_delete_subfolder(self):
        sub = self.group / "to_delete"
        sub.mkdir()
        (sub / "nested.yaml").write_text("experiment_id: nested\n", encoding="utf-8")

        deleted_signals = []
        self.widget.dir_deleted.connect(lambda rel: deleted_signals.append(rel))

        removed, err = self.widget.delete_dir("experiment/cartpole/to_delete")
        self.assertTrue(removed)
        self.assertIsNone(err)
        self.assertFalse(sub.exists())
        self.assertEqual(deleted_signals, ["experiment/cartpole/to_delete"])

    def test_write_experiment_into_subfolder(self):
        created_rel = self.widget._write_experiment(
            "cartpole", "sweep1.yaml", "sweep1", subpath="sweeps"
        )
        self.assertEqual(created_rel, "experiment/cartpole/sweeps/sweep1.yaml")
        self.assertTrue((self.group / "sweeps" / "sweep1.yaml").exists())

    def test_config_model_discovers_nested_experiments(self):
        sub = self.group / "category" / "subcategory"
        sub.mkdir(parents=True)
        nested_exp = sub / "deep_test.yaml"
        nested_exp.write_text("experiment_id: deep_test\nseed: 123\n", encoding="utf-8")

        tree = ConfigTree(self.root)
        self.assertIn("cartpole/category/subcategory/deep_test", tree.experiments)
        exp = tree.experiments["cartpole/category/subcategory/deep_test"]
        self.assertEqual(exp.group, "cartpole")
        self.assertEqual(exp.name, "cartpole/category/subcategory/deep_test")


if __name__ == "__main__":
    unittest.main()
