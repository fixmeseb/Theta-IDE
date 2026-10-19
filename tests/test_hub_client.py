"""Unit and integration tests for Theta HubClient, HubInstaller, and models."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.hub import AuthorInfo, HubClient, HubComponent, HubComponentCard, HubDialog, HubInstaller, ReleaseInfo
    from frontend.theme import theme_color
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHubClientAndInstaller(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="theta_test_hub_"))
        self.workspace_dir = self.temp_dir / "workspace"
        self.data_dir = self.temp_dir / "data"
        self.workspace_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)

        self.client = HubClient(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hub_component_serialization(self):
        data = {
            "id": "test-method",
            "name": "Test RL Method",
            "kind": "method",
            "version": "1.0.0",
            "description": "A test reinforcement learning algorithm.",
            "author": {"name": "Alice", "github": "alice"},
            "repository": "https://github.com/alice/test-method",
            "tags": ["rl", "ppo"],
            "releases": {
                "1.0.0": {
                    "tag": "v1.0.0",
                    "url": "https://example.com/test.zip",
                    "sha256": "abc12345"
                }
            }
        }
        comp = HubComponent.from_dict(data)
        self.assertEqual(comp.id, "test-method")
        self.assertEqual(comp.kind, "method")
        self.assertEqual(comp.author.name, "Alice")
        self.assertEqual(comp.author.github, "alice")
        self.assertIn("1.0.0", comp.releases)
        self.assertEqual(comp.latest_release.tag, "v1.0.0")

        # Roundtrip to dict
        roundtrip = comp.to_dict()
        self.assertEqual(roundtrip["id"], "test-method")
        self.assertEqual(roundtrip["releases"]["1.0.0"]["sha256"], "abc12345")

    def test_search_and_filtering(self):
        c1 = HubComponent.from_dict({
            "id": "cql-continuous", "name": "CQL Continuous", "kind": "method",
            "version": "1.0.0", "description": "Offline RL method", "tags": ["rl", "offline"]
        })
        c2 = HubComponent.from_dict({
            "id": "markdown-notes", "name": "Markdown Notes", "kind": "plugin",
            "version": "0.1.0", "description": "Markdown notebook", "tags": ["notes"]
        })
        c3 = HubComponent.from_dict({
            "id": "neumann-fast", "name": "Neumann Reasoner", "kind": "model",
            "version": "0.2.0", "description": "NeSy model", "tags": ["symbolic"]
        })

        self.client.components = [c1, c2, c3]

        # Kind filter
        plugins = self.client.search(kind="plugin")
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0].id, "markdown-notes")

        # Query filter
        methods = self.client.search(query="cql")
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].id, "cql-continuous")

        # Tag filter
        notes = self.client.search(tag="notes")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].id, "markdown-notes")

    def test_target_directory_resolution_by_kind(self):
        installer = self.client.installer

        plugin_comp = HubComponent(id="my_plugin", name="P", kind="plugin", version="1.0")
        method_comp = HubComponent(id="my_method", name="M", kind="method", version="1.0")
        model_comp = HubComponent(id="my_model", name="Mod", kind="model", version="1.0")
        env_comp = HubComponent(id="my_env", name="E", kind="env", version="1.0")

        self.assertEqual(installer.resolve_target_dir(plugin_comp), self.data_dir / "plugins" / "my_plugin")
        self.assertEqual(installer.resolve_target_dir(method_comp), self.workspace_dir / "src" / "usr" / "algorithms" / "my_method")
        self.assertEqual(installer.resolve_target_dir(model_comp), self.workspace_dir / "src" / "usr" / "models" / "my_model")
        self.assertEqual(installer.resolve_target_dir(env_comp), self.workspace_dir / "in" / "envs" / "my_env")

    def test_download_sha256_verify_and_install(self):
        # 1. Create a dummy package zip
        pkg_src = self.temp_dir / "pkg_src"
        pkg_src.mkdir()
        (pkg_src / "main.py").write_text("# Test component payload", encoding="utf-8")
        (pkg_src / "config.yaml").write_text("learning_rate: 0.001", encoding="utf-8")

        zip_path = self.temp_dir / "sample_package.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.write(pkg_src / "main.py", arcname="main.py")
            zf.write(pkg_src / "config.yaml", arcname="config.yaml")

        # Compute valid SHA-256
        valid_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

        comp = HubComponent(
            id="sample-rl-agent",
            name="Sample RL Agent",
            kind="method",
            version="1.0.0",
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256=valid_sha,
                )
            }
        )

        installer = self.client.installer
        is_installed, ver = installer.check_installed(comp)
        self.assertFalse(is_installed)

        # 2. Test successful install
        success = installer.install(comp, "1.0.0")
        self.assertTrue(success)

        # Verify files on disk
        dest = self.workspace_dir / "src" / "usr" / "algorithms" / "sample-rl-agent"
        deployed_cfg = self.workspace_dir / "in" / "config" / "agent" / "sample-rl-agent.yaml"
        self.assertTrue((dest / "main.py").exists())
        self.assertTrue((dest / "config.yaml").exists())
        self.assertTrue((dest / ".theta_component.json").exists())
        self.assertTrue(deployed_cfg.exists())

        is_installed, ver = installer.check_installed(comp)
        self.assertTrue(is_installed)
        self.assertEqual(ver, "1.0.0")

        # 3. Test uninstall
        uninstalled = installer.uninstall(comp)
        self.assertTrue(uninstalled)
        self.assertFalse(dest.exists())
        self.assertFalse(deployed_cfg.exists())
        is_installed, ver = installer.check_installed(comp)
        self.assertFalse(is_installed)

    def test_checksum_mismatch_fails_safely(self):
        zip_path = self.temp_dir / "corrupted_package.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "hello")

        comp = HubComponent(
            id="corrupted-comp",
            name="Corrupted",
            kind="plugin",
            version="1.0.0",
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",  # Wrong hash!
                )
            }
        )

        installer = self.client.installer
        with self.assertRaises(ValueError) as ctx:
            installer.install(comp, "1.0.0")
        self.assertIn("Checksum verification failed", str(ctx.exception))

        dest = self.data_dir / "plugins" / "corrupted-comp"
        self.assertFalse(dest.exists())

    def test_hub_dialog_ui_initialization(self):
        """HubDialog builds UI and filter pills cleanly."""
        dialog = HubDialog(self.client, mode="components")
        self.assertEqual(dialog.windowTitle(), "Theta Community Hub")
        self.assertIn("method", dialog.pills)
        dialog.close()

        plugins_dialog = HubDialog(self.client, mode="plugins")
        self.assertEqual(plugins_dialog.windowTitle(), "Theta Community Plugins")
        self.assertIn("plugin", plugins_dialog.pills)
        plugins_dialog.close()

    def test_hub_component_card_button_colors_and_state(self):
        """HubComponentCard displays green Install and red Uninstall buttons and transitions cleanly."""
        comp = HubComponent(
            id="test-card-comp",
            name="Card Comp",
            kind="method",
            version="1.0.0",
            is_installed=False,
        )
        card = HubComponentCard(comp, self.client)
        # Uninstalled component has green Install button
        self.assertEqual(card.btn_action.text(), "Install")
        self.assertIn(theme_color("primary"), card.btn_action.styleSheet())  # Install follows the theme
        self.assertEqual(card.btn_action.objectName(), "actionInstall")

        # Simulate install completion
        comp.is_installed = True
        comp.installed_version = "1.0.0"
        card.update_action_state()

        # Installed component has red Uninstall button
        self.assertEqual(card.btn_action.text(), "Uninstall")
        self.assertIn("#cc241d", card.btn_action.styleSheet())
        self.assertEqual(card.btn_action.objectName(), "actionUninstall")

        # Update available has update styling
        comp.version = "2.0.0"
        self.assertTrue(comp.has_update)
        card.update_action_state()
        self.assertIn("Update to v2.0.0", card.btn_action.text())
        self.assertIn(theme_color("secondary"), card.btn_action.styleSheet())  # Update follows the theme

        # Simulate uninstall completion
        comp.is_installed = False
        comp.installed_version = None
        card.update_action_state()
        self.assertEqual(card.btn_action.text(), "Install")
        self.assertIn(theme_color("primary"), card.btn_action.styleSheet())  # Install follows the theme

    def test_thetalib_suite_filtering_and_dialog_display(self):
        """Verify thetalib subcomponents are hidden from search and dialog, while thetalib shows as a plugin."""
        thetalib = HubComponent.from_dict({
            "id": "thetalib",
            "name": "ThetaLib Core Foundation Suite",
            "kind": "suite",
            "version": "1.0.0",
            "description": "Core foundation",
            "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
        })
        std_fig = HubComponent.from_dict({
            "id": "standard-figures",
            "name": "Standard Figures",
            "kind": "plotter",
            "version": "1.0.0",
            "suite": "thetalib",
            "hidden": True,
        })
        img_data = HubComponent.from_dict({
            "id": "image-data",
            "name": "Image Data",
            "kind": "datamodule",
            "version": "1.0.0",
            "suite": "thetalib",
            "hidden": True,
        })
        plugin_comp = HubComponent.from_dict({
            "id": "obsidian-notes",
            "name": "Obsidian Notes",
            "kind": "plugin",
            "version": "0.1.0",
        })

        self.client.components = [thetalib, std_fig, img_data, plugin_comp]

        # 1. Search with include_internal=False (default)
        results = self.client.search()
        result_ids = [c.id for c in results]
        self.assertIn("thetalib", result_ids)
        self.assertIn("obsidian-notes", result_ids)
        self.assertNotIn("standard-figures", result_ids)
        self.assertNotIn("image-data", result_ids)

        # 2. Search with include_internal=True
        all_results = self.client.search(include_internal=True)
        all_ids = [c.id for c in all_results]
        self.assertIn("standard-figures", all_ids)
        self.assertIn("image-data", all_ids)

        # 3. HubDialog filtering for plugins mode
        plugin_dialog = HubDialog(self.client, mode="plugins")
        self.assertTrue(plugin_dialog._matches_mode(thetalib))
        self.assertTrue(plugin_dialog._matches_mode(plugin_comp))
        self.assertFalse(plugin_dialog._matches_mode(std_fig))
        self.assertFalse(plugin_dialog._matches_mode(img_data))
        plugin_dialog.close()

        # 4. HubDialog filtering for components mode
        comp_dialog = HubDialog(self.client, mode="components")
        self.assertTrue(comp_dialog._matches_mode(thetalib))
        self.assertFalse(comp_dialog._matches_mode(plugin_comp))
        self.assertFalse(comp_dialog._matches_mode(std_fig))
        self.assertFalse(comp_dialog._matches_mode(img_data))
        comp_dialog.close()

    def test_thetalib_unified_installation_and_clean_uninstall(self):
        """Verify thetalib installs root-relative files and subcomponent detection works, and uninstalls cleanly."""
        zip_path = self.temp_dir / "thetalib-1.0.0.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr(".theta_component.json", json.dumps({
                "id": "thetalib",
                "name": "ThetaLib Core Foundation Suite",
                "kind": "suite",
                "version": "1.0.0",
                "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
                "installed_dirs": [
                    "plot/standard_figures",
                    "src/usr/data/image_data",
                ],
                "config_paths": [
                    "in/config/plot/thetalib/convergence.yaml",
                    "in/config/data/thetalib/image_data.yaml",
                ]
            }))
            zf.writestr("plot/standard_figures/__init__.py", "")
            zf.writestr("plot/standard_figures/convergence.py", "# Convergence plotter")
            zf.writestr("src/usr/data/image_data/__init__.py", "")
            zf.writestr("src/usr/data/image_data/image_data.py", "# Image data module")
            zf.writestr("in/config/plot/thetalib/convergence.yaml", "title: Convergence\n")
            zf.writestr("in/config/data/thetalib/image_data.yaml", "batch_size: 64\n")

        sha256_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()

        thetalib_comp = HubComponent(
            id="thetalib",
            name="ThetaLib Core Foundation Suite",
            kind="suite",
            version="1.0.0",
            extra={
                "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
                "installed_dirs": ["plot/standard_figures", "src/usr/data/image_data"],
                "config_paths": ["in/config/plot/thetalib/convergence.yaml", "in/config/data/thetalib/image_data.yaml"],
            },
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256=sha256_hash,
                )
            }
        )

        installer = self.client.installer
        is_installed, _ = installer.check_installed(thetalib_comp)
        self.assertFalse(is_installed)

        # Install thetalib
        success = installer.install(thetalib_comp, "1.0.0")
        self.assertTrue(success)

        # Verify files deployed
        conv_py = self.workspace_dir / "plot" / "standard_figures" / "convergence.py"
        img_py = self.workspace_dir / "src" / "usr" / "data" / "image_data" / "image_data.py"
        conv_cfg = self.workspace_dir / "in" / "config" / "plot" / "thetalib" / "convergence.yaml"
        img_cfg = self.workspace_dir / "in" / "config" / "data" / "thetalib" / "image_data.yaml"

        self.assertTrue(conv_py.exists())
        self.assertTrue(img_py.exists())
        self.assertTrue(conv_cfg.exists())
        self.assertTrue(img_cfg.exists())

        # Check installed status
        is_installed, ver = installer.check_installed(thetalib_comp)
        self.assertTrue(is_installed)
        self.assertEqual(ver, "1.0.0")

        # Constituent component lookup and check_installed
        std_fig_comp = HubComponent(id="standard-figures", name="Standard Figures", kind="plotter", version="1.0.0")
        sub_installed, sub_ver = installer.check_installed(std_fig_comp)
        self.assertTrue(sub_installed)
        self.assertEqual(sub_ver, "1.0.0")

        # Uninstall thetalib
        uninstalled = installer.uninstall(thetalib_comp)
        self.assertTrue(uninstalled)

        # Verify all files cleaned up
        self.assertFalse(conv_py.exists())
        self.assertFalse(img_py.exists())
        self.assertFalse(conv_cfg.exists())
        self.assertFalse(img_cfg.exists())

        # Verify empty dirs pruned
        self.assertFalse((self.workspace_dir / "plot" / "standard_figures").exists())
        self.assertFalse((self.workspace_dir / "src" / "usr" / "data" / "image_data").exists())
        self.assertFalse((self.workspace_dir / "in" / "config" / "plot" / "thetalib").exists())
        self.assertFalse((self.workspace_dir / "in" / "config" / "data" / "thetalib").exists())

        is_installed, _ = installer.check_installed(thetalib_comp)
        self.assertFalse(is_installed)


