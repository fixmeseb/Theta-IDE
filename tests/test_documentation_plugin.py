"""Unit and integration tests for the Documentation Core Plugin."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.plugins.core.documentation import DocumentationPlugin
    from frontend.plugins.core.documentation.content import get_all_articles, get_article_by_id, search_articles
    from frontend.plugins.core.documentation.panel import DocumentationPanel
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestDocumentationPlugin(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_docs_")
        self.window = Window(data_dir=self.temp_dir)

    def tearDown(self):
        self.window.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_documentation_plugin_discovered_as_core(self):
        """Documentation plugin is discovered automatically and flagged as a core plugin."""
        manager = self.window.plugin_manager
        self.assertIn("documentation", manager.manifests)
        manifest = manager.manifests["documentation"]
        self.assertEqual(manifest.id, "documentation")
        self.assertEqual(manifest.name, "Documentation")
        self.assertTrue(manifest.is_core, "Documentation must be recognized as a core plugin")

    def test_documentation_disabled_on_initialization(self):
        """Core plugin starts disabled on clean initialization with zero system overhead."""
        manager = self.window.plugin_manager
        self.assertFalse(manager.is_plugin_enabled("documentation"))
        self.assertNotIn("documentation", manager.instances)
        self.assertNotIn("documentation", self.window.tabs.tabs)
        self.assertNotIn("documentation", self.window.tabs.tab_order)

        if "documentation" in self.window.plugin_sliders:
            self.assertFalse(self.window.plugin_sliders["documentation"].isChecked())

    def test_enable_and_disable_lifecycle(self):
        """Enabling adds the Documentation pane to SideTabs; disabling removes it with zero footprint."""
        manager = self.window.plugin_manager

        # 1. Enable plugin
        success = manager.enable_plugin("documentation")
        self.assertTrue(success)
        self.assertTrue(manager.is_plugin_enabled("documentation"))
        self.assertIn("documentation", manager.instances)
        instance = manager.instances["documentation"]
        self.assertIsInstance(instance, DocumentationPlugin)
        self.assertIsInstance(instance.panel, DocumentationPanel)

        # Tab must be present in SideTabs
        self.assertIn("documentation", self.window.tabs.tabs)
        self.assertIn("documentation", self.window.tabs.tab_order)

        # 2. Disable plugin
        success = manager.disable_plugin("documentation")
        self.assertTrue(success)
        self.assertFalse(manager.is_plugin_enabled("documentation"))
        self.assertNotIn("documentation", manager.instances)

        # Tab must be cleanly unmounted from SideTabs
        self.assertNotIn("documentation", self.window.tabs.tabs)
        self.assertNotIn("documentation", self.window.tabs.tab_order)
        self.assertIsNone(instance.panel)
        self.assertIsNone(instance.context)

    def test_cannot_uninstall_core_plugin(self):
        """Core plugins ship with the IDE and cannot be deleted via uninstall."""
        manager = self.window.plugin_manager
        self.assertIn("documentation", manager.manifests)
        manifest = manager.manifests["documentation"]
        plugin_dir = manifest.plugin_dir

        manager.uninstall_plugin("documentation")

        # Must still exist on disk and in manifests
        self.assertTrue(plugin_dir.exists())
        self.assertIn("documentation", manager.manifests)

    def test_settings_core_plugins_page_display(self):
        """Settings UI displays Documentation under Core Plugins without an uninstall button."""
        self.window.refresh_plugins_ui()
        self.assertTrue(hasattr(self.window, "core_plugins_grid"))
        self.assertIn("documentation", self.window.plugin_sliders)

        slider = self.window.plugin_sliders["documentation"]
        self.assertFalse(slider.isChecked())

        # Toggle on via slider
        slider.click()
        self.assertTrue(slider.isChecked())
        self.assertTrue(self.window.plugin_manager.is_plugin_enabled("documentation"))
        self.assertIn("documentation", self.window.tabs.tabs)

        # Toggle off via slider
        slider.click()
        self.assertFalse(slider.isChecked())
        self.assertFalse(self.window.plugin_manager.is_plugin_enabled("documentation"))
        self.assertNotIn("documentation", self.window.tabs.tabs)

    def test_documentation_content_articles(self):
        """Comprehensive documentation articles cover core IDE and ML concepts."""
        articles = get_all_articles()
        self.assertGreaterEqual(len(articles), 8)

        expected_ids = {
            "overview", "ui_panels", "learning_paradigms",
            "blendrl_hybrid", "hotkeys", "config_system",
            "cluster_slurm", "plugins_guide", "cli_workflows", "troubleshooting"
        }
        actual_ids = {art.id for art in articles}
        self.assertTrue(expected_ids.issubset(actual_ids))

        # Check search functionality
        slurm_results = search_articles("slurm")
        self.assertGreater(len(slurm_results), 0)
        self.assertTrue(any(art.id == "cluster_slurm" for art in slurm_results))

        hydra_results = search_articles("hydra")
        self.assertGreater(len(hydra_results), 0)

    def test_panel_navigation_and_search(self):
        """DocumentationPanel displays selected article and filters list on search input."""
        context = self.window.plugin_manager.contexts.get("documentation")
        panel = DocumentationPanel(context=context)

        # Default article is overview
        self.assertEqual(panel.current_article_id, "overview")
        self.assertIn("Welcome & Architecture Overview", panel.title_lbl.text())
        self.assertIn("Theta-IDE", panel.browser.toPlainText())

        # Select a different article
        panel.select_article("hotkeys")
        self.assertEqual(panel.current_article_id, "hotkeys")
        self.assertIn("Keyboard Shortcuts", panel.title_lbl.text())
        self.assertIn("Ctrl+B", panel.browser.toPlainText())

        # Search filter
        panel.search_input.setText("paradigms")
        self.assertGreater(panel.article_list.count(), 0)

        panel.cleanup()

    def test_state_persistence_in_settings(self):
        """Enabled and disabled states persist across sessions in settings.toml."""
        manager = self.window.plugin_manager

        # Enable and persist
        manager.enable_plugin("documentation")
        self.assertIn("documentation", self.window.settings_manager.plugins_enabled)

        # Reload with new window
        window2 = Window(data_dir=self.temp_dir)
        self.assertTrue(window2.plugin_manager.is_plugin_enabled("documentation"))
        self.assertIn("documentation", window2.tabs.tabs)

        # Disable in window2
        window2.plugin_manager.disable_plugin("documentation")
        self.assertNotIn("documentation", window2.settings_manager.plugins_enabled)
        window2.close()

        # Reload with window3 - must remain disabled
        window3 = Window(data_dir=self.temp_dir)
        self.assertFalse(window3.plugin_manager.is_plugin_enabled("documentation"))
        self.assertNotIn("documentation", window3.tabs.tabs)
        window3.close()


if __name__ == "__main__":
    unittest.main()
