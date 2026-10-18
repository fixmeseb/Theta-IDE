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
        self.assertGreater(len(panel.article_items()), 0)

        panel.cleanup()

    def test_toolbar_has_no_emoji(self):
        import unicodedata

        source = Path(DocumentationPanel.__module__.replace(".", "/") + ".py").read_text(encoding="utf-8")
        emoji = [ch for ch in source if unicodedata.category(ch) == "So" and ord(ch) >= 0x2600]
        self.assertEqual(emoji, [])

    def enabled_panel(self):
        """The panel of the enabled plugin, wired to its real PluginContext, as in the app."""
        manager = self.window.plugin_manager
        self.assertTrue(manager.enable_plugin("documentation"))
        self.addCleanup(manager.disable_plugin, "documentation")
        return manager.instances["documentation"].panel

    def test_hotkeys_button_opens_the_hotkeys_settings_page(self):
        panel = self.enabled_panel()
        panel.btn_hotkeys.click()
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)
        self.assertEqual(self.window.settings_view.active_tab_id, "hotkeys")

    def test_hub_button_opens_the_component_hub(self):
        from unittest import mock

        from frontend.hub.client import HubClient

        panel = self.enabled_panel()
        with mock.patch.object(HubClient, "fetch_index_async"):  # no network
            panel.btn_hub.click()
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)
        self.assertTrue(self.window.components_panel.is_hub_active)

    def test_code_blocks_render_as_one_box(self):
        """Qt paints a <pre> background per line (stripes); each block becomes a one-cell table."""
        from frontend.plugins.core.documentation.panel import _boxed_code_blocks

        article = get_article_by_id("cli_workflows")
        blocks = article.html_content.count("<pre><code>")
        self.assertGreater(blocks, 0)
        boxed = _boxed_code_blocks(article.html_content)
        self.assertNotIn("<pre><code>", boxed)
        self.assertEqual(boxed.count('<table class="code"'), blocks)

    def test_open_article_follows_a_theme_switch(self):
        from frontend.theme import BUILTINS

        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        panel.show()
        self.window.theme_manager.apply(BUILTINS["Gruvbox Light"])
        app.processEvents()
        light = BUILTINS["Gruvbox Light"]["colors"]
        self.assertIn(light["text"], panel.browser.document().defaultStyleSheet())
        self.assertIn(light["raised"], panel.toc.styleSheet())
        self.window.theme_manager.apply(BUILTINS["Gruvbox Dark"])

    def test_outline_is_categories_articles_and_sections(self):
        from frontend.plugins.core.documentation.content import ARTICLES
        from frontend.plugins.core.documentation.panel import ARTICLE, CATEGORY, SECTION, article_sections

        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        role = Qt.ItemDataRole.UserRole
        categories = [panel.toc.topLevelItem(i) for i in range(panel.toc.topLevelItemCount())]
        self.assertEqual([c.text(0) for c in categories], list(dict.fromkeys(a.category for a in ARTICLES)))
        self.assertTrue(all(c.data(0, role)[0] == CATEGORY and c.isExpanded() for c in categories))
        self.assertEqual(len(panel.article_items()), len(ARTICLES))
        for item in panel.article_items():
            kind, article_id, _ = item.data(0, role)
            self.assertEqual(kind, ARTICLE)
            article = get_article_by_id(article_id)
            sections = [item.child(i) for i in range(item.childCount())]
            self.assertEqual([s.text(0) for s in sections], article_sections(article))
            self.assertTrue(all(s.data(0, role)[0] == SECTION for s in sections))

    def article_item(self, panel, article_id):
        return next(i for i in panel.article_items() if i.data(0, Qt.ItemDataRole.UserRole)[1] == article_id)

    def expanded_articles(self, panel):
        return sorted(i.data(0, Qt.ItemDataRole.UserRole)[1] for i in panel.article_items() if i.isExpanded())

    def test_category_headers_collapse_with_one_click(self):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        header = panel.toc.topLevelItem(0)
        panel._on_outline_clicked(header)
        self.assertFalse(header.isExpanded())
        panel._on_outline_clicked(header)
        self.assertTrue(header.isExpanded())

    def test_opening_an_article_unfolds_it_without_folding_others(self):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        panel._on_outline_clicked(self.article_item(panel, "launching"))
        panel._on_outline_clicked(self.article_item(panel, "cli_workflows"))
        self.assertEqual(self.expanded_articles(panel), ["cli_workflows", "launching", "overview"])
        self.assertEqual(panel.toc.currentItem().data(0, Qt.ItemDataRole.UserRole)[1], "cli_workflows")

    def test_expand_all_and_collapse_all(self):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        categories = [panel.toc.topLevelItem(i) for i in range(panel.toc.topLevelItemCount())]
        panel.btn_collapse_all.click()
        self.assertFalse(any(c.isExpanded() for c in categories))
        self.assertEqual(self.expanded_articles(panel), [])
        panel.btn_expand_all.click()
        self.assertTrue(all(c.isExpanded() for c in categories))
        self.assertEqual(len(self.expanded_articles(panel)), len(panel.article_items()))

    def test_expansion_survives_rebuilding_the_outline(self):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        panel.btn_collapse_all.click()
        panel._on_outline_clicked(self.article_item(panel, "launching"))
        panel.search_input.setText("install")  # still matches the open article, so it stays open
        panel.search_input.setText("")
        self.assertEqual(panel.current_article_id, "launching")
        self.assertEqual(self.expanded_articles(panel), ["launching"])

    def shown_panel(self, width=1400, height=800):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        panel.resize(width, height)
        panel.show()
        app.processEvents()
        return panel

    def test_selection_follows_the_reading_position(self):
        """One indicator: the selected outline entry is the section being read."""
        panel = self.shown_panel()
        panel.select_article("launching")
        app.processEvents()
        bar = panel.browser.verticalScrollBar()
        selected = []
        for value in range(0, bar.maximum() + 1, 10):
            bar.setValue(value)
            current = panel.toc.currentItem().data(0, Qt.ItemDataRole.UserRole)
            if not selected or selected[-1] != current:
                selected.append(current)
        sections = self.article_item(panel, "launching").childCount()
        self.assertEqual(selected[0], ("article", "launching", None))  # above the first section
        self.assertEqual([anchor for _, _, anchor in selected[1:]], [f"sec-{i}" for i in range(sections)])

    def test_clicking_any_section_puts_it_at_the_top(self):
        """Even an article's last sections: the page leaves room after its last line."""
        for width, height in ((1400, 800), (1000, 500)):
            panel = self.shown_panel(width, height)
            for article_id in ("launching", "ui_panels", "results_browser", "cli_workflows"):
                item = self.article_item(panel, article_id)
                for j in range(item.childCount()):
                    section = self.article_item(panel, article_id).child(j)
                    anchor = section.data(0, Qt.ItemDataRole.UserRole)[2]
                    panel._on_outline_clicked(section)
                    app.processEvents()
                    heading_y = dict((name, y) for y, name in panel._section_tops())[anchor]
                    offset = heading_y - panel.browser.verticalScrollBar().value()
                    self.assertTrue(-40 <= offset <= 2, (article_id, anchor, offset))
                    self.assertEqual(panel.toc.currentItem().data(0, Qt.ItemDataRole.UserRole)[2], anchor)

    def test_clicking_a_section_opens_its_article_and_scrolls_to_it(self):
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        panel.resize(1200, 700)
        panel.show()
        app.processEvents()
        item = next(i for i in panel.article_items() if i.data(0, Qt.ItemDataRole.UserRole)[1] == "launching")
        last_section = item.child(item.childCount() - 1)
        panel._on_outline_clicked(last_section)
        app.processEvents()
        self.assertEqual(panel.current_article_id, "launching")
        self.assertGreater(panel.browser.verticalScrollBar().value(), 0)

    def test_page_header_replaces_the_articles_own_title(self):
        from frontend.plugins.core.documentation.panel import TEXT_WIDTH, article_body

        article = get_article_by_id("overview")
        body = article_body(article)
        self.assertNotIn("<h2>", body.split("<h3>")[0])  # the header titles the page; no second headline
        self.assertIn('<h3><a name="sec-0"></a>', body)
        panel = DocumentationPanel(context=None)
        self.addCleanup(panel.deleteLater)
        self.assertEqual(panel.title_lbl.text(), article.title)
        self.assertIn(article.category, panel.badge_cat.text())
        self.assertEqual(panel.browser.lineWrapColumnOrWidth(), TEXT_WIDTH)

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
