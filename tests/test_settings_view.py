"""Unit tests for the refactored SettingsView and overlay on ASCII Theta."""
import os
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
    from frontend.settings_view import AsciiThetaSplash, SettingsDetailWindow, SettingsView
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSettingsViewStructureAndNavigation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.settings = self.window.settings_view
        self.window.show()
        self.window.tabs.setCurrentWidget(self.settings)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_settings_structure_and_window_bindings(self):
        """All expected tabs, attributes, and window bindings exist."""
        self.assertIsInstance(self.settings, SettingsView)
        self.assertIs(self.window.settings_panel, self.settings)

        # Tab buttons exist in sidebar
        expected_tabs = ["appearance", "core_plugins", "community_plugins", "hotkeys", "backend", "storage"]
        for tid in expected_tabs:
            self.assertIn(tid, self.settings.tab_buttons)
            self.assertIn(tid, self.settings.page_widgets)

        # Window attributes bound
        self.assertIsNotNone(self.window.settings_theme_select)
        self.assertIsNotNone(self.window.settings_action_key_combo)
        self.assertIsNotNone(self.window.settings_backend_url)
        self.assertIsNotNone(self.window.settings_backend_status)
        self.assertIsNotNone(self.window.settings_runs_count_label)
        self.assertIsNotNone(self.window.settings_ascii)
        self.assertIsNotNone(self.window.anim_toggle_btn)
        self.assertEqual(len(self.window.pane_sliders), 10)

    def test_initial_state_shows_ascii_and_not_settings(self):
        """On entering settings, content stack is at index 0 (ASCII Theta)."""
        self.assertIsNone(self.settings.active_tab_id)
        self.assertEqual(self.settings.content_stack.currentIndex(), 0)
        self.assertIs(self.settings.content_stack.currentWidget(), self.settings.ascii_splash)

        # No sidebar tabs should be checked
        for btn in self.settings.tab_buttons.values():
            self.assertFalse(btn.isChecked())

    def test_click_tab_switches_to_settings_window(self):
        """Clicking a tab displays the settings window on top (index 1)."""
        self.settings.tab_buttons["appearance"].click()

        self.assertEqual(self.settings.active_tab_id, "appearance")
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)
        self.assertIs(self.settings.content_stack.currentWidget(), self.settings.settings_window)
        self.assertTrue(self.settings.tab_buttons["appearance"].isChecked())
        self.assertEqual(
            self.settings.settings_window.pages_stack.currentWidget(),
            self.settings.page_widgets["appearance"],
        )
        self.assertIn("Appearance", self.settings.settings_window.title_label.text())

    def test_tab_switching_updates_settings_content(self):
        """Switching between tabs smoothly updates the title and page widget."""
        # Click Core Plugins
        self.settings.tab_buttons["core_plugins"].click()
        self.assertEqual(self.settings.active_tab_id, "core_plugins")
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)
        self.assertIn("Core Plugins", self.settings.settings_window.title_label.text())

        # Click Hotkeys
        self.settings.tab_buttons["hotkeys"].click()
        self.assertEqual(self.settings.active_tab_id, "hotkeys")
        self.assertTrue(self.settings.tab_buttons["hotkeys"].isChecked())
        self.assertFalse(self.settings.tab_buttons["core_plugins"].isChecked())
        self.assertIn("Hotkeys", self.settings.settings_window.title_label.text())

        # Click Community Plugins
        self.settings.tab_buttons["community_plugins"].click()
        self.assertEqual(self.settings.active_tab_id, "community_plugins")
        self.assertIn("Community Plugins", self.settings.settings_window.title_label.text())

    def test_close_button_returns_to_ascii_theta(self):
        """Clicking the close '✕' button returns to ASCII Theta (index 0)."""
        self.settings.tab_buttons["appearance"].click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)

        # Click close
        self.settings.settings_window.btn_close.click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 0)
        self.assertIsNone(self.settings.active_tab_id)
        self.assertFalse(self.settings.tab_buttons["appearance"].isChecked())

    def test_click_active_tab_again_toggles_back_to_ascii(self):
        """Clicking the active tab button toggles back to ASCII Theta."""
        btn = self.settings.tab_buttons["hotkeys"]
        btn.click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)

        # Click again to dismiss
        btn.click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 0)
        self.assertIsNone(self.settings.active_tab_id)
        self.assertFalse(btn.isChecked())

    def test_reset_to_ascii_helper(self):
        """reset_to_ascii() cleanly returns to ASCII Theta."""
        self.settings.tab_buttons["backend"].click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)

        self.settings.reset_to_ascii()
        self.assertEqual(self.settings.content_stack.currentIndex(), 0)
        self.assertIsNone(self.settings.active_tab_id)

    def test_overlay_sizing_per_tab(self):
        """Settings detail card is appropriately sized per tab and has drop shadow."""
        from frontend.settings_view import TAB_SIZES
        self.assertIsNotNone(self.settings.settings_window.shadow)

        # Backend panel sizing
        self.settings.tab_buttons["backend"].click()
        self.assertEqual(self.settings.settings_window.target_size, TAB_SIZES["backend"])
        self.assertEqual(self.settings.settings_window.card.maximumWidth(), TAB_SIZES["backend"][0])

        # Storage panel sizing
        self.settings.tab_buttons["storage"].click()
        self.assertEqual(self.settings.settings_window.target_size, TAB_SIZES["storage"])
        self.assertEqual(self.settings.settings_window.card.maximumWidth(), TAB_SIZES["storage"][0])

        # Core plugins panel sizing
        self.settings.tab_buttons["core_plugins"].click()
        self.assertEqual(self.settings.settings_window.target_size, TAB_SIZES["core_plugins"])
        self.assertEqual(self.settings.settings_window.card.maximumWidth(), TAB_SIZES["core_plugins"][0])

        # Appearance panel sizing
        self.settings.tab_buttons["appearance"].click()
        self.assertEqual(self.settings.settings_window.target_size, TAB_SIZES["appearance"])
        self.assertEqual(self.settings.settings_window.card.maximumWidth(), TAB_SIZES["appearance"][0])

    def test_ascii_sculpture_stays_active_in_overlay(self):
        """ASCII splash remains visible in background when overlay is shown."""
        self.settings.tab_buttons["appearance"].click()
        self.assertTrue(self.settings.settings_window.isVisible())
        self.assertTrue(self.settings.ascii_splash.isVisible())

    def test_click_backdrop_dismisses_overlay(self):
        """Clicking on the translucent backdrop outside the card dismisses the overlay."""
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QMouseEvent

        self.settings.tab_buttons["appearance"].click()
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)

        # Synthesize a click in the outer margin outside the card (to the right and underneath)
        click_x = float(self.settings.settings_window.width() - 10)
        click_y = float(self.settings.settings_window.height() - 10)
        click_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(click_x, click_y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.settings.settings_window.mousePressEvent(click_event)

        # Overlay should now be dismissed back to index 0
        self.assertEqual(self.settings.content_stack.currentIndex(), 0)
        self.assertIsNone(self.settings.active_tab_id)

    def test_no_horizontal_scrollbars_in_settings(self):
        """Settings scroll area policy is ScrollBarAlwaysOff and no horizontal scroll range exists."""
        self.assertEqual(
            self.settings.settings_window.scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        for tab_id in ["appearance", "core_plugins", "community_plugins", "backend", "storage", "animation"]:
            self.settings.tab_buttons[tab_id].click()
            QApplication.processEvents()
            h_bar = self.settings.settings_window.scroll_area.horizontalScrollBar()
            self.assertEqual(h_bar.maximum(), 0)

    def test_splash_removes_thetaide_title_and_header(self):
        """Splash view background contains the ASCII sculpture without ThetaIDE title header."""
        from PyQt6.QtWidgets import QLabel
        labels = self.settings.ascii_splash.findChildren(QLabel)
        for lbl in labels:
            self.assertNotEqual(lbl.text(), "ThetaIDE")
        self.assertIsNotNone(self.settings.ascii_splash.ascii_sculpture)

    def test_animation_settings_button_in_sidebar_bottom_left(self):
        """Animation settings button is located in the settings sidebar at the bottom left."""
        btn = self.settings.anim_settings_btn
        self.assertIsNotNone(btn)
        self.assertEqual(btn.text(), "Animation Settings")
        self.assertIn("animation", self.settings.tab_buttons)
        self.assertIs(self.settings.tab_buttons["animation"], btn)
        # Check it is in sidebar_frame children
        self.assertIn(btn, self.settings.sidebar_frame.findChildren(type(btn)))

    def test_animation_settings_box_docks_bottom_left(self):
        """Clicking animation settings opens the detail box docked in the bottom left."""
        self.settings.anim_settings_btn.click()
        self.assertEqual(self.settings.active_tab_id, "animation")
        self.assertEqual(self.settings.content_stack.currentIndex(), 1)
        self.assertTrue(self.settings.anim_settings_btn.isChecked())

        # Check detail window docking and styling property
        detail_win = self.settings.settings_window
        self.assertEqual(detail_win.dock_position, "bottom_left")
        self.assertEqual(detail_win.outer_layout.alignment(), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        self.assertEqual(detail_win.card.property("dock"), "bottom-left")
        self.assertIn("Animation Settings", detail_win.title_label.text())

        # Switching to another tab restores top-left docking
        self.settings.tab_buttons["appearance"].click()
        self.assertEqual(self.settings.active_tab_id, "appearance")
        self.assertEqual(detail_win.dock_position, "top_left")
        self.assertEqual(detail_win.outer_layout.alignment(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.assertEqual(detail_win.card.property("dock"), "top-left")

    def test_animation_settings_controls_update_parameters(self):
        """Animation settings sliders, preset selection, and reset defaults update live sculpture."""
        self.settings.anim_settings_btn.click()
        target = self.settings.ascii_splash.ascii_sculpture

        # Slider speed
        self.settings.anim_slider_speed.setValue(180)
        self.assertAlmostEqual(target.speed, 1.8)
        self.assertEqual(self.settings.anim_preset_combo.currentText(), "Custom")

        # Preset selection
        self.settings.anim_preset_combo.setCurrentText("Bold & Chunky")
        self.assertAlmostEqual(target.speed, 0.8)
        self.assertAlmostEqual(target.scale, 1.1)
        self.assertAlmostEqual(target.thickness, 1.9)

        # Toggle playback
        self.settings.anim_toggle_active.setChecked(False)
        self.assertTrue(target.paused)
        self.assertEqual(self.settings.anim_status_lbl.text(), "Paused")

        # Reset defaults
        self.settings._reset_anim_defaults()
        self.assertAlmostEqual(target.speed, 1.0)
        self.assertAlmostEqual(target.scale, 1.0)
        self.assertAlmostEqual(target.thickness, 1.0)
        self.assertEqual(self.settings.anim_preset_combo.currentText(), "Default (Balanced)")


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHotkeysSettingsPage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.settings = self.window.settings_view
        self.window.show()
        self.window.tabs.setCurrentWidget(self.settings)
        # Open hotkeys tab
        self.settings.tab_buttons["hotkeys"].click()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_hotkeys_page_components_and_minimal_card_1(self):
        """Card 1 has only the Action Key selector and Terminal Precedence slider."""
        # Kept in Card 1
        self.assertIsNotNone(self.window.settings_action_key_combo)
        self.assertIsNotNone(self.settings.slider_term)

        # Removed from Card 1 (no clutter)
        self.assertFalse(hasattr(self.settings, "hk_enabled_slider"))
        self.assertFalse(hasattr(self.settings, "custom_action_input"))
        self.assertFalse(hasattr(self.settings, "active_action_badge"))
        self.assertFalse(hasattr(self.settings, "leader_timeout_spin"))

        # Card 2 Hotkey Menu components
        self.assertIsNotNone(self.settings.conflict_banner)
        self.assertIsNotNone(self.settings.conflict_banner_label)
        self.assertFalse(self.settings.conflict_banner.isVisible())

        # All canonical panes tracked in Hotkey Menu
        expected_panes = ["components", "config", "monitor", "results", "plots", "tensorboard", "queue", "terminal", "console", "settings"]
        for pid in expected_panes:
            self.assertIn(pid, self.settings.pane_hotkey_edits)
            self.assertIn(pid, self.settings.pane_conflict_labels)

        # Card 3 Menu shortcuts
        self.assertEqual(len(self.settings.menu_shortcut_edits), 9)

    def test_customize_pane_hotkey_and_persistence(self):
        """User can change hotkey combo for any pane and it persists across sessions."""
        edit_components = self.settings.pane_hotkey_edits["components"]
        self.assertEqual(edit_components.get_canonical_combo(), "Action + 1")

        # Change components to Action + C
        edit_components.set_canonical_combo("Action + C")
        edit_components.comboChanged.emit("Action + C")

        # Verify persistence in settings manager
        self.assertEqual(self.window.settings_manager.hotkey_bindings.get("components"), "Action + C")

        # Clear shortcut
        edit_components.set_canonical_combo("")
        edit_components.comboChanged.emit("")
        self.assertNotIn("components", self.window.settings_manager.hotkey_bindings)

    def test_conflict_warning_detection_and_resolution(self):
        """Assigning duplicate combos triggers the conflict banner and row warnings."""
        edit_components = self.settings.pane_hotkey_edits["components"]
        edit_config = self.settings.pane_hotkey_edits["config"]

        # Initially no conflicts
        self.assertFalse(self.settings._validate_hotkey_conflicts())
        self.assertFalse(self.settings.conflict_banner.isVisible())

        # Cause conflict: assign Action + 1 to config as well
        edit_config.set_canonical_combo("Action + 1")
        edit_config.comboChanged.emit("Action + 1")

        # Conflict detected
        self.assertTrue(self.settings.conflict_banner.isVisible())
        self.assertIn("Action + 1", self.settings.conflict_banner_label.text())
        self.assertTrue(self.settings.pane_conflict_labels["components"].isVisible())
        self.assertTrue(self.settings.pane_conflict_labels["config"].isVisible())
        self.assertIn("Conflict", self.settings.pane_conflict_labels["components"].text())

        # Resolve conflict: assign Action + 2 back to config
        edit_config.set_canonical_combo("Action + 2")
        edit_config.comboChanged.emit("Action + 2")

        # Conflict banner and row labels should now be cleared
        self.assertFalse(self.settings.conflict_banner.isVisible())
        self.assertFalse(self.settings.pane_conflict_labels["components"].isVisible())
        self.assertFalse(self.settings.pane_conflict_labels["config"].isVisible())

    def test_reset_pane_mappings_to_default(self):
        """Resetting pane mappings restores default 0-9 bindings and clears conflicts."""
        # Change several hotkeys
        self.settings.pane_hotkey_edits["components"].set_canonical_combo("Action + X")
        self.settings.pane_hotkey_edits["components"].comboChanged.emit("Action + X")

        # Reset
        self.settings._reset_pane_mappings_to_default()

        self.assertEqual(self.settings.pane_hotkey_edits["components"].get_canonical_combo(), "Action + 1")
        self.assertEqual(self.settings.pane_hotkey_edits["settings"].get_canonical_combo(), "Action + 0")
        self.assertEqual(self.window.settings_manager.hotkey_bindings.get("components"), "Action + 1")
        self.assertFalse(self.settings.conflict_banner.isVisible())

    def test_action_key_preset_and_terminal_precedence(self):
        """Action key combo and terminal precedence slider persist changes."""
        # Select Caps Lock preset
        idx_caps = self.window.settings_action_key_combo.findData("caps_lock")
        self.assertGreaterEqual(idx_caps, 0)
        self.window.settings_action_key_combo.setCurrentIndex(idx_caps)
        self.assertEqual(self.window.settings_manager.action_key, "caps_lock")

        # Toggle terminal precedence
        init_term = self.window.settings_manager.hotkeys_terminal_precedence
        self.settings.slider_term.setChecked(not init_term)
        self.assertEqual(self.window.settings_manager.hotkeys_terminal_precedence, not init_term)

    def test_customize_menu_shortcuts(self):
        """User can customize menu action shortcuts and reset them."""
        from PyQt6.QtGui import QKeySequence

        # Change Launch Training from F5 to F6
        edit = self.settings.menu_shortcut_edits["launch_training"]
        edit.setKeySequence(QKeySequence("F6"))
        self.assertEqual(self.window.settings_manager.get("shortcuts", "launch_training"), "F6")
        if "launch_training" in self.window.menu_actions:
            self.assertEqual(self.window.menu_actions["launch_training"].shortcut().toString(), "F6")

        # Reset menu shortcuts
        self.settings._reset_menu_shortcuts_to_default()
        self.assertEqual(self.window.settings_manager.get("shortcuts", "launch_training"), "F5")
        if "launch_training" in self.window.menu_actions:
            self.assertEqual(self.window.menu_actions["launch_training"].shortcut().toString(), "F5")

