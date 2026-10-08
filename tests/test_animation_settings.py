"""Unit tests for the ASCII Theta animation settings, popup dialog, and sliders."""
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
    from frontend.about import AsciiTheta, AnimationSettingsDialog, AboutDialog
    from frontend.settings import SettingsManager
    from frontend.settings_view import AsciiThetaSplash
    from frontend.app import Window
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestAsciiThetaCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.settings_manager = SettingsManager(self.data_dir)
        self.widget = AsciiTheta(settings_manager=self.settings_manager)

    def tearDown(self):
        self.widget.deleteLater()
        self.temp_dir.cleanup()

    def test_default_values(self):
        self.assertAlmostEqual(self.widget.speed, 1.0)
        self.assertAlmostEqual(self.widget.scale, 1.0)
        self.assertAlmostEqual(self.widget.thickness, 1.0)
        self.assertAlmostEqual(self.widget.tilt_factor, 1.0)
        self.assertAlmostEqual(self.widget.distance, 4.8)
        self.assertFalse(self.widget.paused)

    def test_setters_and_persistence(self):
        self.widget.set_speed(1.75)
        self.assertAlmostEqual(self.widget.speed, 1.75)
        self.assertAlmostEqual(self.settings_manager.ascii_speed, 1.75)

        self.widget.set_scale(1.4)
        self.assertAlmostEqual(self.widget.scale, 1.4)
        self.assertAlmostEqual(self.settings_manager.ascii_size, 1.4)

        self.widget.set_thickness(1.6)
        self.assertAlmostEqual(self.widget.thickness, 1.6)
        self.assertAlmostEqual(self.settings_manager.ascii_thickness, 1.6)

        self.widget.set_tilt(1.3)
        self.assertAlmostEqual(self.widget.tilt_factor, 1.3)
        self.assertAlmostEqual(self.settings_manager.ascii_tilt, 1.3)

        self.widget.set_distance(3.6)
        self.assertAlmostEqual(self.widget.distance, 3.6)
        self.assertAlmostEqual(self.settings_manager.ascii_distance, 3.6)

        self.widget.set_paused(True)
        self.assertTrue(self.widget.paused)
        self.assertFalse(self.settings_manager.ascii_animation)

    def test_reset_defaults(self):
        self.widget.set_speed(2.5)
        self.widget.set_scale(1.5)
        self.widget.set_thickness(2.0)
        self.widget.reset_defaults()
        self.assertAlmostEqual(self.widget.speed, 1.0)
        self.assertAlmostEqual(self.widget.scale, 1.0)
        self.assertAlmostEqual(self.widget.thickness, 1.0)
        self.assertAlmostEqual(self.widget.tilt_factor, 1.0)
        self.assertAlmostEqual(self.widget.distance, 4.8)

    def test_frame_rendering_without_errors(self):
        self.widget.setFixedSize(500, 300)
        # Check standard frame
        cells = self.widget.frame()
        self.assertIsInstance(cells, dict)
        self.assertGreater(len(cells), 0)

        # Check frame with custom sliders
        self.widget.set_scale(0.5)
        self.widget.set_thickness(0.4)
        self.widget.set_tilt(2.0)
        cells_custom = self.widget.frame()
        self.assertIsInstance(cells_custom, dict)
        self.assertGreater(len(cells_custom), 0)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestAnimationSettingsDialog(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.settings_manager = SettingsManager(self.data_dir)
        self.theta = AsciiTheta(settings_manager=self.settings_manager)
        self.dialog = AnimationSettingsDialog(target=self.theta, settings_manager=self.settings_manager)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.theta.deleteLater()
        self.temp_dir.cleanup()

    def test_dialog_elements_exist(self):
        self.assertIsNotNone(self.dialog.slider_speed)
        self.assertIsNotNone(self.dialog.slider_size)
        self.assertIsNotNone(self.dialog.slider_thickness)
        self.assertIsNotNone(self.dialog.slider_tilt)
        self.assertIsNotNone(self.dialog.slider_dist)
        self.assertIsNotNone(self.dialog.toggle_active)
        self.assertIsNotNone(self.dialog.preset_combo)
        self.assertIsNotNone(self.dialog.reset_btn)
        self.assertIsNotNone(self.dialog.close_btn)

    def test_speed_slider_updates_target(self):
        self.dialog.slider_speed.setValue(150)
        self.assertAlmostEqual(self.theta.speed, 1.5)
        self.assertIn("1.50x", self.dialog.val_speed.text())
        self.assertEqual(self.dialog.preset_combo.currentText(), "Custom")

    def test_size_slider_updates_target(self):
        self.dialog.slider_size.setValue(135)
        self.assertAlmostEqual(self.theta.scale, 1.35)
        self.assertEqual(self.dialog.val_size.text(), "135%")

    def test_thickness_slider_updates_target(self):
        self.dialog.slider_thickness.setValue(180)
        self.assertAlmostEqual(self.theta.thickness, 1.8)
        self.assertEqual(self.dialog.val_thickness.text(), "180%")

    def test_tilt_slider_updates_target(self):
        self.dialog.slider_tilt.setValue(175)
        self.assertAlmostEqual(self.theta.tilt_factor, 1.75)
        self.assertIn("1.75x", self.dialog.val_tilt.text())

    def test_perspective_slider_updates_target(self):
        self.dialog.slider_dist.setValue(32)
        self.assertAlmostEqual(self.theta.distance, 3.2)
        self.assertEqual(self.dialog.val_dist.text(), "3.2")

    def test_toggle_active_pauses_resumes(self):
        self.dialog.toggle_active.setChecked(False)
        self.assertTrue(self.theta.paused)
        self.assertEqual(self.dialog.status_lbl.text(), "Paused")

        self.dialog.toggle_active.setChecked(True)
        self.assertFalse(self.theta.paused)
        self.assertEqual(self.dialog.status_lbl.text(), "Running")

    def test_preset_selection(self):
        self.dialog.preset_combo.setCurrentText("Bold & Chunky")
        self.assertAlmostEqual(self.theta.speed, 0.8)
        self.assertAlmostEqual(self.theta.scale, 1.1)
        self.assertAlmostEqual(self.theta.thickness, 1.9)
        self.assertEqual(self.dialog.slider_speed.value(), 80)
        self.assertEqual(self.dialog.slider_thickness.value(), 190)

    def test_reset_defaults_button(self):
        self.dialog.slider_speed.setValue(220)
        self.dialog.slider_thickness.setValue(180)
        self.dialog.reset_btn.click()
        self.assertAlmostEqual(self.theta.speed, 1.0)
        self.assertAlmostEqual(self.theta.thickness, 1.0)
        self.assertAlmostEqual(self.theta.scale, 1.0)
        self.assertEqual(self.dialog.slider_speed.value(), 100)
        self.assertEqual(self.dialog.slider_thickness.value(), 100)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSettingsViewAndAboutIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.settings = self.window.settings_view
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_animation_settings_button_on_splash(self):
        btn = self.settings.ascii_splash.anim_settings_btn
        self.assertIsNotNone(btn)
        self.assertEqual(btn.text(), "Animation Settings")
        self.assertIs(self.window.anim_toggle_btn, btn)
        self.assertIs(self.window.anim_settings_btn, btn)

    def test_animation_settings_button_on_about_dialog(self):
        about = AboutDialog()
        self.assertIsNotNone(about.anim_btn)
        self.assertEqual(about.anim_btn.text(), "Animation Settings")
        about.close()
        about.deleteLater()
