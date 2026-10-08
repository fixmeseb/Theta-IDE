import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window
from frontend.model import Config, new_run, sample


class TestRunNavigator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_run_combo_initial_population(self):
        """Ensure run_combo has all loaded runs and selection matches."""
        self.assertGreaterEqual(self.window.run_combo.count(), 1)
        self.assertEqual(self.window.run_combo.count(), len(self.window.runs))
        current_data = self.window.run_combo.currentData()
        self.assertEqual(current_data, self.window.selected["id"])

    def test_navigation_prev_next(self):
        """Test Prev and Next buttons navigate through runs."""
        # Ensure we have at least 2 runs
        if len(self.window.runs) < 2:
            r2 = new_run(Config(name="extra_run", seed=99))
            self.window.runs.append(r2)
            self.window.refresh_runs()

        total = self.window.run_combo.count()
        self.assertGreaterEqual(total, 2)

        # Select index 0
        self.window.run_combo.setCurrentIndex(0)
        self.assertFalse(self.window.btn_prev_run.isEnabled())
        self.assertTrue(self.window.btn_next_run.isEnabled())

        # Click Next
        self.window.select_next_run()
        self.assertEqual(self.window.run_combo.currentIndex(), 1)
        self.assertEqual(self.window.selected["id"], self.window.runs[1]["id"])
        self.assertTrue(self.window.btn_prev_run.isEnabled())

        # Click Prev
        self.window.select_prev_run()
        self.assertEqual(self.window.run_combo.currentIndex(), 0)
        self.assertEqual(self.window.selected["id"], self.window.runs[0]["id"])

    def test_smoothing_slider(self):
        """Test smoothing slider updates label and chart smoothing factors."""
        slider = self.window.smoothing_slider
        self.assertEqual(slider.value(), 0)
        self.assertIn("0%", self.window.smoothing_label.text())

        slider.setValue(60)
        self.assertIn("60%", self.window.smoothing_label.text())
        self.assertAlmostEqual(self.window.reward_chart.smoothing, 0.6, places=2)
        self.assertAlmostEqual(self.window.loss_chart.smoothing, 0.6, places=2)

    def test_baseline_pinning_and_overlay(self):
        """Test pinning a run overlays its metrics when viewing another run."""
        # Setup run 1 with metrics
        run1 = self.window.runs[0]
        run1["metrics"] = [
            {"step": 0, "reward": 10.0, "entropy": 0.69},
            {"step": 100, "reward": 50.0, "entropy": 0.50},
        ]
        # Create run 2 with metrics
        run2 = new_run(Config(name="test_run_2", seed=101))
        run2["metrics"] = [
            {"step": 0, "reward": 5.0, "entropy": 0.68},
            {"step": 100, "reward": 30.0, "entropy": 0.55},
        ]
        self.window.runs.append(run2)
        self.window.refresh_runs()

        # Select run 1 and pin it
        self.window.select_run(run1)
        self.assertIsNone(self.window.pinned_baseline_run)
        self.window.toggle_pin_baseline()
        self.assertIs(self.window.pinned_baseline_run, run1)
        self.assertTrue(self.window.btn_pin_baseline.isChecked())
        # While on run 1, baseline series is empty (no self-overlay)
        self.assertEqual(len(self.window.reward_chart.baseline_series), 0)

        # Switch to run 2
        self.window.select_run(run2)
        self.assertFalse(self.window.btn_pin_baseline.isChecked())
        # Baseline series should now be present on both charts!
        self.assertEqual(len(self.window.reward_chart.baseline_series), 1)
        self.assertEqual(len(self.window.loss_chart.baseline_series), 1)
        b_name, b_metrics, b_color = self.window.reward_chart.baseline_series[0]
        self.assertIn("baseline", b_name)
        self.assertEqual(len(b_metrics), 2)

        # Unpin baseline
        self.window.select_run(run1)
        self.window.toggle_pin_baseline()
        self.assertIsNone(self.window.pinned_baseline_run)
        self.window.select_run(run2)
        self.assertEqual(len(self.window.reward_chart.baseline_series), 0)

    def test_live_jump_visibility(self):
        """Test 'Jump to Live' chip appears when inspecting historical run during active training."""
        run1 = self.window.runs[0]
        run2 = new_run(Config(name="live_run", seed=77))
        self.window.runs.insert(0, run2)
        self.window.active = run2

        # Select the active run: chip should be hidden
        self.window.select_run(run2)
        self.assertTrue(self.window.btn_live_jump.isHidden())

        # Select the historical run: chip should appear
        self.window.select_run(run1)
        self.assertFalse(self.window.btn_live_jump.isHidden())
        self.assertIn("Jump to live", self.window.btn_live_jump.text())

        # Clicking Jump to Live snaps back to active run
        self.window.jump_to_live_run()
        self.assertEqual(self.window.selected, run2)
        self.assertTrue(self.window.btn_live_jump.isHidden())

    def test_storage_label_shows_persistence(self):
        """Verify storage_label indicates disk logs and jobs.db."""
        self.window.render_run()
        text = self.window.storage_label.text()
        self.assertIn("jobs.db", text)

    def test_telemetry_streaming_updates_monitor(self):
        """Verify telemetry_received streams metrics from ppo_blendrl_human_neural and updates UI."""
        run = new_run(Config(name="tests_cartpole", seed=42), simulated=False)
        run["backend"] = {
            "job_id": "test-job-123",
            "group": "tests",
            "experiment_id": "cartpole_test",
            "agents": ["ppo_blendrl_human_neural"],
            "total_timesteps": 640,
        }
        self.window.runs.insert(0, run)
        self.window.active = run
        self.window.select_run(run)

        telemetry_payload = {
            "job_id": "test-job-123",
            "job": {
                "job_id": "test-job-123",
                "status": "running",
                "log": ["Training in progress..."],
                "log_total": 1,
            },
            "hardware": {"cpu_percent": 15.2, "memory_mb": 210.5},
            "agents": {
                "ppo_blendrl_human_neural": {
                    "source": "results/logs/tests/cartpole_test/ppo_blendrl_human_neural/version_0/metrics.csv",
                    "rows": [
                        {"eval/reward": 78.5, "eval/reward_std": 12.0, "step": 0.0, "transitions": 0.0},
                        {"losses/total_loss": 32.1, "losses/entropy": 0.61, "step": 0.0, "transitions": 512.0},
                    ],
                    "next_byte": 120,
                    "reset": False,
                }
            },
        }

        self.window.telemetry_received(run, telemetry_payload, None)
        self.assertEqual(len(run["metrics"]), 2)
        self.assertEqual(run["metrics"][0]["reward"], 78.5)
        self.assertEqual(self.window.reward_card.value.text(), "78.5")
        self.assertIn("15.2%", self.window.monitor_note.text())


if __name__ == "__main__":
    unittest.main()
