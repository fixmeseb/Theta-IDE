"""Results browser: the Results tab is the Curves view, and runs on disk become run records (PyQt6, offscreen)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QTabWidget
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.model import Config, new_run
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


def entry(experiment_id, started, reward, seconds=60.0, finished=True, group="sweeps"):
    return {"group": group, "experiment_id": experiment_id, "metadata": {}, "finished": finished,
            "config": {"env": "cartpole", "seed": 1, "total_timesteps": 1000,
                       "methods": {"ppo": {"agent": "ppo", "model": "dnn"}}},
            "agents": [{"name": "ppo", "started": started, "finished": started if finished else None,
                        "training_time_seconds": seconds, "timesteps": 1024, "best_reward": reward,
                        "latest_reward": reward}]}


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestResultsBrowser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # A runs folder inside the temp dir keeps this test's settings.toml out of the shared temp folder.
        self.window = Window(Path(self.temp_dir.name) / "runs")

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def listing(self, *entries):
        self.window.disk_runs_received({"runs": list(entries)}, None)

    def cli_ids(self):
        return [r["backend"]["experiment_id"] for r in self.window.runs if r.get("origin") == "cli"]

    def test_the_results_tab_is_the_curves_view_with_no_run_list(self):
        self.assertIs(self.window.results_panel, self.window.results_view)
        self.assertFalse(hasattr(self.window, "table"))
        tabs = [t.tabText(i) for t in self.window.results_panel.findChildren(QTabWidget) for i in range(t.count())]
        self.assertNotIn("Run list", tabs)

    def test_command_line_runs_become_records_newest_first(self):
        self.listing(entry("cli_a", "2026-09-01T10:00:00+00:00", 50.0),
                     entry("cli_b", "2026-09-02T10:00:00+00:00", 80.0))
        self.assertEqual(self.cli_ids(), ["cli_b", "cli_a"])

    def test_app_run_is_linked_not_duplicated(self):
        trained = new_run(Config(name="mine"), simulated=False)
        trained.update(status="completed", backend={"group": "thetaide", "experiment_id": "mine_1"})
        self.window.runs.insert(0, trained)
        self.listing(entry("mine_1", "2026-09-03T10:00:00+00:00", 70.0, seconds=14.76, group="thetaide"))
        self.assertEqual(self.cli_ids(), [])  # the app's own record stands for it
        self.assertEqual([r for r in self.window.runs if (r.get("backend") or {}).get("experiment_id") == "mine_1"], [trained])

    def test_relisting_keeps_records_and_drops_deleted_folders(self):
        self.listing(entry("small", "2026-09-01T10:00:00+00:00", 9.5),
                     entry("big", "2026-09-02T10:00:00+00:00", 100.0))
        record = next(r for r in self.window.runs if r.get("backend", {}).get("experiment_id") == "big")
        record["metrics"] = [{"step": 1024, "reward": 100.0}]
        self.listing(entry("big", "2026-09-02T10:00:00+00:00", 100.0))  # "small" was deleted
        self.assertEqual(self.cli_ids(), ["big"])
        self.assertIs(next(r for r in self.window.runs if r.get("origin") == "cli"), record)  # same record kept


if __name__ == "__main__":
    unittest.main()
