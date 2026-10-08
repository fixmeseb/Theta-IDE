"""Results browser panel: command-line runs, sorting, columns and filters (PyQt6, offscreen)."""
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
    from frontend.model import RESULT_COLUMNS, Config, new_run
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
        self.window = Window(Path(self.temp_dir.name) / "runs", api_url="http://127.0.0.1:1")

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def column(self, name):
        index = RESULT_COLUMNS.index(name)
        return [self.window.table.item(row, index).text() for row in range(self.window.table.rowCount())]

    def listing(self, *entries):
        self.window.disk_runs_received({"runs": list(entries)}, None)

    def test_offline_shows_examples_and_explains(self):
        self.assertEqual(self.column("Source"), ["Example", "Example"])
        self.assertTrue(self.window.show_examples.isHidden())  # nothing to hide examples from yet
        self.assertIn("Backend offline", self.window.disk_status.text())

    def test_command_line_runs_are_listed_and_examples_hidden(self):
        self.listing(entry("cli_a", "2026-09-01T10:00:00+00:00", 50.0),
                     entry("cli_b", "2026-09-02T10:00:00+00:00", 80.0))
        self.assertEqual(self.column("Experiment"), ["cli_b", "cli_a"])  # newest first by default
        self.assertEqual(set(self.column("Source")), {"Command line"})
        self.assertFalse(self.window.show_examples.isHidden())
        self.window.show_examples.setChecked(True)
        self.assertEqual(self.window.table.rowCount(), 4)

    def test_app_run_is_linked_not_duplicated_and_gets_duration(self):
        trained = new_run(Config(name="mine"), simulated=False)
        trained.update(status="completed", backend={"group": "thetaide", "experiment_id": "mine_1"})
        self.window.runs.insert(0, trained)
        self.listing(entry("mine_1", "2026-09-03T10:00:00+00:00", 70.0, seconds=14.76, group="thetaide"))
        self.assertEqual(self.column("Experiment"), ["mine_1"])
        self.assertEqual(self.column("Source"), ["Trained (app)"])
        self.assertEqual(self.column("Duration"), ["14.8 s"])

    def test_numeric_sort_and_relisting_keeps_records(self):
        self.listing(entry("small", "2026-09-01T10:00:00+00:00", 9.5),
                     entry("big", "2026-09-02T10:00:00+00:00", 100.0),
                     entry("none", "2026-09-03T10:00:00+00:00", None))
        self.window.table.sortByColumn(RESULT_COLUMNS.index("Best reward"), Qt.SortOrder.AscendingOrder)
        self.assertEqual(self.column("Best reward"), ["9.5", "100.0", "—"])  # by value, missing last
        self.window.table.sortByColumn(RESULT_COLUMNS.index("Best reward"), Qt.SortOrder.DescendingOrder)
        self.assertEqual(self.column("Best reward"), ["100.0", "9.5", "—"])
        record = next(r for r in self.window.runs if r.get("backend", {}).get("experiment_id") == "big")
        record["metrics"] = [{"step": 1024, "reward": 100.0}]
        self.listing(entry("big", "2026-09-02T10:00:00+00:00", 100.0))  # "small" and "none" were deleted
        self.assertEqual(self.column("Experiment"), ["big"])
        self.assertIs(next(r for r in self.window.runs if r.get("origin") == "cli"), record)  # same record kept

    def test_status_filter_and_search(self):
        self.listing(entry("done", "2026-09-01T10:00:00+00:00", 1.0),
                     entry("partial", "2026-09-02T10:00:00+00:00", 2.0, finished=False))
        self.window.status_filter.setCurrentIndex(self.window.status_filter.findData("stopped"))
        self.assertEqual(self.column("Experiment"), ["partial"])
        self.window.status_filter.setCurrentIndex(self.window.status_filter.findData("all"))
        self.window.search.setText("command line")
        self.assertEqual(self.window.table.rowCount(), 2)

    def test_selection_survives_refresh(self):
        self.listing(entry("a", "2026-09-01T10:00:00+00:00", 1.0), entry("b", "2026-09-02T10:00:00+00:00", 2.0))
        self.window.table.selectRow(1)
        chosen = self.window.table.item(1, 0).text()
        self.window.refresh_runs()
        (index,) = self.window.table.selectionModel().selectedRows()
        self.assertEqual(self.window.table.item(index.row(), 0).text(), chosen)


if __name__ == "__main__":
    unittest.main()
