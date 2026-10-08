"""Unit tests for the frontend QueuePanel component."""
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.queue_panel import QueuePanel, clock
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available in current Python environment")
class TestQueuePanel(unittest.TestCase):

    def setUp(self):
        self.moves = []
        self.removals = []
        self.opened = []
        self.toggled = []
        self.panel = QueuePanel(
            on_move=lambda jid, pos: self.moves.append((jid, pos)),
            on_remove=lambda jid: self.removals.append(jid),
            on_open=lambda jid: self.opened.append(jid),
            on_toggle=lambda run: self.toggled.append(run),
        )

    def test_clock_helper(self):
        self.assertEqual(clock(0), "0:00")
        self.assertEqual(clock(65), "1:05")
        self.assertEqual(clock(None), "0:00")
        self.assertEqual(clock(-10), "0:00")
        self.assertEqual(clock("invalid"), "0:00")

    def test_set_data_with_pending_job_null_started(self):
        """Active job with started=None (e.g. pending status) must not raise TypeError."""
        data = {
            "running": True,
            "active": [{
                "job_id": "job-pending-1",
                "experiment_id": "test_exp_pending",
                "status": "pending",
                "created": 1000.0,
                "started": None,
                "finished": None,
                "total_timesteps": 10000,
                "effective_timesteps": 10240,
            }],
            "queued": [],
            "finished": [],
        }
        self.panel.set_data(data)
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertEqual(self.panel.table.item(0, 0).text(), "▶")
        self.assertEqual(self.panel.table.item(0, 1).text(), "test_exp_pending")
        self.assertEqual(self.panel.table.item(0, 2).text(), "10,240")
        self.assertEqual(self.panel.table.item(0, 3).text(), "pending")
        self.assertTrue(self.panel.table.item(0, 4).text().startswith("starting"))

    def test_set_data_with_null_and_missing_fields(self):
        """Jobs with null/missing experiment_id, timesteps, and timestamps should render cleanly."""
        data = {
            "running": False,
            "active": [{
                "job_id": "job-active-nulls",
                "status": "running",
            }],
            "queued": [{
                "job_id": "job-queued-nulls",
                "position": 0,
            }],
            "finished": [{
                "job_id": "job-finished-nulls",
                "status": "cancelled",
            }],
        }
        self.panel.set_data(data)
        self.assertEqual(self.panel.table.rowCount(), 3)
        # Check active row
        self.assertEqual(self.panel.table.item(0, 1).text(), "job-active-nulls")
        self.assertEqual(self.panel.table.item(0, 2).text(), "—")
        # Check queued row
        self.assertEqual(self.panel.table.item(1, 0).text(), "1")
        self.assertEqual(self.panel.table.item(1, 4).text(), "queued")
        # Check finished row
        self.assertEqual(self.panel.table.item(2, 4).text(), "never started")

    def test_button_actions_and_selection(self):
        data = {
            "running": False,
            "active": [],
            "queued": [
                {"job_id": "q1", "experiment_id": "exp1", "position": 0, "status": "queued"},
                {"job_id": "q2", "experiment_id": "exp2", "position": 1, "status": "queued"},
            ],
            "finished": [],
        }
        self.panel.set_data(data)
        self.panel.table.selectRow(0)
        self.panel.update_buttons()

        self.assertFalse(self.panel.up_button.isEnabled())
        self.assertTrue(self.panel.down_button.isEnabled())
        self.assertTrue(self.panel.remove_button.isEnabled())
        self.assertTrue(self.panel.open_button.isEnabled())

        self.panel.move_selected(1)
        self.assertEqual(self.moves, [("q1", 1)])

        self.panel.remove_selected()
        self.assertEqual(self.removals, ["q1"])

        self.panel.open_selected()
        self.assertEqual(self.opened, ["q1"])

    def test_set_offline(self):
        self.panel.set_offline("Connection refused")
        self.assertFalse(self.panel.run_button.isEnabled())
        self.assertEqual(self.panel.summary.toolTip(), "Connection refused")


if __name__ == "__main__":
    unittest.main()
