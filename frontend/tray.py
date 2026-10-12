"""System tray icon: closing the window while training hides ThetaIDE to the tray so runs keep going.

Training runs are children of the app (QProcess), so they only survive the window closing if the app
itself keeps running. The tray keeps it running, shows what is training, notifies when runs finish, and
offers Show / Stop all training / Quit.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .app_icon import app_icon

FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")


class TrayController(QObject):
    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window = window
        self.icon = QSystemTrayIcon(app_icon(), self)
        self.icon.activated.connect(self._on_activated)

        self.menu = QMenu()
        self.show_action = QAction("Show ThetaIDE", self.menu)
        self.show_action.triggered.connect(self.show_window)
        self.menu.addAction(self.show_action)
        self.menu.addSeparator()
        self.status_action = QAction("No training running", self.menu)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)
        self.stop_action = QAction("Stop all training", self.menu)
        self.stop_action.triggered.connect(self.stop_all)
        self.menu.addAction(self.stop_action)
        self.menu.addSeparator()
        quit_action = QAction("Quit ThetaIDE", self.menu)
        quit_action.triggered.connect(self.window.quit_app)
        self.menu.addAction(quit_action)
        self.icon.setContextMenu(self.menu)

        window.jobs.job_changed.connect(self._on_job_changed)
        window.jobs.queue_changed.connect(lambda _state: self.refresh())
        self.refresh()
        if self.available():
            self.icon.show()

    @staticmethod
    def available() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def keeps_running(self) -> bool:
        """Closing the window should hide it here: something is training, or the queue will start more."""
        jobs = self.window.jobs
        return self.available() and (jobs.is_busy() or (jobs.queue_running and bool(jobs.queue_order)))

    # ── Window ───────────────────────────────────────────────────────────────

    def hide_to_tray(self):
        self.window.hide()
        running = self._running_names()
        what = ", ".join(running) if running else "the job queue"
        self.icon.showMessage("ThetaIDE is still training",
                              f"{what} keeps running in the background. Click the tray icon to reopen ThetaIDE.",
                              QSystemTrayIcon.MessageIcon.Information, 6000)

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _on_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_window()

    def stop_all(self):
        jobs = self.window.jobs
        jobs.pause_queue()
        for job in jobs.active_jobs():
            jobs.cancel(job["job_id"])

    # ── Status ───────────────────────────────────────────────────────────────

    def _running_names(self) -> list[str]:
        return [job["experiment_id"] for job in self.window.jobs.active_jobs()]

    def refresh(self):
        jobs = self.window.jobs
        running = self._running_names()
        queued = len(jobs.queue_order)
        if running:
            status = f"Training: {', '.join(running)}"
        elif queued:
            status = f"{queued} queued" + ("" if jobs.queue_running else " (queue paused)")
        else:
            status = "No training running"
        if running and queued:
            status += f" · {queued} queued"
        self.status_action.setText(status)
        self.stop_action.setEnabled(bool(running) or (jobs.queue_running and bool(queued)))
        self.icon.setToolTip(f"ThetaIDE · {status}")

    def _on_job_changed(self, job):
        self.refresh()
        status = job.get("status")
        if status not in FINAL_STATUSES or self.window.isVisible():
            return
        name = job.get("experiment_id", "A run")
        if status == "completed":
            title, icon = f"{name} finished", QSystemTrayIcon.MessageIcon.Information
            body = "Training completed. Open ThetaIDE to see the results."
        elif status == "cancelled":
            title, icon, body = f"{name} stopped", QSystemTrayIcon.MessageIcon.Information, "The run was stopped."
        else:
            code = job.get("returncode")
            title, icon = f"{name} failed", QSystemTrayIcon.MessageIcon.Warning
            body = f"Training ended with exit code {code}." if code is not None else "Training could not run."
        self.icon.showMessage(title, body, icon, 8000)

    def remove(self):
        """Take the icon out of the tray (the app is quitting)."""
        self.icon.hide()
        QApplication.processEvents()
