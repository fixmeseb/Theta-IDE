"""TensorBoard panel: runs TensorBoard over results/tensorboard/ as a QProcess and opens it in the browser."""
from __future__ import annotations

import importlib.util
import re
import socket
import sys

from PyQt6.QtCore import QObject, QProcess, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from src.app.services import PROJECT_ROOT, TENSORBOARD_DIR
from src.app.services.processes import kill_process_tree

from .theme import theme_color
from .widgets import label

READY_LINE = re.compile(r"TensorBoard \S+ at (http://\S+)")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TensorBoardServer(QObject):
    """One local TensorBoard server. `ready` fires when TensorBoard prints that it is serving."""

    starting = pyqtSignal()
    ready = pyqtSignal(str)  # url
    stopped = pyqtSignal(str)  # error text, or "" when stopped on purpose

    def __init__(self, python: str | None = None, logdir=None, parent=None):
        super().__init__(parent)
        self.python = python or sys.executable
        self.logdir = logdir or TENSORBOARD_DIR
        self.process: QProcess | None = None
        self.url: str | None = None
        self._output = ""
        self._stopping = False

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("tensorboard") is not None

    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self):
        if self.is_running():
            return
        if not self.available():
            self.stopped.emit("tensorboard is not installed in this environment")
            return
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.url, self._output, self._stopping = None, "", False
        self.process = QProcess(self)
        self.process.setProgram(self.python)
        self.process.setArguments(["-m", "tensorboard.main", "--logdir", str(self.logdir), "--host", "127.0.0.1",
                                   "--port", str(_free_port()), "--reload_interval", "5"])
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.starting.emit()
        self.process.start()

    def _on_output(self, *_):
        if self.process is None:
            return
        self._output += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._output = self._output[-4000:]
        if self.url is None and (match := READY_LINE.search(self._output)):
            self.url = match.group(1)
            self.ready.emit(self.url)

    def _on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.stopped.emit(f"Could not start TensorBoard: {self.process.errorString()}")

    def _on_finished(self, *_):
        error = "" if self._stopping else (self._output.strip().splitlines() or ["TensorBoard exited"])[-1]
        self.url = None
        self.process.deleteLater()
        self.process = None
        self.stopped.emit(error)

    def stop(self):
        if not self.is_running():
            return
        self._stopping = True
        kill_process_tree(int(self.process.processId()))
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)


class TensorBoardPanel(QWidget):
    """Native server management card for TensorBoard without Chromium webview overhead."""

    def __init__(self, server: TensorBoardServer, log, parent=None):
        super().__init__(parent)
        self.server, self.log = server, log
        self.url = None
        self.server.starting.connect(lambda: self.set_ui_state(starting=True))
        self.server.ready.connect(self._on_ready)
        self.server.stopped.connect(self._on_stopped)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = label("TensorBoard", "heading")
        subtitle = label(
            "Inspect binary training event logs, computational graphs, and histograms in full fidelity.",
            "muted"
        )
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Main Server Control Card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        status_row = QHBoxLayout()
        self.status_badge = label("● Stopped", "muted")
        self.server_url_label = label("Server offline", "muted")
        status_row.addWidget(self.status_badge)
        status_row.addSpacing(12)
        status_row.addWidget(self.server_url_label)
        status_row.addStretch()
        card_layout.addLayout(status_row)

        self.info_text = label(
            "ThetaIDE runs TensorBoard over results/tensorboard/.\n"
            "Runs configured with 'Log to TensorBoard' write event files here.",
            "muted"
        )
        self.info_text.setWordWrap(True)
        card_layout.addWidget(self.info_text)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start server")
        self.start_btn.clicked.connect(self.server.start)
        self.stop_btn = QPushButton("Stop server")
        self.stop_btn.clicked.connect(self.stop)
        self.browser_btn = QPushButton("Open in browser")
        self.browser_btn.clicked.connect(self.open_in_browser)
        self.browser_btn.setEnabled(False)

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.browser_btn)
        button_row.addStretch()
        card_layout.addLayout(button_row)

        layout.addWidget(card)
        layout.addStretch(1)

        self.set_ui_state(running=False)

    def activate(self):
        """Called when the tab is shown."""
        if self.server.url:
            self.set_ui_state(running=True)
        elif self.server.is_running():
            self.set_ui_state(starting=True)
        elif not self.server.available():
            self.set_ui_state(running=False, error="tensorboard is not installed in this environment")

    def _on_ready(self, url):
        self.url = url
        self.log(f"TensorBoard running at {url}")
        self.set_ui_state(running=True)

    def _on_stopped(self, error):
        self.url = None
        if error:
            self.log(f"TensorBoard stopped: {error}")
        self.set_ui_state(running=False, error=error or None)

    def set_ui_state(self, running=False, starting=False, error=None):
        if starting:
            self.status_badge.setText("● Starting…")
            self.status_badge.setStyleSheet(f"color: {theme_color('comment')}; font-weight: bold;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.browser_btn.setEnabled(False)
        elif running:
            self.status_badge.setText("● Running")
            self.status_badge.setStyleSheet(f"color: {theme_color('primary')}; font-weight: bold;")
            self.server_url_label.setText(self.url or "")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.browser_btn.setEnabled(True)
        else:
            self.status_badge.setText("● Stopped")
            self.status_badge.setStyleSheet(f"color: {theme_color('muted')};")
            self.server_url_label.setText("Server offline" if not error else f"Error: {error}")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.browser_btn.setEnabled(False)

    def stop(self):
        self.server.stop()
        self.url = None
        self.set_ui_state(running=False)

    def open_in_browser(self):
        if self.url:
            QDesktopServices.openUrl(QUrl(self.url))
