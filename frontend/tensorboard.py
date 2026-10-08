"""TensorBoard panel: controls the background TensorBoard server and launches the external dashboard."""
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .theme import theme_color
from .widgets import label


class TensorBoardPanel(QWidget):
    """Native server management card for TensorBoard without Chromium webview overhead."""

    def __init__(self, backend, log, parent=None):
        super().__init__(parent)
        self.backend, self.log = backend, log
        self.url = None
        self.busy = False
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh)

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
        self.server_url_label = label("http://127.0.0.1:6006", "muted")
        status_row.addWidget(self.status_badge)
        status_row.addSpacing(12)
        status_row.addWidget(self.server_url_label)
        status_row.addStretch()
        card_layout.addLayout(status_row)

        self.info_text = label(
            "The backend runs TensorBoard over results/tensorboard/.\n"
            "Runs configured with 'Log to TensorBoard' write event files here.",
            "muted"
        )
        self.info_text.setWordWrap(True)
        card_layout.addWidget(self.info_text)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start server")
        self.start_btn.clicked.connect(lambda: self.refresh(start=True))
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
        self.refresh()

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

    def refresh(self, start=False):
        if self.busy:
            return
        self.busy = True
        self.backend.get("/api/tensorboard", lambda data, error: self.status_received(data, error, start))

    def status_received(self, data, error, start):
        self.busy = False
        if error:
            self.timer.stop()
            self.set_ui_state(running=False, error="Backend unreachable")
            return
        if not data.get("available"):
            self.timer.stop()
            self.set_ui_state(running=False, error="tensorboard is not installed in the backend environment")
            return
        if not data.get("running"):
            self.url = None
            if start:
                self.set_ui_state(starting=True)
                self.backend.post("/api/tensorboard/start", {}, self.start_received)
                return
            self.timer.stop()
            self.set_ui_state(running=False)
            return

        self.url = data.get("url")
        if not data.get("ready"):
            self.set_ui_state(starting=True)
            self.timer.start()
            return

        self.timer.stop()
        self.set_ui_state(running=True)

    def start_received(self, data, error):
        if error:
            self.set_ui_state(running=False, error=str(error))
            return
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.backend.post("/api/tensorboard/stop", {}, lambda data, error: self.status_received(data, error, False))
        self.url = None
        self.set_ui_state(running=False)

    def open_in_browser(self):
        if self.url:
            QDesktopServices.openUrl(QUrl(self.url))
