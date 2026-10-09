"""PyQt6 QWebEngineView wrapper for xterm.js terminal emulator with PTY integration."""
import json
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QObject, QTimer, QPoint
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QPlainTextEdit, QFrame, QSizePolicy, QMenu, QApplication
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel

from .pty_session import PtySession


HTML_PATH = Path(__file__).resolve().parent / "terminal.html"

# Authentic terminal palettes for built-in themes
NATIVE_TERMINAL_PALETTES = {
    "catppuccin": {
        "background": "#1e1e2e",
        "foreground": "#cdd6f4",
        "cursor": "#f5e0dc",
        "cursorAccent": "#1e1e2e",
        "selectionBackground": "rgba(88, 91, 112, 0.45)",
        "selectionForeground": "#cdd6f4",
        "black": "#45475a",
        "red": "#f38ba8",
        "green": "#a6e3a1",
        "yellow": "#f9e2af",
        "blue": "#89b4fa",
        "magenta": "#f5c2e7",
        "cyan": "#94e2d5",
        "white": "#bac2de",
        "brightBlack": "#585b70",
        "brightRed": "#f38ba8",
        "brightGreen": "#a6e3a1",
        "brightYellow": "#f9e2af",
        "brightBlue": "#89b4fa",
        "brightMagenta": "#cba6f7",
        "brightCyan": "#94e2d5",
        "brightWhite": "#a6adc8",
    },
    "catppuccin macchiato": {
        "background": "#24273a",
        "foreground": "#cad3f5",
        "cursor": "#f4dbd6",
        "cursorAccent": "#24273a",
        "selectionBackground": "rgba(110, 115, 141, 0.45)",
        "selectionForeground": "#cad3f5",
        "black": "#494d64",
        "red": "#ed8796",
        "green": "#a6da95",
        "yellow": "#eed49f",
        "blue": "#8aadf4",
        "magenta": "#f5bde6",
        "cyan": "#8bd5ca",
        "white": "#b8c0e0",
        "brightBlack": "#5b6078",
        "brightRed": "#ed8796",
        "brightGreen": "#a6da95",
        "brightYellow": "#eed49f",
        "brightBlue": "#8aadf4",
        "brightMagenta": "#c6a0f6",
        "brightCyan": "#8bd5ca",
        "brightWhite": "#cad3f5",
    },
    "catppuccin latte": {
        "background": "#eff1f5",
        "foreground": "#4c4f69",
        "cursor": "#dc8a78",
        "cursorAccent": "#eff1f5",
        "selectionBackground": "rgba(172, 176, 190, 0.45)",
        "selectionForeground": "#4c4f69",
        "black": "#5c5f77",
        "red": "#d20f39",
        "green": "#40a02b",
        "yellow": "#df8e1d",
        "blue": "#1e66f5",
        "magenta": "#ea76cb",
        "cyan": "#179299",
        "white": "#acb0be",
        "brightBlack": "#6c6f85",
        "brightRed": "#d20f39",
        "brightGreen": "#40a02b",
        "brightYellow": "#df8e1d",
        "brightBlue": "#1e66f5",
        "brightMagenta": "#8839ef",
        "brightCyan": "#179299",
        "brightWhite": "#bcc0cc",
    },
    "dracula": {
        "background": "#282a36",
        "foreground": "#f8f8f2",
        "cursor": "#f8f8f2",
        "cursorAccent": "#282a36",
        "selectionBackground": "rgba(68, 71, 90, 0.5)",
        "black": "#21222c",
        "red": "#ff5555",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "blue": "#bd93f9",
        "magenta": "#ff79c6",
        "cyan": "#8be9fd",
        "white": "#f8f8f2",
        "brightBlack": "#6272a4",
        "brightRed": "#ff6e6e",
        "brightGreen": "#69ff94",
        "brightYellow": "#ffffa5",
        "brightBlue": "#d6acff",
        "brightMagenta": "#ff92df",
        "brightCyan": "#a4ffff",
        "brightWhite": "#ffffff",
    },
    "nord": {
        "background": "#2e3440",
        "foreground": "#d8dee9",
        "cursor": "#d8dee9",
        "cursorAccent": "#2e3440",
        "selectionBackground": "rgba(76, 86, 106, 0.5)",
        "black": "#3b4252",
        "red": "#bf616a",
        "green": "#a3be8c",
        "yellow": "#ebcb8b",
        "blue": "#81a1c1",
        "magenta": "#b48ead",
        "cyan": "#88c0d0",
        "white": "#e5e9f0",
        "brightBlack": "#4c566a",
        "brightRed": "#bf616a",
        "brightGreen": "#a3be8c",
        "brightYellow": "#ebcb8b",
        "brightBlue": "#81a1c1",
        "brightMagenta": "#b48ead",
        "brightCyan": "#8fbcbb",
        "brightWhite": "#eceff4",
    },
    "gruvbox light": {
        "background": "#fbf1c7",
        "foreground": "#3c3836",
        "cursor": "#9d6000",
        "cursorAccent": "#fbf1c7",
        "selectionBackground": "rgba(213, 196, 161, 0.6)",
        "selectionForeground": "#3c3836",
        "black": "#282828",
        "red": "#cc241d",
        "green": "#79740e",
        "yellow": "#b57614",
        "blue": "#076678",
        "magenta": "#8f3f71",
        "cyan": "#427b58",
        "white": "#7c6f64",
        "brightBlack": "#928374",
        "brightRed": "#9d0006",
        "brightGreen": "#79740e",
        "brightYellow": "#b57614",
        "brightBlue": "#076678",
        "brightMagenta": "#8f3f71",
        "brightCyan": "#427b58",
        "brightWhite": "#3c3836",
    },
    "paper": {
        "background": "#ffffff",
        "foreground": "#202b3b",
        "cursor": "#6246b5",
        "cursorAccent": "#ffffff",
        "selectionBackground": "rgba(226, 231, 239, 0.7)",
        "selectionForeground": "#202b3b",
        "black": "#202b3b",
        "red": "#d32f2f",
        "green": "#287448",
        "yellow": "#c77700",
        "blue": "#176b91",
        "magenta": "#8851c5",
        "cyan": "#00838f",
        "white": "#526176",
        "brightBlack": "#65748a",
        "brightRed": "#c62828",
        "brightGreen": "#2e7d32",
        "brightYellow": "#e65100",
        "brightBlue": "#1565c0",
        "brightMagenta": "#6a1b9a",
        "brightCyan": "#00695c",
        "brightWhite": "#101620",
    },
    "apollo": {
        "background": "#f5f0e6",
        "foreground": "#292722",
        "cursor": "#8e4a32",
        "cursorAccent": "#f5f0e6",
        "selectionBackground": "rgba(217, 200, 169, 0.65)",
        "selectionForeground": "#292722",
        "black": "#292722",
        "red": "#a34832",
        "green": "#637447",
        "yellow": "#9a741e",
        "blue": "#496a8a",
        "magenta": "#765b7a",
        "cyan": "#397d78",
        "white": "#817a6d",
        "brightBlack": "#918a76",
        "brightRed": "#b56d32",
        "brightGreen": "#748754",
        "brightYellow": "#9a741e",
        "brightBlue": "#496a8a",
        "brightMagenta": "#765b7a",
        "brightCyan": "#397d78",
        "brightWhite": "#eae2d3",
    },
    "athena": {
        "background": "#f1f3f0",
        "foreground": "#252b2c",
        "cursor": "#456e72",
        "cursorAccent": "#f1f3f0",
        "selectionBackground": "rgba(203, 218, 217, 0.65)",
        "selectionForeground": "#252b2c",
        "black": "#252b2c",
        "red": "#a34d4a",
        "green": "#55745f",
        "yellow": "#8a741f",
        "blue": "#496a8c",
        "magenta": "#6d617f",
        "cyan": "#397b7d",
        "white": "#737d7d",
        "brightBlack": "#7b8580",
        "brightRed": "#a96e3f",
        "brightGreen": "#658770",
        "brightYellow": "#8a741f",
        "brightBlue": "#496a8c",
        "brightMagenta": "#6d617f",
        "brightCyan": "#397b7d",
        "brightWhite": "#e4e9e7",
    },
    "dionysus": {
        "background": "#211820",
        "foreground": "#e5d8c8",
        "cursor": "#c9946a",
        "cursorAccent": "#211820",
        "selectionBackground": "rgba(74, 48, 64, 0.6)",
        "selectionForeground": "#e5d8c8",
        "black": "#2d2029",
        "red": "#c45a5a",
        "green": "#71875a",
        "yellow": "#c5a05a",
        "blue": "#687a9b",
        "magenta": "#a66a91",
        "cyan": "#65958d",
        "white": "#e5d8c8",
        "brightBlack": "#71656b",
        "brightRed": "#c17a45",
        "brightGreen": "#839d69",
        "brightYellow": "#c5a05a",
        "brightBlue": "#687a9b",
        "brightMagenta": "#a66a91",
        "brightCyan": "#65958d",
        "brightWhite": "#ffffff",
    },
    "ares": {
        "background": "#171514",
        "foreground": "#e4ddd2",
        "cursor": "#c65a45",
        "cursorAccent": "#171514",
        "selectionBackground": "rgba(71, 37, 34, 0.6)",
        "selectionForeground": "#e4ddd2",
        "black": "#211c1a",
        "red": "#d04a3e",
        "green": "#697354",
        "yellow": "#c09a55",
        "blue": "#596b7a",
        "magenta": "#765866",
        "cyan": "#557a78",
        "white": "#e4ddd2",
        "brightBlack": "#6f6861",
        "brightRed": "#c16a3a",
        "brightGreen": "#7b8663",
        "brightYellow": "#c09a55",
        "brightBlue": "#596b7a",
        "brightMagenta": "#765866",
        "brightCyan": "#557a78",
        "brightWhite": "#ffffff",
    },
    "poseidon": {
        "background": "#080f14",
        "foreground": "#d3dedc",
        "cursor": "#5ba8a5",
        "cursorAccent": "#080f14",
        "selectionBackground": "rgba(24, 52, 60, 0.6)",
        "selectionForeground": "#d3dedc",
        "black": "#0d171d",
        "red": "#c95b59",
        "green": "#527d69",
        "yellow": "#b6a263",
        "blue": "#527ba3",
        "magenta": "#716b8c",
        "cyan": "#3f9698",
        "white": "#d3dedc",
        "brightBlack": "#526369",
        "brightRed": "#b77d4e",
        "brightGreen": "#63947e",
        "brightYellow": "#b6a263",
        "brightBlue": "#527ba3",
        "brightMagenta": "#716b8c",
        "brightCyan": "#3f9698",
        "brightWhite": "#ffffff",
    },
}



def _is_light_hex(hex_code: str) -> bool:
    """Determine whether a hex color has light luminance."""
    try:
        c = hex_code.lstrip("#")
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) > 135
    except Exception:
        pass
    return False


class TerminalBridge(QObject):
    """Bridge object exposed to JavaScript via QWebChannel."""

    data_received = pyqtSignal(str)
    theme_received = pyqtSignal(str)
    font_size_received = pyqtSignal(int)
    clear_requested = pyqtSignal()

    def __init__(self, terminal_widget: "TerminalWidget"):
        super().__init__()
        self.terminal_widget = terminal_widget

    @pyqtSlot(str)
    def send_input(self, data: str):
        """Called from xterm.js on user keyboard input."""
        if not self.terminal_widget.pty.is_alive():
            # If the process exited, pressing Enter or newline restarts the session
            if "\r" in data or "\n" in data or data == " ":
                self.terminal_widget.restart_session()
                return
            return
        self.terminal_widget.pty.write(data)

    @pyqtSlot(int, int)
    def resize_pty(self, cols: int, rows: int):
        """Called from xterm.js when container or window resizes."""
        if cols > 0 and rows > 0:
            self.terminal_widget._current_cols = cols
            self.terminal_widget._current_rows = rows
        self.terminal_widget.pty.resize(cols, rows)

    @pyqtSlot(int, int, str)
    def show_context_menu(self, x: int, y: int, selected_text: str = ""):
        """Called from JavaScript on right-click."""
        self.terminal_widget.show_custom_context_menu(x, y, selected_text)

    @pyqtSlot(str)
    def set_clipboard(self, text: str):
        """Called from JavaScript when tmux or an inner program sets the clipboard via OSC 52."""
        if text:
            QApplication.clipboard().setText(text)

    @pyqtSlot()
    def terminal_ready(self):
        """Called when xterm.js is initialized and mounted in DOM."""
        self.terminal_widget._on_terminal_ready()


class TerminalWidget(QWidget):
    """Interactive xterm.js terminal widget backed by an OS pseudo-terminal."""

    session_started = pyqtSignal()
    session_exited = pyqtSignal(int)

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None,
                 command: Optional[list[str]] = None, env: Optional[dict[str, str]] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())
        self.pty = PtySession(cwd=self.cwd, command=command, parent=self)
        self.pty.extra_env = dict(env or {})
        self.bridge = TerminalBridge(self)
        self._is_ready = False
        self._pending_theme: Optional[dict] = None
        self._current_cols = 80
        self._current_rows = 24

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.web_view = QWebEngineView(self)
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Configure WebEngine settings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, False)

        # Attach WebChannel
        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout.addWidget(self.web_view)

        # Connect PTY signals
        self.pty.data_ready.connect(self._on_pty_data)
        self.pty.process_exited.connect(self._on_pty_exit)

        # Load terminal HTML
        self.web_view.load(QUrl.fromLocalFile(str(HTML_PATH)))

    def _on_terminal_ready(self):
        """xterm.js has loaded, mounted, and reported ready."""
        self._is_ready = True
        if not self.pty.is_alive():
            self.pty.start(cols=self._current_cols, rows=self._current_rows)
            self.session_started.emit()

        if self._pending_theme:
            self.apply_theme(self._pending_theme)
            self._pending_theme = None

        if hasattr(self, "_font_size") and self._font_size:
            self.bridge.font_size_received.emit(self._font_size)

        # Only claim focus if the terminal is currently visible to the user
        if self.isVisible():
            self.focus_terminal()

    def _on_pty_data(self, text: str):
        """Send raw data from PTY process to xterm.js."""
        self.bridge.data_received.emit(text)

    def _on_pty_exit(self, exit_code: int):
        """Child shell exited; notify xterm and signal."""
        msg = f"\r\n\x1b[90m[Process completed with exit code {exit_code} — Press '+ New shell' or Enter to restart]\x1b[0m\r\n"
        self.bridge.data_received.emit(msg)
        self.session_exited.emit(exit_code)

    def send_command(self, cmd: str):
        """Write a command string followed by Enter to the shell."""
        if not self.pty.is_alive():
            self.restart_session()
        self.pty.write(f"{cmd}\n")
        self.focus_terminal()

    def restart_session(self):
        """Terminate current process and start a fresh shell session."""
        self.pty.close()
        self.bridge.clear_requested.emit()
        self.pty.start(cols=self._current_cols, rows=self._current_rows)
        self.session_started.emit()
        self.fit_terminal()
        self.focus_terminal()

    def configure(self, command: Optional[list[str]] = None, cwd: Optional[str] = None,
                  env: Optional[dict[str, str]] = None):
        """Set the shell, start folder and extra environment. Applies from the next shell
        started (New Shell / restart); the running one is left alone."""
        self.pty.command = command
        if cwd:
            self.cwd = cwd
            self.pty.cwd = cwd
        self.pty.extra_env = dict(env or {})

    def clear(self):
        """Clear and reset xterm terminal viewport."""
        self.bridge.clear_requested.emit()

    def fit_terminal(self):
        """Trigger fitAddon.fit() in xterm.js."""
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof fitTerminal === 'function') fitTerminal();")

    def set_font_size(self, size: int):
        """Update terminal font size and refit."""
        self._font_size = size
        if self._is_ready:
            self.bridge.font_size_received.emit(size)

    def focus_terminal(self):
        """Focus keyboard events on the terminal."""
        self.web_view.setFocus()
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.focus();")

    def showEvent(self, event):
        """Handle showing after tab switch to ensure layout coordinates are settled."""
        super().showEvent(event)
        QTimer.singleShot(60, self.fit_terminal)
        QTimer.singleShot(60, self.focus_terminal)

    def blur_terminal(self):
        """Blur terminal to trigger DEC 1004 FocusLost in tmux/vim."""
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.blur();")

    def hideEvent(self, event):
        super().hideEvent(event)
        self.blur_terminal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.fit_terminal)

    def show_custom_context_menu(self, x: int, y: int, selected_text: str = ""):
        """Display native context menu with standard terminal actions."""
        menu = QMenu(self)

        copy_act = QAction("Copy", menu)
        copy_act.setEnabled(bool(selected_text.strip()))
        def do_copy():
            if selected_text:
                QApplication.clipboard().setText(selected_text)
        copy_act.triggered.connect(do_copy)
        menu.addAction(copy_act)

        paste_act = QAction("Paste", menu)
        clip_text = QApplication.clipboard().text()
        paste_act.setEnabled(bool(clip_text))
        def do_paste():
            text = QApplication.clipboard().text()
            if text:
                if not self.pty.is_alive():
                    self.restart_session()
                self.pty.write(text)
        paste_act.triggered.connect(do_paste)
        menu.addAction(paste_act)

        select_all_act = QAction("Select All", menu)
        select_all_act.triggered.connect(lambda: self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.selectAll();"))
        menu.addAction(select_all_act)

        menu.addSeparator()

        clear_act = QAction("Clear Terminal", menu)
        clear_act.triggered.connect(self.clear)
        menu.addAction(clear_act)

        restart_act = QAction("Restart Shell", menu)
        restart_act.triggered.connect(self.restart_session)
        menu.addAction(restart_act)

        global_pos = self.web_view.mapToGlobal(QPoint(x, y))
        menu.exec(global_pos)

    def resolve_theme_palette(self, theme_dict: dict) -> dict:
        """Map Theta-IDE palette to xterm.js theme object."""
        name_key = (theme_dict.get("name") or "").strip().casefold()
        if name_key in NATIVE_TERMINAL_PALETTES:
            return NATIVE_TERMINAL_PALETTES[name_key]
        elif "catppuccin" in name_key and "latte" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin latte"]
        elif "catppuccin" in name_key and "mocha" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin"]
        elif "catppuccin" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin macchiato"]
        elif "dracula" in name_key:
            return NATIVE_TERMINAL_PALETTES["dracula"]
        elif "nord" in name_key:
            return NATIVE_TERMINAL_PALETTES["nord"]
        elif "gruvbox" in name_key and "light" in name_key:
            return NATIVE_TERMINAL_PALETTES["gruvbox light"]
        elif "paper" in name_key:
            return NATIVE_TERMINAL_PALETTES["paper"]
        elif "apollo" in name_key:
            return NATIVE_TERMINAL_PALETTES["apollo"]
        elif "athena" in name_key:
            return NATIVE_TERMINAL_PALETTES["athena"]
        elif "ares" in name_key:
            return NATIVE_TERMINAL_PALETTES["ares"]
        elif "dionysus" in name_key:
            return NATIVE_TERMINAL_PALETTES["dionysus"]
        elif "poseidon" in name_key:
            return NATIVE_TERMINAL_PALETTES["poseidon"]

        colors = theme_dict.get("colors", {})
        if not colors:
            return {}

        base = colors.get("base", "#1d2021")
        text = colors.get("text", "#ebdbb2")
        accent = colors.get("accent", "#fabd2f")
        primary = colors.get("primary", "#b8bb26")
        secondary = colors.get("secondary", "#83a598")
        number = colors.get("number", "#d3869b")
        comment = colors.get("comment", "#928374")
        border = colors.get("border", "#504945")

        is_light = _is_light_hex(base)
        if is_light:
            return {
                "background": base,
                "foreground": text,
                "cursor": accent,
                "cursorAccent": base,
                "selectionBackground": border,
                "black": text,
                "red": "#cc241d",
                "green": primary,
                "yellow": "#b57614",
                "blue": secondary,
                "magenta": number,
                "cyan": "#427b58",
                "white": "#7c6f64",
                "brightBlack": comment,
                "brightRed": "#9d0006",
                "brightGreen": primary,
                "brightYellow": "#b57614",
                "brightBlue": secondary,
                "brightMagenta": number,
                "brightCyan": "#427b58",
                "brightWhite": text,
            }
        return {
            "background": base,
            "foreground": text,
            "cursor": accent,
            "cursorAccent": base,
            "selectionBackground": border,
            "black": base,
            "red": "#ea6962",
            "green": primary,
            "yellow": accent,
            "blue": secondary,
            "magenta": number,
            "cyan": "#8ec07c",
            "white": text,
            "brightBlack": comment,
            "brightRed": "#fb4934",
            "brightGreen": primary,
            "brightYellow": accent,
            "brightBlue": secondary,
            "brightMagenta": number,
            "brightCyan": "#8ec07c",
            "brightWhite": "#fbf1c7",
        }

    def apply_theme(self, theme_dict: dict):
        """Map Theta-IDE palette to xterm.js theme object and update web view."""
        xterm_theme = self.resolve_theme_palette(theme_dict)
        if not xterm_theme:
            return
        self._current_terminal_theme = xterm_theme
        if not self._is_ready:
            self._pending_theme = theme_dict
            return
        self.bridge.theme_received.emit(json.dumps(xterm_theme))

    @property
    def current_terminal_theme(self) -> dict:
        return getattr(self, "_current_terminal_theme", {})

    def close(self):
        self.pty.close()
        super().close()


class TerminalPanel(QWidget):
    """Pure, modern interactive terminal pane with compact toolbar controls."""

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None,
                 command: Optional[list[str]] = None, env: Optional[dict[str, str]] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Compact top control header
        self.toolbar = QFrame(self)
        self.toolbar.setObjectName("terminalToolbar")
        self.toolbar.setFixedHeight(28)
        tb_layout = QHBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)
        tb_layout.setSpacing(8)

        self.status_dot = QLabel("●", self)
        self.status_dot.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        tb_layout.addWidget(self.status_dot)

        self.title_label = QLabel("Terminal", self)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 11px; opacity: 0.85;")
        tb_layout.addWidget(self.title_label)

        tb_layout.addStretch()

        self.restart_btn = QPushButton("+ New Shell", self)
        self.restart_btn.setToolTip("Start a fresh shell session")
        self.restart_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.restart_btn.clicked.connect(self._restart_shell)
        tb_layout.addWidget(self.restart_btn)

        self.clear_btn = QPushButton("Clear", self)
        self.clear_btn.setToolTip("Clear and reset terminal output")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_terminal)
        tb_layout.addWidget(self.clear_btn)

        root_layout.addWidget(self.toolbar)

        # Full-bleed, edge-to-edge interactive terminal
        self.terminal = TerminalWidget(cwd=self.cwd, parent=self, command=command, env=env)
        root_layout.addWidget(self.terminal, 1)

        self.terminal.session_started.connect(self._on_session_started)
        self.terminal.session_exited.connect(self._on_session_exited)

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(2000)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start()

    def _update_status(self):
        if not self.terminal.pty.is_alive():
            return
        if self.terminal.pty.is_tmux_active():
            self.title_label.setText("Terminal · tmux (Action: C-b)")
            self.status_dot.setStyleSheet("color: #89b4fa; font-size: 11px;")
        else:
            sh_path = os.environ.get("SHELL", "shell")
            shell_name = Path(sh_path).name or "shell"
            self.title_label.setText(f"Terminal · {shell_name}")
            self.status_dot.setStyleSheet("color: #a6e3a1; font-size: 11px;")

    def _on_session_started(self):
        self._update_status()

    def _on_session_exited(self, code: int):
        self.title_label.setText(f"Terminal · Exited ({code})")
        self.status_dot.setStyleSheet("color: #f38ba8; font-size: 11px;")

    def _restart_shell(self):
        self.terminal.restart_session()

    def _clear_terminal(self):
        self.terminal.clear()

    def set_font_size(self, size: int):
        """Update font size of underlying terminal."""
        self.terminal.set_font_size(size)

    def apply_theme(self, theme_dict: dict):
        """Apply active theme palette to terminal and toolbar."""
        self.terminal.apply_theme(theme_dict)
        colors = theme_dict.get("colors", {})
        if not colors:
            return
        panel_bg = colors.get("surface", colors.get("panel", "#282828"))
        text_color = colors.get("text", "#ebdbb2")
        border_color = colors.get("border", "#504945")
        btn_bg = colors.get("raised", "#3c3836")
        accent = colors.get("accent", "#fabd2f")

        self.toolbar.setStyleSheet(f"""
            QFrame#terminalToolbar {{
                background-color: {panel_bg};
                border-bottom: 1px solid {border_color};
            }}
            QLabel {{
                color: {text_color};
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                border-color: {accent};
            }}
        """)

    def closeEvent(self, event):
        self.terminal.close()
        super().closeEvent(event)
