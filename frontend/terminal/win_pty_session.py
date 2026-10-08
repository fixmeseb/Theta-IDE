"""Windows pseudo-console (ConPTY) session, via pywinpty.

Same interface as the POSIX PtySession: start/write/resize/close/is_alive, plus the
data_ready and process_exited signals, so TerminalWidget doesn't care which one it has.

pywinpty's read() blocks, so a daemon thread reads output and hands it to the Qt thread
through a signal (Qt queues signals emitted from another thread).

ConPTY opens with a Primary Device Attributes query (ESC [ c) and holds back all output until
the terminal answers, or for about 3 seconds. xterm.js would answer, but the shell starts while
the page is still loading, so the session answers that first query itself and removes it from
the output (otherwise xterm.js would reply a second time and the shell would read it as input).
"""
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

try:
    from winpty import PtyProcess
    HAS_WINPTY = True
except ImportError:
    PtyProcess = None
    HAS_WINPTY = False

DA1_QUERY = "[c"
DA1_REPLY = "[?1;2c"  # VT100 with advanced video option: what xterm.js reports


def get_default_windows_shell() -> list[str]:
    """PowerShell 7 if installed, else Windows PowerShell, else cmd.exe (always present)."""
    for name, args in (("pwsh.exe", ["-NoLogo"]), ("powershell.exe", ["-NoLogo"])):
        path = shutil.which(name)
        if path:
            return [path, *args]
    comspec = os.environ.get("COMSPEC")
    if comspec and Path(comspec).is_file():
        return [comspec]
    return [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe")]


class WinPtySession(QObject):
    """Manages an interactive shell process bound to a Windows pseudo-console."""

    data_ready = pyqtSignal(str)
    process_exited = pyqtSignal(int)
    _chunk = pyqtSignal(str)  # reader thread -> Qt thread
    _eof = pyqtSignal()

    def __init__(self, cwd: Optional[str] = None, command: Optional[list[str]] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.cwd = str(cwd or Path.cwd())
        self.command = command
        self.process = None  # winpty.PtyProcess
        self.initial_cols = 80
        self.initial_rows = 24
        self.extra_env: dict[str, str] = {}  # merged over os.environ at start (e.g. venv activation)
        self._reader: Optional[threading.Thread] = None
        self._generation = 0  # ignores late output from a session that was closed/restarted
        self._chunk.connect(self.data_ready)
        self._eof.connect(self._handle_exit)

    def start(self, cols: int = 80, rows: int = 24) -> bool:
        """Start the child shell inside a new pseudo-console."""
        if not HAS_WINPTY:
            return False
        self.close()

        self.initial_cols = max(10, cols)
        self.initial_rows = max(4, rows)

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["TERM_PROGRAM"] = "ghostty"
        env.update(self.extra_env)

        try:
            self.process = PtyProcess.spawn(
                self.command or get_default_windows_shell(),
                cwd=self.cwd,
                env=env,
                dimensions=(self.initial_rows, self.initial_cols),
            )
        except Exception:
            self.process = None
            return False

        self._generation += 1
        self._reader = threading.Thread(
            target=self._read_loop, args=(self.process, self._generation), name="winpty-reader", daemon=True
        )
        self._reader.start()
        return True

    def _read_loop(self, process, generation):
        """Runs on the reader thread until the shell's output ends."""
        awaiting_da1 = True
        while True:
            try:
                text = process.read(8192)
            except EOFError:
                break
            except Exception:
                break
            if generation != self._generation:
                return
            if awaiting_da1 and DA1_QUERY in text:
                awaiting_da1 = False
                text = text.replace(DA1_QUERY, "", 1)
                try:
                    process.write(DA1_REPLY)
                except Exception:
                    pass
            if text:
                self._chunk.emit(text)
        if generation == self._generation:
            self._eof.emit()

    def _handle_exit(self):
        """The shell's output ended: report its exit code once."""
        if self.process is None:
            return
        process, self.process = self.process, None
        exit_code = process.exitstatus
        if exit_code is None:
            try:
                exit_code = process.wait()
            except Exception:
                exit_code = 0
        self.process_exited.emit(exit_code or 0)

    def write(self, data: str):
        if self.process is None:
            return
        try:
            self.process.write(data)
        except (EOFError, OSError):
            pass

    def resize(self, cols: int, rows: int):
        if self.process is None:
            return
        try:
            self.process.setwinsize(max(4, rows), max(10, cols))
        except Exception:
            pass

    def close(self):
        """Terminate the shell and stop reading; safe to call repeatedly."""
        self._generation += 1  # the old reader's remaining output and EOF are dropped
        process, self.process = self.process, None
        if process is not None:
            try:
                if process.isalive():
                    process.terminate(force=True)
            except Exception:
                pass
            try:
                process.close(force=True)
            except Exception:
                pass
        self._reader = None

    def is_alive(self) -> bool:
        try:
            return self.process is not None and self.process.isalive()
        except Exception:
            return False

    def is_tmux_active(self) -> bool:
        return False  # no tmux on Windows
