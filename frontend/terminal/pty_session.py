"""Pseudo-terminal (PTY) session management for interactive shells.

PtySession is the backend for this platform: PosixPtySession (pty/termios) on Linux and macOS,
WinPtySession (ConPTY via pywinpty, see win_pty_session.py) on Windows.
"""
import codecs
import errno
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
from typing import Optional

try:
    import fcntl
    import pty
    import termios
    HAS_POSIX_PTY = True
except ImportError:
    fcntl = None
    pty = None
    termios = None
    HAS_POSIX_PTY = False

from PyQt6.QtCore import QObject, QSocketNotifier, pyqtSignal

from .win_pty_session import HAS_WINPTY, WinPtySession, get_default_windows_shell


def get_default_shell() -> list[str]:
    """Resolve the preferred interactive user shell."""
    if sys.platform == "win32":
        return get_default_windows_shell()
    shell_env = os.environ.get("SHELL")
    if shell_env and Path(shell_env).is_file() and os.access(shell_env, os.X_OK):
        return [shell_env, "-l"]

    try:
        import pwd
        pw_shell = pwd.getpwuid(os.getuid()).pw_shell
        if pw_shell and Path(pw_shell).is_file() and os.access(pw_shell, os.X_OK):
            return [pw_shell, "-l"]
    except Exception:
        pass

    for candidate in ["/bin/zsh", "/bin/bash", "/bin/sh"]:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return [candidate, "-l"]

    return ["/bin/sh"]


def available_shells() -> list[tuple[str, str]]:
    """Shells installed on this machine, as (label, command line) for the Settings picker."""
    found = []
    if sys.platform == "win32":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        candidates = [
            ("PowerShell 7", shutil.which("pwsh.exe"), ["-NoLogo"]),
            ("Windows PowerShell", shutil.which("powershell.exe"), ["-NoLogo"]),
            ("Command Prompt", shutil.which("cmd.exe"), []),
            ("Git Bash", str(git_bash) if git_bash.is_file() else None, ["--login", "-i"]),
            ("WSL", shutil.which("wsl.exe"), []),
        ]
        for name, path, args in candidates:
            if path:
                found.append((name, subprocess.list2cmdline([path, *args])))
    else:
        for name in ("zsh", "bash", "fish", "sh"):
            path = shutil.which(name)
            if path:
                found.append((name, shlex.join([path, "-l"])))
    return found


def parse_shell_command(text: str) -> Optional[list[str]]:
    """Split a shell setting into argv; None when empty (auto-detect) or the program isn't found."""
    text = (text or "").strip()
    if not text:
        return None
    if sys.platform == "win32":
        argv = [part.strip('"') for part in shlex.split(text, posix=False)]
    else:
        argv = shlex.split(text)
    if not argv:
        return None
    program = argv[0] if Path(argv[0]).is_file() else shutil.which(argv[0])
    if not program:
        return None
    return [program, *argv[1:]]


def venv_environment(venv_dir: Path) -> dict[str, str]:
    """Environment changes that make `venv_dir` the active Python, in any shell."""
    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    if not bin_dir.is_dir():
        return {}
    return {
        "VIRTUAL_ENV": str(venv_dir),
        "VIRTUAL_ENV_PROMPT": venv_dir.name,
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
    }


class PosixPtySession(QObject):
    """Manages an interactive shell process bound to a POSIX pseudo-terminal."""

    data_ready = pyqtSignal(str)
    process_exited = pyqtSignal(int)

    def __init__(self, cwd: Optional[str] = None, command: Optional[list[str]] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.cwd = str(cwd or Path.cwd())
        self.command = command
        self.master_fd: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self.notifier: Optional[QSocketNotifier] = None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.initial_cols = 80
        self.initial_rows = 24
        self.extra_env: dict[str, str] = {}  # merged over os.environ at start (e.g. venv activation)

    def start(self, cols: int = 80, rows: int = 24) -> bool:
        """Start the child shell process inside a new PTY."""
        if not HAS_POSIX_PTY:
            return False

        # Cleanly shut down any existing session and descriptors first
        self.close()

        self.initial_cols = max(10, cols)
        self.initial_rows = max(4, rows)

        try:
            self.master_fd, slave_fd = pty.openpty()
        except OSError:
            return False

        # Set initial PTY dimensions before process launch
        try:
            winsize = struct.pack("HHHH", self.initial_rows, self.initial_cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

        shell = self.command or get_default_shell()

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["TERM_PROGRAM"] = "ghostty"
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"
        env.update(self.extra_env)

        def _preexec():
            # Create a new session leader and set controlling terminal so /dev/tty,
            # job control, and SIGWINCH resize signals function properly.
            os.setsid()
            try:
                fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            except Exception:
                pass

        try:
            self.process = subprocess.Popen(
                shell,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.cwd,
                env=env,
                preexec_fn=_preexec,
                close_fds=True,
            )
        except Exception:
            os.close(slave_fd)
            if self.master_fd is not None:
                os.close(self.master_fd)
                self.master_fd = None
            return False

        # Close slave in parent process now that child inherited it
        os.close(slave_fd)

        # Set master_fd non-blocking
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Hook Qt event loop notifier for ready-to-read events
        self.notifier = QSocketNotifier(self.master_fd, QSocketNotifier.Type.Read, self)
        self.notifier.activated.connect(self._on_readable)
        return True

    def _on_readable(self):
        """Read pending bytes from master PTY and emit decoded text."""
        if self.master_fd is None:
            return

        try:
            # Read up to a budget per event invocation to keep Qt event loop responsive
            chunks_read = 0
            while chunks_read < 32:
                chunk = os.read(self.master_fd, 8192)
                if not chunk:
                    self._handle_exit()
                    return
                text = self.decoder.decode(chunk)
                if text:
                    self.data_ready.emit(text)
                chunks_read += 1
        except BlockingIOError:
            # All available bytes read for this cycle
            return
        except OSError as exc:
            if exc.errno in (errno.EIO, errno.EBADF):
                self._handle_exit()
            return

    def _handle_exit(self):
        """Process reached EOF or closed terminal."""
        if self.notifier:
            self.notifier.setEnabled(False)
            self.notifier = None

        exit_code = 0
        if self.process:
            try:
                exit_code = self.process.poll()
                if exit_code is None:
                    exit_code = self.process.wait(timeout=0.2)
            except Exception:
                exit_code = 0

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

        self.process_exited.emit(exit_code or 0)

    def write(self, data: str):
        """Write string to the master PTY, handling partial writes and buffer drains."""
        if self.master_fd is None:
            return
        try:
            encoded = data.encode("utf-8", errors="replace")
            total = len(encoded)
            offset = 0
            while offset < total:
                try:
                    written = os.write(self.master_fd, encoded[offset:])
                    if written <= 0:
                        break
                    offset += written
                except BlockingIOError:
                    # Buffer full; wait briefly for PTY to drain
                    _, writable, _ = select.select([], [self.master_fd], [], 0.05)
                    if not writable:
                        break
        except OSError:
            pass

    def resize(self, cols: int, rows: int):
        """Resize terminal window via ioctl TIOCSWINSZ."""
        if self.master_fd is None or not HAS_POSIX_PTY:
            return
        cols = max(10, cols)
        rows = max(4, rows)
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def close(self):
        """Cleanly terminate child process group and close file descriptors."""
        if self.notifier:
            self.notifier.setEnabled(False)
            self.notifier = None

        if self.process and self.process.poll() is None:
            pid = self.process.pid
            try:
                # Terminate entire process group to avoid leaving orphan processes
                os.killpg(pid, signal.SIGTERM)
                self.process.wait(timeout=0.3)
            except (ProcessLookupError, OSError):
                pass
            except Exception:
                try:
                    os.killpg(pid, signal.SIGKILL)
                    self.process.wait(timeout=0.2)
                except Exception:
                    pass

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_tmux_active(self) -> bool:
        """Check if tmux is currently running in this terminal session."""
        if not self.is_alive() or not self.process:
            return False
        try:
            pid = self.process.pid
            out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True, timeout=0.1)
            child_pids = [int(p) for p in out.strip().split() if p.isdigit()]
            for cpid in child_pids:
                comm = subprocess.check_output(["ps", "-p", str(cpid), "-o", "comm="], text=True, timeout=0.1).strip()
                if "tmux" in comm:
                    return True
        except Exception:
            pass
        return False


if sys.platform == "win32":
    PtySession = WinPtySession
    HAS_PTY = HAS_WINPTY
else:
    PtySession = PosixPtySession
    HAS_PTY = HAS_POSIX_PTY
