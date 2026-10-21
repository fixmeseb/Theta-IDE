"""Stopping a training job and everything it started, and sampling its resource use."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

try:
    import psutil
except ImportError:
    psutil = None


def kill_process_tree(pid: int) -> None:
    """Terminate the pipeline and every training/plotting subprocess it started."""
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        # Process enumeration can be denied on managed macOS hosts even when
        # the IDE is allowed to signal its own child process. Keep cancellation
        # working by terminating the known runner when its descendants cannot
        # be enumerated.
        try:
            children = parent.children(recursive=True)
        except (psutil.AccessDenied, PermissionError, OSError):
            children = []

        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
        try:
            _, alive = psutil.wait_procs(children + [parent], timeout=2.0)
        except (psutil.AccessDenied, PermissionError, OSError):
            alive = []
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def process_stats(pid: int) -> dict[str, float]:
    """CPU percent and resident memory (MB) for a running process; empty if unavailable."""
    if psutil is None or not pid:
        return {}
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return {}
        return {"cpu_percent": proc.cpu_percent(), "memory_mb": round(proc.memory_info().rss / (1024 * 1024), 1)}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {}
