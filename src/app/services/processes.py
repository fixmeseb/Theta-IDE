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
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.terminate()
            _, alive = psutil.wait_procs(children + [parent], timeout=2.0)
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
