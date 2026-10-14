"""Resource use of the training process, sampled from inside it so every run gets it (app, CLI or Slurm).

Uses psutil when it is installed (CPU and memory of the process and its children, and of the machine);
otherwise only what the standard library can tell (this process's CPU time, and its memory on Linux).
GPU memory comes from torch when the run already uses CUDA. No dependency is added for any of it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable

try:
    import psutil
except ImportError:  # the training environment may not have the GUI's extras
    psutil = None

DEFAULT_INTERVAL = 2.0  # seconds between samples


class SystemProbe:
    """Takes one sample at a time. CPU percentages are measured since the previous sample."""

    def __init__(self, pid: int | None = None):
        self.pid = pid or os.getpid()
        self._process = None
        self._children: dict[int, Any] = {}
        self._last_wall = time.monotonic()
        self._last_cpu = time.process_time()
        if psutil is not None:
            try:
                self._process = psutil.Process(self.pid)
                self._process.cpu_percent(None)  # the first reading only sets the baseline
                psutil.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None

    def sample(self) -> dict[str, float]:
        data: dict[str, float] = {}
        if self._process is not None:
            data.update(self._psutil_sample())
        else:
            data.update(self._stdlib_sample())
        data.update(_gpu_sample())
        return data

    def _psutil_sample(self) -> dict[str, float]:
        data: dict[str, float] = {}
        try:
            cpu = self._process.cpu_percent(None)
            rss = self._process.memory_info().rss
            data["threads"] = float(self._process.num_threads())
            alive = {}
            for child in self._process.children(recursive=True):
                known = self._children.get(child.pid, child)
                try:
                    if known is child:
                        child.cpu_percent(None)  # new child: start measuring from now
                    else:
                        cpu += known.cpu_percent(None)
                    rss += known.memory_info().rss
                    alive[child.pid] = known
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self._children = alive
            data["cpu_percent"] = round(cpu, 1)
            data["rss_mb"] = round(rss / 2**20, 1)
            data["children"] = float(len(alive))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
        data["host_cpu_percent"] = round(psutil.cpu_percent(None), 1)
        data["host_mem_percent"] = round(psutil.virtual_memory().percent, 1)
        return data

    def _stdlib_sample(self) -> dict[str, float]:
        wall, cpu = time.monotonic(), time.process_time()
        elapsed = wall - self._last_wall
        data: dict[str, float] = {}
        if elapsed > 0:
            data["cpu_percent"] = round(100.0 * (cpu - self._last_cpu) / elapsed, 1)
        self._last_wall, self._last_cpu = wall, cpu
        rss = _linux_rss_mb()
        if rss is not None:
            data["rss_mb"] = rss
        if hasattr(os, "getloadavg"):
            try:
                data["host_load_1m"] = round(os.getloadavg()[0], 2)
            except OSError:
                pass
        return data


def _linux_rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _gpu_sample() -> dict[str, float]:
    """CUDA memory this process holds, per device, if the run already imported torch and uses CUDA."""
    torch = sys.modules.get("torch")  # never import torch just to look
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return {}
        data = {}
        for i in range(torch.cuda.device_count()):
            data[f"gpu{i}_mem_allocated_mb"] = round(torch.cuda.memory_allocated(i) / 2**20, 1)
            data[f"gpu{i}_mem_reserved_mb"] = round(torch.cuda.memory_reserved(i) / 2**20, 1)
        return data
    except Exception:  # a CUDA error must never stop training
        return {}


class SystemSampler:
    """A daemon thread calling `emit(sample)` every `interval` seconds until stopped."""

    def __init__(
        self,
        emit: Callable[[dict[str, float]], None],
        interval: float = DEFAULT_INTERVAL,
        probe: SystemProbe | None = None,
    ):
        self.emit = emit
        self.interval = interval
        self.probe = probe or SystemProbe()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="telemetry-system", daemon=True)

    def start(self) -> SystemSampler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.interval + 1)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                sample = self.probe.sample()
            except Exception:  # sampling is best effort
                continue
            if sample:
                self.emit(sample)
