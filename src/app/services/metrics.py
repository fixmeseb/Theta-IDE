"""Reading Lightning's metrics.csv files: whole files, or only what was appended since last time."""

from __future__ import annotations

import csv
from pathlib import Path


def latest_version_file(agent_dir: Path, name: str) -> Path | None:
    """`name` inside the newest version_N folder of an agent's log directory (by N, so version_10 > version_9)."""

    def number(path: Path) -> int:
        suffix = path.parent.name.removeprefix("version_")
        return int(suffix) if suffix.isdigit() else -1

    files = sorted(agent_dir.glob(f"version_*/{name}"), key=number)
    return files[-1] if files else None


def latest_metrics_csv(agent_dir: Path) -> Path | None:
    return latest_version_file(agent_dir, "metrics.csv")


def _numeric(row: dict) -> dict[str, float]:
    values = {}
    for key, value in row.items():
        if key and value not in (None, ""):
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                pass
    return values


def read_metrics_rows(path: Path) -> list[dict[str, float]]:
    """Every complete row of a metrics CSV, numeric columns only."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = text.split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]  # the last line is still being written
    return [_numeric(row) for row in csv.DictReader(lines)]


def read_metrics_table(path: Path) -> list[dict[str, str]]:
    """Every row of a metrics CSV as strings (all columns), for the Results browser."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MetricsTail:
    """Reads only the rows appended to one metrics CSV since the previous read.

    Lightning rewrites the file from scratch when a new column appears, and the rewrite is
    usually *longer* than what it replaces: a supervised run logs `epoch,step,val/loss` during
    the validation sanity check and then `epoch,step,train/loss,training_time_seconds,val/loss`
    once training starts. Watching for the file to shrink therefore misses the common case, and
    a stale header zipped onto wider rows files every value under the wrong name - train/loss
    arriving as val/loss, and the columns past the end of the old header disappearing entirely.

    So the header itself is compared on every read. A changed header means the file was
    rewritten, whichever direction its size went, and `reset` tells the caller to drop what it
    had and take these rows as the whole run.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.header: list[str] = []

    def read(self) -> tuple[list[dict[str, float]], bool]:
        """(new rows, reset). Rows are only returned once their line is complete."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], False
        reset = False
        if size < self.offset:
            self.offset, self.header, reset = 0, [], True
        try:
            with open(self.path, "rb") as f:
                header_line = f.readline().decode("utf-8", errors="replace")
                if not header_line.endswith("\n"):
                    return [], reset  # header still being written
                header = next(csv.reader([header_line.rstrip("\r\n")]), [])
                if header != self.header:
                    # A rewrite. Only a reset if we had already read a different header:
                    # the first read of a file is not one.
                    reset = reset or bool(self.header)
                    self.header, self.offset = header, f.tell()
                elif not self.offset:
                    self.offset = f.tell()
                f.seek(self.offset)
                raw = f.read()
        except OSError:
            return [], reset
        last_newline = raw.rfind(b"\n")
        if last_newline == -1:
            return [], reset
        complete = raw[: last_newline + 1]
        self.offset += len(complete)
        rows = []
        for line in complete.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            values = next(csv.reader([line]), [])
            numeric = _numeric(dict(zip(self.header, values)))
            if numeric:
                rows.append(numeric)
        return rows, reset
