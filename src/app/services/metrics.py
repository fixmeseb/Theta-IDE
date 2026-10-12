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

    Lightning rewrites the file from scratch when a new column appears; the tail notices the
    file shrinking, starts over, and reports `reset` so the caller can drop what it had.
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
                if not self.header:
                    header_line = f.readline().decode("utf-8", errors="replace")
                    if not header_line.endswith("\n"):
                        return [], reset  # header still being written
                    self.header = next(csv.reader([header_line.rstrip("\r\n")]), [])
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
