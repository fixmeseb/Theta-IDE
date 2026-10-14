"""Telemetry records: the one format every sink receives and every reader understands.

A record is a plain dict, so it can be written as one line of JSON (events.jsonl) or handed to any
other destination unchanged:

    {"v": 1, "t": 1728745000.1, "type": "metrics", "step": 1200, "data": {"eval/reward": 187.3, ...}}

Types:
    start     the run began: "run" (group, experiment_id, agent, version, pid, host, ...), "config"
    metrics   one row of logged values: "step" and "data" (the same columns metrics.csv gets)
    system    resource use of the training process: "data" (cpu_percent, rss_mb, ...)
    event     something happened: "kind" (checkpoint, warning, error, ...), "level", "message"
    artifact  a file was written: "path", "kind" (checkpoint, figure, ...)
    config    hyperparameters reported after the start: "config"
    finish    the run ended: "status" (completed, failed), "summary"
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

SCHEMA_VERSION = 1
TYPES = ("start", "metrics", "system", "event", "artifact", "config", "finish")
LEVELS = ("info", "warning", "error")


def record(record_type: str, /, **fields: Any) -> dict[str, Any]:
    """A new record of `record_type`, stamped with the schema version and the current time.

    Positional-only, because event and artifact records have a field named "kind" of their own.
    """
    if record_type not in TYPES:
        raise ValueError(f"Unknown telemetry record type {record_type!r}; expected one of {', '.join(TYPES)}")
    return {"v": SCHEMA_VERSION, "t": time.time(), "type": record_type, **fields}


def scalar(value: Any) -> Any:
    """A loggable value: numbers, booleans and strings as they are, tensors and numpy scalars as numbers."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_safe(value: Any) -> Any:
    """JSON has no NaN or infinity: write them as null (the run also gets a warning event for them)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool, float)) or value is None:
        return value
    return _json_safe(scalar(value))  # a tensor or numpy number becomes a plain value first


def encode(rec: dict[str, Any]) -> str:
    """One line of JSON, without a trailing newline."""
    return json.dumps(_json_safe(rec), ensure_ascii=False, separators=(",", ":"), default=str)


def decode(line: str) -> dict[str, Any] | None:
    """The record on one line, or None for a blank, unreadable or unknown line (skipped by readers)."""
    line = line.strip()
    if not line:
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(rec, dict) or rec.get("type") not in TYPES:
        return None
    return rec


def metrics_row(rec: dict[str, Any]) -> dict[str, float]:
    """A metrics record as a metrics.csv row: its numeric values plus "step"."""
    row = {k: float(v) for k, v in (rec.get("data") or {}).items() if is_finite_number(v)}
    if is_finite_number(rec.get("step")):
        row["step"] = float(rec["step"])
    return row
