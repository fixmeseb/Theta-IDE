"""Telemetry: one logging API for every training engine, written to swappable sinks.

    from src.app import telemetry
    run = telemetry.init(group, experiment_id, agent, config=cfg)
    run.log({"eval/reward": 187.3}, step=1200, transitions=24576)
    run.finish()

Records (records.py) are the shared format; sinks (sinks.py) decide where they go: metrics.csv,
events.jsonl (the live feed the app reads), memory, or any registered destination. Lightning runs reach
it through telemetry.lightning.ThetaLogger, imported separately so other engines never load Lightning.
"""

from .records import SCHEMA_VERSION, decode, encode, metrics_row, record
from .run import Run, init, settings_from_config, version_dir_for
from .sinks import DEFAULT_SINKS, SINKS, STREAM_FILE, CsvSink, MemorySink, Sink, StreamSink, register_sink

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_SINKS",
    "SINKS",
    "STREAM_FILE",
    "CsvSink",
    "MemorySink",
    "Run",
    "Sink",
    "StreamSink",
    "decode",
    "encode",
    "init",
    "metrics_row",
    "record",
    "register_sink",
    "settings_from_config",
    "version_dir_for",
]
