"""Telemetry: the run API, its sinks, system sampling and the Lightning bridge."""

import json
import sys
import time
import types
from pathlib import Path

import pytest

from src.app import telemetry
from src.app.core.contract_logger import ContractLogger
from src.app.services.metrics import MetricsTail
from src.app.telemetry import system as system_module
from src.app.telemetry.records import decode, encode, metrics_row, record
from src.app.telemetry.sinks import MemorySink, resolve_sink


def memory_run(tmp_path, **kwargs):
    sink = MemorySink()
    run = telemetry.Run(tmp_path / "version_0", [sink], system_interval=0, **kwargs)
    return run, sink


def of_type(sink, kind):
    return [r for r in sink.records if r["type"] == kind]


# ── The run API ──────────────────────────────────────────────────────────────


def test_run_turns_calls_into_records_for_every_sink(tmp_path):
    run, sink = memory_run(tmp_path, config={"lr": 0.001}, meta={"agent": "ppo"})
    run.log({"eval/reward": 10.0}, transitions=512)
    run.log({"losses/total_loss": 0.5}, step=7, epoch=1)
    run.log({"eval/reward": 30.0})  # the step carries on from the previous one
    run.event("checkpoint", "saved best_model.ckpt")
    run.artifact("ckpt/best_model.ckpt", kind="checkpoint")
    run.finish()
    run.finish()  # finishing twice does nothing

    start = sink.records[0]
    assert start["type"] == "start" and start["config"] == {"lr": 0.001}
    assert start["run"]["agent"] == "ppo" and start["run"]["version"] == "version_0"
    metrics = of_type(sink, "metrics")
    assert [(r["step"], r["data"]) for r in metrics] == [
        (1, {"eval/reward": 10.0, "transitions": 512}),
        (7, {"losses/total_loss": 0.5, "epoch": 1}),
        (8, {"eval/reward": 30.0}),
    ]
    assert of_type(sink, "event")[0]["message"] == "saved best_model.ckpt"
    assert of_type(sink, "artifact")[0]["kind"] == "checkpoint"
    (finish,) = of_type(sink, "finish")
    assert finish["status"] == "completed" and finish["summary"]["eval/reward"] == 30.0
    assert all(r["v"] == telemetry.SCHEMA_VERSION and r["t"] > 0 for r in sink.records)


def test_non_finite_values_warn_once_per_metric(tmp_path):
    run, sink = memory_run(tmp_path)
    for step in range(3):
        run.log({"losses/value_loss": float("nan"), "eval/reward": 1.0}, step=step)
    warnings = [r for r in of_type(sink, "event") if r["level"] == "warning"]
    assert len(warnings) == 1 and "losses/value_loss is nan at step 0" in warnings[0]["message"]
    assert "losses/value_loss" not in run.summary
    run.finish()


def test_a_failure_inside_the_run_finishes_it_as_failed(tmp_path):
    sink = MemorySink()
    with pytest.raises(RuntimeError):
        with telemetry.Run(tmp_path / "version_0", [sink], system_interval=0) as run:
            run.log({"eval/reward": 1.0})
            raise RuntimeError("diverged")
    assert of_type(sink, "event")[-1]["message"] == "RuntimeError: diverged"
    assert of_type(sink, "finish")[0]["status"] == "failed"


def test_a_broken_sink_is_skipped_and_the_others_carry_on(tmp_path, capsys):
    class Broken(MemorySink):
        def write(self, rec):
            raise OSError("disk full")

    good = MemorySink()
    run = telemetry.Run(tmp_path / "version_0", [Broken(), good], system_interval=0)
    run.log({"eval/reward": 1.0})
    run.log({"eval/reward": 2.0})
    run.finish()
    assert [r["data"]["eval/reward"] for r in of_type(good, "metrics")] == [1.0, 2.0]
    assert capsys.readouterr().err.count("stopped after an error") == 1


def test_init_picks_the_next_version_folder(tmp_path):
    (tmp_path / "g" / "exp" / "ppo" / "version_0").mkdir(parents=True)
    run = telemetry.init("g", "exp", "ppo", base_dir=tmp_path, sinks=["memory"], system_interval=0)
    assert run.version_dir == tmp_path / "g" / "exp" / "ppo" / "version_1"
    run.finish()
    with pytest.raises(ValueError):
        telemetry.init("g", None, "ppo")


# ── Sinks ────────────────────────────────────────────────────────────────────


def test_csv_sink_writes_exactly_what_contract_logger_writes(tmp_path):
    calls = [
        ({"eval/reward": 10.0, "train/reward": 9.0}, {"step": 1000, "transitions": 1000}),
        ({"losses/value_loss": 3.5}, {"step": 2000, "transitions": 2000}),  # a new column rewrites the file
        ({"eval/reward": 20.0}, {"step": 3000, "transitions": 3000, "epoch": 2}),
        ({"eval/reward": 25.0}, {}),  # auto step
    ]
    direct = ContractLogger(log_dir=tmp_path / "direct", version="version_0")
    for metrics, kwargs in calls:
        direct.log(dict(metrics), **kwargs)
    direct.close()
    run = telemetry.Run(tmp_path / "via_run" / "version_0", ["csv"], system_interval=0)
    for metrics, kwargs in calls:
        run.log(dict(metrics), **kwargs)
    run.finish()
    expected = (tmp_path / "direct" / "version_0" / "metrics.csv").read_bytes()
    assert (tmp_path / "via_run" / "version_0" / "metrics.csv").read_bytes() == expected


def test_stream_sink_appends_one_json_record_per_line(tmp_path):
    run = telemetry.Run(tmp_path / "version_0", ["stream"], system_interval=0)
    run.log({"eval/reward": float("inf"), "losses/total_loss": 0.5}, step=3)
    run.finish()
    lines = (tmp_path / "version_0" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]  # strict JSON: infinity became null
    assert [r["type"] for r in records] == ["start", "metrics", "event", "finish"]
    assert records[1]["data"] == {"eval/reward": None, "losses/total_loss": 0.5}


def test_sinks_resolve_from_names_import_paths_and_objects(tmp_path, monkeypatch):
    module = types.ModuleType("custom_sinks_for_test")
    module.Collect = MemorySink
    monkeypatch.setitem(sys.modules, "custom_sinks_for_test", module)
    assert isinstance(resolve_sink("custom_sinks_for_test:Collect", tmp_path), MemorySink)
    telemetry.register_sink("collect", MemorySink)
    try:
        run = telemetry.Run(tmp_path / "version_0", ["collect"], system_interval=0)
        assert isinstance(run.sinks[0], MemorySink)
        run.finish()
    finally:
        telemetry.SINKS.pop("collect")
    ready = MemorySink()
    assert resolve_sink(ready, tmp_path) is ready
    with pytest.raises(ValueError, match="Unknown telemetry sink"):
        resolve_sink("nope", tmp_path)


def test_settings_come_from_the_telemetry_config_key():
    from omegaconf import OmegaConf

    assert telemetry.settings_from_config(OmegaConf.create({})) == (["csv", "stream"], 2.0)
    cfg = OmegaConf.create({"telemetry": {"sinks": ["stream"], "system_interval": 0}})
    assert telemetry.settings_from_config(cfg) == (["stream"], 0.0)


def test_records_round_trip_and_unreadable_lines_are_skipped():
    rec = record("metrics", step=5, data={"a": 1.5})
    assert decode(encode(rec)) == rec
    assert decode("") is None and decode("{not json") is None and decode('{"type": "mystery"}') is None
    assert metrics_row(rec) == {"a": 1.5, "step": 5.0}
    with pytest.raises(ValueError):
        record("mystery")


# ── System sampling ──────────────────────────────────────────────────────────


def test_system_probe_reports_cpu_and_memory():
    probe = system_module.SystemProbe()
    sum(i * i for i in range(200000))
    time.sleep(0.05)  # Windows' clock can advance in 15 ms steps: make sure time has passed
    sample = probe.sample()
    assert sample["cpu_percent"] >= 0
    if system_module.psutil is not None:
        assert sample["rss_mb"] > 0 and "host_cpu_percent" in sample


def test_system_probe_works_without_psutil(monkeypatch):
    monkeypatch.setattr(system_module, "psutil", None)
    probe = system_module.SystemProbe()
    sum(i * i for i in range(200000))
    time.sleep(0.05)  # Windows' clock can advance in 15 ms steps: make sure time has passed
    assert probe.sample()["cpu_percent"] >= 0


def test_the_sampler_reports_into_the_run_until_it_finishes(tmp_path):
    sink = MemorySink()
    run = telemetry.Run(tmp_path / "version_0", [sink], system_interval=0.05)
    deadline = time.time() + 5
    while not of_type(sink, "system") and time.time() < deadline:
        time.sleep(0.02)
    run.finish()
    assert of_type(sink, "system")[0]["data"]["cpu_percent"] >= 0
    count = len(of_type(sink, "system"))
    time.sleep(0.15)
    assert len(of_type(sink, "system")) == count  # stopped with the run


# ── Lightning ────────────────────────────────────────────────────────────────


def _lightning_fit(tmp_path, fail=False):
    torch = pytest.importorskip("torch")
    L = pytest.importorskip("lightning")
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from src.app.telemetry.lightning import TelemetryCallback, ThetaLogger, finish_telemetry

    class Tiny(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, _index):
            if fail and self.global_step == 3:
                raise RuntimeError("loss exploded")
            loss = self.layer(batch[0]).pow(2).mean()
            self.log("train/loss", loss, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def train_dataloader(self):
            data = torch.utils.data.TensorDataset(torch.randn(16, 2))
            return torch.utils.data.DataLoader(data, batch_size=4)

    csv_logger = CSVLogger(tmp_path, name="tiny")
    theta = ThetaLogger(lambda: csv_logger.log_dir, ["stream"], {"lr": 0.01}, {"agent": "tiny"}, system_interval=0)
    trainer = L.Trainer(
        max_epochs=2,
        logger=[csv_logger, theta],
        callbacks=[TelemetryCallback(theta), ModelCheckpoint(dirpath=tmp_path / "ckpt", monitor="train/loss_epoch")],
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    if fail:
        with pytest.raises(RuntimeError):
            trainer.fit(Tiny())
    else:
        trainer.fit(Tiny())
        for lg in trainer.loggers:  # what finalize_training does after fit
            lg.log_metrics({"training_time_seconds": 1.5}, step=trainer.global_step)
        finish_telemetry(trainer)
    stream = Path(csv_logger.log_dir) / "events.jsonl"
    return csv_logger, [json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines()]


def test_lightning_runs_stream_the_same_rows_csvlogger_writes(tmp_path):
    csv_logger, records = _lightning_fit(tmp_path)
    kinds = [r["type"] for r in records]
    assert kinds[0] == "start" and kinds[-1] == "finish" and records[-1]["status"] == "completed"
    csv_rows, _ = MetricsTail(Path(csv_logger.log_dir) / "metrics.csv").read()
    streamed = [metrics_row(r) for r in records if r["type"] == "metrics"]
    assert streamed[: len(csv_rows)] == csv_rows
    assert streamed[-1]["training_time_seconds"] == 1.5  # logged after fit, before the run finished
    assert records[-1]["summary"]["training_time_seconds"] == 1.5
    checkpoints = [r for r in records if r["type"] == "artifact" and r["kind"] == "checkpoint"]
    assert checkpoints and Path(checkpoints[-1]["path"]).exists()
    assert any(r["type"] == "event" and r["kind"] == "stage" for r in records)


def test_a_failing_lightning_run_records_why(tmp_path):
    _, records = _lightning_fit(tmp_path, fail=True)
    errors = [r for r in records if r["type"] == "event" and r["level"] == "error"]
    assert errors and "loss exploded" in errors[0]["message"]
    assert records[-1]["type"] == "finish" and records[-1]["status"] == "failed"


def test_the_lightning_builder_adds_the_stream_beside_csvlogger(tmp_path):
    pytest.importorskip("lightning")
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf

    from src.app.core.lightning_builder import setup_loggers
    from src.app.telemetry.lightning import ThetaLogger

    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "ppo"}})
    loggers = setup_loggers(cfg, str(tmp_path))
    assert [type(lg) for lg in loggers] == [CSVLogger, ThetaLogger]
    assert loggers[1].version == "version_0"
    cfg.telemetry = {"sinks": ["csv"]}  # CSV only: CSVLogger already writes it, so no bridge
    assert [type(lg) for lg in setup_loggers(cfg, str(tmp_path))] == [CSVLogger]
