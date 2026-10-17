"""Periodic checkpoints set by the experiment: checkpoint_every_steps or num_checkpoints."""

from types import SimpleNamespace

import pytest

from src.app.core.callbacks import PeriodicCheckpointCallback
from src.app.core.checkpoints import CheckpointSchedule, checkpoint_errors, checkpoint_interval


def saved_at(interval, total, steps):
    schedule = CheckpointSchedule(interval, total)
    return [t for t in steps if schedule.due(t)]


def test_every_n_steps_saves_at_each_multiple_and_at_the_end():
    assert saved_at(50_000, 144_000, range(0, 146_000, 2_000)) == [50_000, 100_000, 144_000]


def test_a_count_spreads_that_many_over_the_run():
    cfg = {"num_checkpoints": 3, "total_timesteps": 100}
    interval = checkpoint_interval(cfg)
    assert interval == 34
    assert saved_at(interval, 100, range(0, 101, 10)) == [40, 70, 100]


def test_crossing_several_thresholds_at_once_saves_one_checkpoint():
    assert saved_at(10, 100, [5, 35, 41, 100]) == [35, 41, 100]


def test_no_keys_means_no_periodic_checkpoints():
    assert checkpoint_interval({"total_timesteps": 1000}) is None
    assert checkpoint_errors({"paradigm": "offline_rl"}) == []


@pytest.mark.parametrize(
    "cfg, problem",
    [
        ({"paradigm": "online_rl", "checkpoint_every_steps": 10, "num_checkpoints": 2}, "not both"),
        ({"paradigm": "online_rl", "num_checkpoints": 0}, "at least 1"),
        ({"paradigm": "online_rl", "checkpoint_every_steps": 2.5}, "whole number"),
        ({"paradigm": "supervised", "num_checkpoints": 5}, "eval_every_n_epochs"),
    ],
)
def test_bad_checkpoint_settings_are_reported(cfg, problem):
    (error,) = checkpoint_errors(cfg)
    assert problem in error


def test_the_callback_saves_and_reports_to_telemetry(tmp_path, monkeypatch):
    from src.app.telemetry import lightning as telemetry_lightning

    class FakeLogger:
        def __init__(self):
            self.records = []
            self.run = SimpleNamespace(artifact=lambda path, **kw: self.records.append(("artifact", path)),
                                       event=lambda kind, message, **kw: self.records.append((kind, message)))

    monkeypatch.setattr(telemetry_lightning, "ThetaLogger", FakeLogger)
    theta = FakeLogger()
    saved = []
    trainer = SimpleNamespace(loggers=[theta], save_checkpoint=saved.append)
    callback = PeriodicCheckpointCallback(str(tmp_path), interval=1000, total=2000)
    for transitions in (512, 1024, 1536, 2048):
        callback.on_train_epoch_end(trainer, SimpleNamespace(global_step_count=transitions))
    assert [p.split("step_")[-1] for p in saved] == ["000001024.ckpt", "000002048.ckpt"]
    assert theta.records[1] == ("checkpoint", "Saved step_000001024.ckpt at 1,024 transitions")


def test_the_gui_can_import_the_schedule_without_lightning():
    import subprocess
    import sys

    code = "import sys, src.app.pipeline.validation; print('lightning' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"  # importing Lightning (and torch) added ~6 s to the IDE's startup
