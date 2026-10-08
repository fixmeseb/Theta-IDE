"""EnvironmentEvaluatorCallback logging."""

from types import SimpleNamespace

from src.app.core.callbacks import EnvironmentEvaluatorCallback


class RecordingLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((dict(metrics), step))


def test_evaluation_metrics_reach_every_logger(monkeypatch):
    csv_logger, tensorboard_logger = RecordingLogger(), RecordingLogger()
    trainer = SimpleNamespace(global_step=40, loggers=[csv_logger, tensorboard_logger], logger=csv_logger,
                              fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=5)))
    logged = []
    pl_module = SimpleNamespace(cfg=SimpleNamespace(paradigm="online_rl"),
                                log=lambda name, value, **kwargs: logged.append(name))

    callback = EnvironmentEvaluatorCallback(cfg=None)
    callback.train_start_time = 0.0
    monkeypatch.setattr(callback, "evaluate", lambda trainer, pl_module: (25.0, 4.0))
    callback.evaluate_and_log(trainer, pl_module, transitions=10240)

    for recorded in (csv_logger, tensorboard_logger):
        assert len(recorded.calls) == 1
        metrics, step = recorded.calls[0]
        assert step == 4  # Lightning's own epoch-end step, not global_step (optimizer steps)
        assert (metrics["eval/reward"], metrics["eval/reward_std"], metrics["transitions"]) == (25.0, 4.0, 10240.0)
        assert {"time/eval", "time/train", "time/total"} <= metrics.keys()
    # Still reported to Lightning for checkpoint monitoring and the progress bar
    assert "eval/reward" in logged


def test_log_step_matches_lightning_and_falls_back_to_global_step():
    step = EnvironmentEvaluatorCallback._lightning_log_step
    loop = lambda n: SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=n))  # noqa: E731
    assert step(SimpleNamespace(global_step=0, fit_loop=loop(0)), 0) == 0  # evaluation before training
    assert step(SimpleNamespace(global_step=32, fit_loop=loop(4)), 2048) == 3
    assert step(SimpleNamespace(global_step=32), 2048) == 32  # attribute renamed in a future Lightning


def test_build_trainer_callbacks_online_vs_offline(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from src.app.core.lightning_builder import build_trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    monkeypatch.chdir(tmp_path)

    # Online RL paradigm on cartpole
    online_cfg = OmegaConf.create({
        "group": "test_grp",
        "experiment_id": "test_exp",
        "seed": 42,
        "paradigm": "online_rl",
        "total_timesteps": 1000,
        "intervals_count": 5,
        "eval_episodes": 10,
        "agent": {"name": "ppo"},
        "env": {"name": "cartpole", "offline_only": False, "num_envs": 1, "num_steps": 10},
    })
    trainer_online, _, _ = build_trainer(online_cfg)
    cb_types_online = [type(c) for c in trainer_online.callbacks]
    assert EnvironmentEvaluatorCallback in cb_types_online

    # Offline RL paradigm on cartpole (offline_only is False, but paradigm is offline_rl)
    offline_cfg = OmegaConf.create({
        "group": "test_grp",
        "experiment_id": "test_exp",
        "seed": 42,
        "paradigm": "offline_rl",
        "total_timesteps": 1000,
        "intervals_count": 1,
        "eval_episodes": 0,
        "agent": {"name": "cql", "epochs_per_interval": 10},
        "env": {"name": "cartpole", "offline_only": False, "monitor_metric": "val/loss"},
    })
    trainer_offline, _, _ = build_trainer(offline_cfg)
    cb_types_offline = [type(c) for c in trainer_offline.callbacks]
    assert EnvironmentEvaluatorCallback not in cb_types_offline
    ckpt_cbs = [c for c in trainer_offline.callbacks if isinstance(c, ModelCheckpoint) and c.monitor == "val/loss"]
    assert len(ckpt_cbs) >= 1

