import os
import pickle

import numpy as np
import pytest
import torch

from src.app.dataset_utils import DatasetWriter


@pytest.fixture
def writer(tmp_path):
    return DatasetWriter(save_dir=tmp_path / "dataset", chunk_size=10, env_name="test_env")


def test_writer_init_creates_dir(tmp_path):
    DatasetWriter(save_dir=tmp_path / "new_dataset", chunk_size=10)
    assert (tmp_path / "new_dataset").exists()


def test_writer_add_standard_transition(writer):
    obs = torch.zeros((4,))
    action = torch.tensor(1)
    reward = torch.tensor(1.0)
    next_obs = torch.ones((4,))
    done = torch.tensor(False)

    writer.add(obs, action, reward, next_obs, done)
    assert len(writer.buffer) == 1
    t = writer.buffer[0]
    assert "obs" in t
    assert set(t) == {"obs", "action", "reward", "next_obs", "done"}


def test_writer_flush_writes_pickle(writer, tmp_path):
    obs = np.zeros((4,))
    action = 1
    reward = 1.0
    next_obs = np.ones((4,))
    done = False
    writer.add(obs, action, reward, next_obs, done)

    writer.flush()
    assert len(list((tmp_path / "dataset").glob("*.pkl"))) == 1


def test_writer_round_trip(writer, tmp_path):
    obs = np.array([1, 2, 3, 4], dtype=np.float32)
    action = 1
    reward = 1.0
    next_obs = np.array([5, 6, 7, 8], dtype=np.float32)
    done = True

    writer.add(obs, action, reward, next_obs, done)
    writer.flush()

    pkl_files = list((tmp_path / "dataset").glob("*.pkl"))
    assert len(pkl_files) == 1

    with open(pkl_files[0], "rb") as f:
        data = pickle.load(f)

    assert len(data) == 1
    t = data[0]
    assert np.array_equal(t["obs"], obs)
    assert t["action"] == action
    assert t["reward"] == reward
    assert np.array_equal(t["next_obs"], next_obs)
    assert t["done"] == done


def test_writer_empty_flush(writer, tmp_path):
    writer.flush()
    assert len(list((tmp_path / "dataset").glob("*.pkl"))) == 0


def test_rl_data_module_dataloaders(tmp_path):
    from omegaconf import OmegaConf
    from src.app.data.rl_data_module import RLDataModule

    # 1. Online RL mode -> val_dataloader returns None
    online_cfg = OmegaConf.create({
        "paradigm": "online_rl",
        "agent": {"batch_size": 16},
    })
    dm_online = RLDataModule(online_cfg)
    dm_online.setup()
    assert dm_online.train_dataloader() is not None
    assert dm_online.val_dataloader() is None

    # 2. Offline RL without validation split -> val_dataloader returns None
    dataset_dir = tmp_path / "offline_ds"
    writer = DatasetWriter(save_dir=dataset_dir, chunk_size=10)
    for _ in range(5):
        writer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
    writer.flush()
    writer.close()

    offline_cfg = OmegaConf.create({
        "paradigm": "offline_rl",
        "dataset_path": str(dataset_dir),
        "val_split": 0.0,
        "seed": 42,
        "env": {"name": "cartpole"},
        "agent": {"batch_size": 2},
    })
    dm_offline = RLDataModule(offline_cfg)
    dm_offline.setup()
    assert dm_offline.train_dataloader() is not None
    assert dm_offline.val_dataloader() is None


