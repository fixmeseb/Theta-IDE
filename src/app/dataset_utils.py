import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


class DatasetWriter:
    """Writes RL transitions to chunked offline datasets."""

    def __init__(self, save_dir, chunk_size=100000, env_name="env", cfg=None):
        """
        Initialize the DatasetWriter.

        Args:
            save_dir: Path where dataset chunks will be saved.
            chunk_size: Maximum number of transitions per chunk file.
            env_name: Name of the environment.
            cfg: Configuration dictionary or object.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.env_name = env_name
        self.total_steps = 0
        self.cfg = cfg

        # Determine starting chunk index for recovery
        existing_chunks = list(self.save_dir.glob(f"dataset_{self.env_name}_*.pkl"))
        if existing_chunks:
            max_idx = max(
                [int(p.stem.split("_")[-1]) for p in existing_chunks if p.stem.split("_")[-1].isdigit()] + [-1]
            )
            self.chunk_idx = max_idx + 1
        else:
            self.chunk_idx = 0

    def add(self, *args, **kwargs):
        """Add a single transition to the buffer.

        Supports both modern 5-param clean RL:
            add(obs, action, reward, next_obs, done, logic_obs=None, next_logic_obs=None)
        and legacy 7-param:
            add(obs, logic_obs, action, reward, next_obs, next_logic_obs, done)
        """
        if len(args) == 7:
            obs, logic_obs, action, reward, next_obs, next_logic_obs, done = args
        elif len(args) == 5:
            obs, action, reward, next_obs, done = args
            logic_obs = kwargs.get("logic_obs", None)
            next_logic_obs = kwargs.get("next_logic_obs", None)
        else:
            obs = kwargs["obs"]
            action = kwargs["action"]
            reward = kwargs["reward"]
            next_obs = kwargs["next_obs"]
            done = kwargs["done"]
            logic_obs = kwargs.get("logic_obs", None)
            next_logic_obs = kwargs.get("next_logic_obs", None)

        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                if len(x.shape) > 2:
                    return x.astype(np.uint8)
                return x.astype(np.float32)
            return x

        transition = {
            "obs": to_cpu(obs, is_obs=True),
            "action": to_cpu(action),
            "reward": to_cpu(reward),
            "next_obs": to_cpu(next_obs, is_obs=True),
            "done": to_cpu(done),
        }
        if logic_obs is not None:
            transition["logic_obs"] = to_cpu(logic_obs)
        if next_logic_obs is not None:
            transition["next_logic_obs"] = to_cpu(next_logic_obs)

        self.buffer.append(transition)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def batch_add(self, *args, **kwargs):
        """Add a batch of transitions to the buffer."""
        if len(args) == 7:
            obs, logic_obs, action, reward, next_obs, next_logic_obs, done = args
        elif len(args) == 5:
            obs, action, reward, next_obs, done = args
            logic_obs = kwargs.get("logic_obs", None)
            next_logic_obs = kwargs.get("next_logic_obs", None)
        else:
            obs = kwargs["obs"]
            action = kwargs["action"]
            reward = kwargs["reward"]
            next_obs = kwargs["next_obs"]
            done = kwargs["done"]
            logic_obs = kwargs.get("logic_obs", None)
            next_logic_obs = kwargs.get("next_logic_obs", None)

        # Ensure input is batch-like (at least 1D)
        if len(obs.shape) == 1:
            batch_size = 1
            obs = obs.unsqueeze(0) if isinstance(obs, torch.Tensor) else np.expand_dims(obs, 0)
            if logic_obs is not None:
                logic_obs = (
                    logic_obs.unsqueeze(0) if isinstance(logic_obs, torch.Tensor) else np.expand_dims(logic_obs, 0)
                )
            action = action.unsqueeze(0) if isinstance(action, torch.Tensor) else np.expand_dims(action, 0)
            reward = torch.tensor([reward]) if not isinstance(reward, torch.Tensor) else reward.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0) if isinstance(next_obs, torch.Tensor) else np.expand_dims(next_obs, 0)
            if next_logic_obs is not None:
                next_logic_obs = (
                    next_logic_obs.unsqueeze(0)
                    if isinstance(next_logic_obs, torch.Tensor)
                    else np.expand_dims(next_logic_obs, 0)
                )
            done = torch.tensor([done]) if not isinstance(done, torch.Tensor) else done.unsqueeze(0)
        else:
            batch_size = len(obs)

        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                if len(x.shape) > 2:
                    return x.astype(np.uint8)
                return x.astype(np.float32)
            return x

        obs_cpu = to_cpu(obs, is_obs=True)
        logic_obs_cpu = to_cpu(logic_obs) if logic_obs is not None else None
        action_cpu = to_cpu(action)
        reward_cpu = to_cpu(reward)
        next_obs_cpu = to_cpu(next_obs, is_obs=True)
        next_logic_obs_cpu = to_cpu(next_logic_obs) if next_logic_obs is not None else None
        done_cpu = to_cpu(done)

        for i in range(batch_size):
            transition = {
                "obs": obs_cpu[i],
                "action": action_cpu[i],
                "reward": reward_cpu[i],
                "next_obs": next_obs_cpu[i],
                "done": done_cpu[i],
            }
            if logic_obs_cpu is not None:
                transition["logic_obs"] = logic_obs_cpu[i]
            if next_logic_obs_cpu is not None:
                transition["next_logic_obs"] = next_logic_obs_cpu[i]
            self.buffer.append(transition)

        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def write(self, obs, action, reward, next_obs, done, logic_obs=None, next_logic_obs=None):
        """Standard keyword-based write interface for online RL agents."""
        if hasattr(obs, "shape") and len(obs.shape) > 1 and not (len(obs.shape) == 3 and obs.shape[0] in [1, 3, 4]):
            self.batch_add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                logic_obs=logic_obs,
                next_logic_obs=next_logic_obs,
            )
        else:
            self.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                logic_obs=logic_obs,
                next_logic_obs=next_logic_obs,
            )

    def flush(self):
        """
        Flush the current buffer to a chunked dataset file on disk.
        """
        if not self.buffer:
            return

        filename = self.save_dir / f"dataset_{self.env_name}_{self.chunk_idx:05d}.pkl"
        tmp_filename = filename.with_suffix(".pkl.tmp")
        with open(tmp_filename, "wb") as f:
            pickle.dump(self.buffer, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)

        # print(f"Saved dataset chunk {self.chunk_idx} with {len(self.buffer)} transitions to {filename}")
        self.total_steps += len(self.buffer)
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        """
        Flush any remaining transitions and generate the dataset manifest.
        """
        self.flush()

        try:
            import datetime
            import json

            from src.app.core.metadata import collect_run_metadata

            meta = collect_run_metadata(getattr(self, "cfg", None))

            if hasattr(self, "cfg") and self.cfg is not None:
                agent = self.cfg.agent.name
                exp_id = self.cfg.experiment_id
                group = self.cfg.group
            else:
                parts = self.save_dir.parts
                agent = parts[-1] if len(parts) >= 1 else "unknown"
                exp_id = parts[-2] if len(parts) >= 2 else "unknown"
                group = parts[-3] if len(parts) >= 3 else "unknown"

            manifest = {
                "generator_agent": agent,
                "experiment_id": exp_id,
                "group": group,
                "env_name": self.env_name,
                "seed": meta.get("seed"),
                "total_transitions": self.total_steps,
                "num_chunks": self.chunk_idx,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit": meta.get("git_commit"),
                "git_branch": meta.get("git_branch"),
                "git_dirty": meta.get("git_dirty"),
            }

            with open(self.save_dir / "dataset_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Notice: Could not save dataset manifest: {e}")


class DatasetReader:
    """Reads and manages offline transitions datasets saved by DatasetWriter."""

    def __init__(self, dataset_dirs, device="cpu"):
        """
        Initialize the DatasetReader.

        Args:
            dataset_dirs: Path or list of paths to dataset directories.
            device: The device to load tensors onto.
        """
        self.device = device
        self.files = []
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]

        for d in dataset_dirs:
            p = Path(d)
            if p.exists():
                self.files.extend(sorted(list(p.glob("*.pkl"))))

        if not self.files:
            print(f"Warning: No dataset files found in {dataset_dirs}")

        import pickle

        import numpy as np
        import torch

        obs_list, logic_obs_list, actions_list = [], [], []
        rewards_list, next_obs_list, next_logic_obs_list, dones_list = [], [], [], []
        has_logic = False

        for f in self.files:
            try:
                with open(f, "rb") as fh:
                    data = pickle.load(fh)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Skipping corrupted dataset chunk {f}: {e}")
                continue

            if not data:
                continue
            if not has_logic and data[0].get("logic_obs") is not None:
                has_logic = True

            obs_list.append(np.asarray([t["obs"] for t in data]))
            if has_logic:
                logic_obs_list.append(np.asarray([t["logic_obs"] for t in data]))
            actions_list.append(np.asarray([t["action"] for t in data]))
            rewards_list.append(np.asarray([t["reward"] for t in data]))
            next_obs_list.append(np.asarray([t["next_obs"] for t in data]))
            if has_logic:
                next_logic_obs_list.append(np.asarray([t["next_logic_obs"] for t in data]))
            dones_list.append(np.asarray([t["done"] for t in data]))
            del data

        if obs_list:
            # Use torch.from_numpy to share memory with NumPy concatenation without duplicating RAM
            self.obs = torch.from_numpy(np.concatenate(obs_list, axis=0))
            self.actions = torch.from_numpy(np.concatenate(actions_list, axis=0))
            self.rewards = torch.from_numpy(np.concatenate(rewards_list, axis=0))
            self.next_obs = torch.from_numpy(np.concatenate(next_obs_list, axis=0))
            self.dones = torch.from_numpy(np.concatenate(dones_list, axis=0))
            del obs_list, actions_list, rewards_list, next_obs_list, dones_list

            if has_logic:
                self.logic_obs = torch.from_numpy(np.concatenate(logic_obs_list, axis=0))
                self.next_logic_obs = torch.from_numpy(np.concatenate(next_logic_obs_list, axis=0))
                del logic_obs_list, next_logic_obs_list
            else:
                self.logic_obs = None
                self.next_logic_obs = None
        else:
            self.obs = torch.empty(0)
            self.actions = torch.empty(0)
            self.rewards = torch.empty(0)
            self.next_obs = torch.empty(0)
            self.dones = torch.empty(0)
            self.logic_obs = None
            self.next_logic_obs = None

        self.limit = len(self.obs)

    @property
    def device(self):
        return getattr(self, "_device", "cpu")

    @device.setter
    def device(self, dev):
        self._device = dev
        # For vector/tabular datasets (non-image, obs.ndim <= 2), preload directly onto GPU VRAM
        if (
            hasattr(self, "obs")
            and isinstance(self.obs, torch.Tensor)
            and self.obs.numel() > 0
            and self.obs.ndim <= 2
            and str(dev) != "cpu"
        ):
            try:
                self.obs = self.obs.to(dev, dtype=torch.float32)
                self.actions = self.actions.to(dev, dtype=torch.long)
                self.rewards = self.rewards.to(dev, dtype=torch.float32)
                self.next_obs = self.next_obs.to(dev, dtype=torch.float32)
                self.dones = self.dones.to(dev, dtype=torch.float32)
                if self.logic_obs is not None:
                    self.logic_obs = self.logic_obs.to(dev, dtype=torch.float32)
                if self.next_logic_obs is not None:
                    self.next_logic_obs = self.next_logic_obs.to(dev, dtype=torch.float32)
            except Exception as e:
                print(f"Notice: Could not preload dataset to device {dev}: {e}")

    def to(self, device):
        self.device = device
        return self

    def set_limit(self, limit):
        """Set a maximum limit on the number of transitions exposed by the dataset."""
        new_limit = min(limit, len(self.obs))
        if new_limit != self.limit:
            self.limit = new_limit
            print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size, last=False):
        """Sample a batch of transitions."""
        target_device = self.obs.device if (isinstance(self.obs, torch.Tensor) and self.obs.is_cuda) else "cpu"
        if last:
            start = max(0, self.limit - batch_size)
            idxs = torch.arange(start, self.limit, device=target_device)
        else:
            idxs = torch.randint(0, self.limit, (batch_size,), device=target_device)

        # Cast to correct types on the fly during transfer to device
        batch = {
            "obs": self.obs[idxs].to(self.device, dtype=torch.float32, non_blocking=True),
            "action": self.actions[idxs].to(self.device, dtype=torch.long, non_blocking=True),
            "reward": self.rewards[idxs].to(self.device, dtype=torch.float32, non_blocking=True),
            "next_obs": self.next_obs[idxs].to(self.device, dtype=torch.float32, non_blocking=True),
            "done": self.dones[idxs].to(self.device, dtype=torch.float32, non_blocking=True),
        }

        if self.logic_obs is not None:
            batch["logic_obs"] = self.logic_obs[idxs].to(self.device, dtype=torch.float32, non_blocking=True)
            batch["next_logic_obs"] = self.next_logic_obs[idxs].to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None

        return batch

    def get_batch(self, idxs, device=None):
        """Get a specific batch of transitions by indices."""
        if device is None:
            device = self.device
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs, dtype=torch.long)
        elif not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.long)

        target_device = self.obs.device if (isinstance(self.obs, torch.Tensor) and self.obs.is_cuda) else "cpu"
        idxs = idxs.to(target_device)

        batch = {
            "obs": self.obs[idxs].to(device, dtype=torch.float32, non_blocking=True),
            "action": self.actions[idxs].to(device, dtype=torch.long, non_blocking=True),
            "reward": self.rewards[idxs].to(device, dtype=torch.float32, non_blocking=True),
            "next_obs": self.next_obs[idxs].to(device, dtype=torch.float32, non_blocking=True),
            "done": self.dones[idxs].to(device, dtype=torch.float32, non_blocking=True),
        }

        if self.logic_obs is not None:
            batch["logic_obs"] = self.logic_obs[idxs].to(device, dtype=torch.float32, non_blocking=True)
            batch["next_logic_obs"] = self.next_logic_obs[idxs].to(device, dtype=torch.float32, non_blocking=True)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
        return batch

    def split(self, val_ratio=0.1, seed=42):
        """Split this reader into (train_reader, val_reader) by trajectory where possible."""
        n_total = len(self.obs)
        if n_total <= 1 or val_ratio <= 0.0 or val_ratio >= 1.0:
            return self, None

        rng = np.random.default_rng(seed)

        # If dones has trajectory markers, split by complete trajectories
        done_indices = torch.where(self.dones == 1.0)[0].cpu().numpy()
        if len(done_indices) > 1:
            trajectories = []
            start = 0
            for end_idx in done_indices:
                trajectories.append(list(range(start, int(end_idx) + 1)))
                start = int(end_idx) + 1
            if start < n_total:
                trajectories.append(list(range(start, n_total)))

            n_val_trajs = max(1, int(round(len(trajectories) * val_ratio)))
            shuffled_traj_indices = rng.permutation(len(trajectories))
            val_traj_indices = set(shuffled_traj_indices[:n_val_trajs])

            train_indices = []
            val_indices = []
            for t_idx, traj in enumerate(trajectories):
                if t_idx in val_traj_indices:
                    val_indices.extend(traj)
                else:
                    train_indices.extend(traj)
        else:
            raise ValueError(
                "Cannot split dataset by trajectory: insufficient trajectory 'done' markers found. Random split is disabled to prevent train/val leakage."
            )

        train_reader = DatasetReader.__new__(DatasetReader)
        train_reader._device = "cpu"
        train_reader.files = self.files
        train_reader.obs = self.obs[train_indices]
        train_reader.actions = self.actions[train_indices]
        train_reader.rewards = self.rewards[train_indices]
        train_reader.next_obs = self.next_obs[train_indices]
        train_reader.dones = self.dones[train_indices]
        train_reader.logic_obs = self.logic_obs[train_indices] if self.logic_obs is not None else None
        train_reader.next_logic_obs = self.next_logic_obs[train_indices] if self.next_logic_obs is not None else None
        train_reader.limit = len(train_reader.obs)

        val_reader = DatasetReader.__new__(DatasetReader)
        val_reader._device = "cpu"
        val_reader.files = self.files
        val_reader.obs = self.obs[val_indices]
        val_reader.actions = self.actions[val_indices]
        val_reader.rewards = self.rewards[val_indices]
        val_reader.next_obs = self.next_obs[val_indices]
        val_reader.dones = self.dones[val_indices]
        val_reader.logic_obs = self.logic_obs[val_indices] if self.logic_obs is not None else None
        val_reader.next_logic_obs = self.next_logic_obs[val_indices] if self.next_logic_obs is not None else None
        val_reader.limit = len(val_reader.obs)

        print(
            f"[Dataset Split] Train set: {len(train_reader)} transitions | Validation set: {len(val_reader)} transitions (val_ratio={val_ratio})"
        )
        return train_reader, val_reader

    def __len__(self):
        """Return the total number of transitions in the dataset."""
        return len(self.obs)


def convert_mimic_npz_to_transitions(
    npz_path: str | Path,
    out_dir: str | Path | None = None,
    chunk_size: int = 200000,
) -> int:
    """Convert a MIMIC trajectory NPZ file into chunked transition PKL files.

    Args:
        npz_path: Path to the input MIMIC .npz trajectory file.
        out_dir: Directory where chunks should be written. If None, defaults to a directory
                 named after the NPZ stem alongside the file.
        chunk_size: Maximum transitions per chunk file.

    Returns:
        Total number of transitions written.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ dataset not found at {npz_path}")

    npz_stem = npz_path.stem
    if out_dir is None:
        out_dir = npz_path.parent / npz_stem
    else:
        out_dir = Path(out_dir)

    print(f"Loading NPZ dataset from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # (N, 240, 49)
    y = data["y"]  # (N, 1)
    mask = data["mask"]  # (N, 240, 1)

    writer = DatasetWriter(save_dir=out_dir, chunk_size=chunk_size, env_name=npz_stem)

    print("Converting trajectories to transitions...")
    total_transitions = 0

    for i in range(len(X)):
        mask_patient = mask[i].squeeze()
        active_steps = np.where(mask_patient != -1)[0]
        if len(active_steps) == 0:
            continue

        T = len(active_steps)
        outcome = y[i, 0]

        for t_idx, t in enumerate(active_steps):
            obs = X[i, t, :46]
            logic_obs = np.stack([obs, obs], axis=0)

            action = int(X[i, t, 47])
            reward = (1.0 / T) if action == int(X[i, t, 47]) else 0.0
            if outcome != 0:
                reward = -reward

            if t_idx == len(active_steps) - 1:
                next_obs = obs.copy()
                next_logic_obs = logic_obs.copy()
                done = True
            else:
                t_next = active_steps[t_idx + 1]
                next_obs = X[i, t_next, :46]
                next_logic_obs = np.stack([next_obs, next_obs], axis=0)
                done = False

            writer.add(
                obs=obs,
                logic_obs=logic_obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                next_logic_obs=next_logic_obs,
                done=done,
            )
            total_transitions += 1

    writer.close()
    print(f"Successfully converted {len(X)} patients and saved {total_transitions} transitions to {out_dir}")
    return total_transitions
