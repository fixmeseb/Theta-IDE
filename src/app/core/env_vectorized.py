"""Vectorized environment base wrapper for NeSyRL benchmark environments."""

from abc import ABC
from collections.abc import Sequence
from typing import Any

import torch

from src.app.core.utils import load_module
from src.app.pipeline import runtime


class VectorizedBaseEnv(ABC):
    """Base abstract class for vectorized benchmark environments.

    Adheres to the standard Gymnasium vectorized environment protocol:
    - reset() -> obs tensor of shape (n_envs, *obs_shape)
    - step(actions) -> (next_obs, rewards, terminations, truncations, infos)
    """

    name: str
    pred2action: dict[str, int] = {}
    n_actions: int

    def __init__(self, mode: str = "ppo"):
        self.mode = mode

    def reset(self, seed=None) -> torch.Tensor:
        """Reset all environments and return batched observations.

        Returns:
            Observation tensor of shape (n_envs, *obs_shape).
        """
        raise NotImplementedError

    def step(self, actions) -> tuple[torch.Tensor, Any, Any, Any, Any]:
        """Step all environments with batched actions.

        Returns:
            Tuple of (next_obs, rewards, terminations, truncations, infos).
        """
        raise NotImplementedError

    def get_action_meanings(self) -> Sequence[str]:
        return list(self.pred2action.keys())

    def n_actions(self) -> int:
        return len(list(set(self.pred2action.items())))

    @staticmethod
    def from_name(name: str, **kwargs):
        """Factory to load environment by name from in/envs/<name>/env_vectorized.py."""
        env_path = f"{runtime.ENVS_DIR}/{name}/env_vectorized.py"
        env_module = load_module(env_path)
        cls = getattr(env_module, "VectorizedEnv", None) or getattr(env_module, "VectorizedNudgeEnv")
        return cls(**kwargs)

    def close(self):
        if hasattr(self, "env") and hasattr(self.env, "close"):
            self.env.close()
        elif hasattr(self, "envs"):
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()
        elif hasattr(self, "venv") and hasattr(self.venv, "close"):
            self.venv.close()


# Backward-compatibility alias
VectorizedNudgeBaseEnv = VectorizedBaseEnv
