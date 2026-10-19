"""Periodic checkpoints during training, set by the experiment with one of two keys:

    checkpoint_every_steps: 25000   # save each time training passes another 25,000 transitions
    num_checkpoints: 10             # or: this many, spread evenly over total_timesteps

The last checkpoint of either schedule falls at total_timesteps, so the trained model is always among them.
Neither key set: no periodic checkpoints (best_model and the final save still happen as before).
Only transition-counted training (online RL) takes them; epoch-based paradigms use eval_every_n_epochs.
The Lightning callback that saves them is PeriodicCheckpointCallback in callbacks.py: this module stays
free of Lightning and torch because the GUI imports it (through validation) at startup.
"""

from __future__ import annotations

import math
from typing import Any

EVERY_KEY = "checkpoint_every_steps"
COUNT_KEY = "num_checkpoints"
KEYS = (EVERY_KEY, COUNT_KEY)
TRANSITION_PARADIGMS = ("online_rl",)


def _get(cfg: Any, key: str) -> Any:
    return cfg.get(key, None) if hasattr(cfg, "get") else getattr(cfg, key, None)


def checkpoint_errors(cfg: Any) -> list[str]:
    """What is wrong with the experiment's checkpoint keys, if anything."""
    set_keys = [key for key in KEYS if _get(cfg, key) is not None]
    if not set_keys:
        return []
    if len(set_keys) == 2:
        return [f"Set either '{EVERY_KEY}' or '{COUNT_KEY}', not both."]
    key = set_keys[0]
    value = _get(cfg, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"'{key}' must be a whole number of at least 1 (got {value!r})."]
    paradigm = _get(cfg, "paradigm")
    if paradigm not in TRANSITION_PARADIGMS:
        return [
            f"'{key}' counts transitions, which paradigm '{paradigm}' does not train by. "
            f"Use 'eval_every_n_epochs' to checkpoint by epoch instead."
        ]
    return []


def checkpoint_interval(cfg: Any) -> int | None:
    """Transitions between periodic checkpoints, or None when the experiment asks for none."""
    every, count = _get(cfg, EVERY_KEY), _get(cfg, COUNT_KEY)
    if every:
        return int(every)
    if count:
        total = int(_get(cfg, "total_timesteps") or 0)
        return max(1, math.ceil(total / int(count))) if total else None
    return None


class CheckpointSchedule:
    """Says when a checkpoint is due as the transition count grows."""

    def __init__(self, interval: int, total: int | None):
        self.interval, self.total = interval, total
        self.next = self._capped(interval)

    def _capped(self, target: int) -> int:
        return min(target, self.total) if self.total else target

    def due(self, transitions: int) -> bool:
        """True once per crossed threshold (several crossed at once still give one checkpoint)."""
        if transitions < self.next:
            return False
        reached_total = bool(self.total) and transitions >= self.total
        self.next = math.inf if reached_total else self._capped((transitions // self.interval + 1) * self.interval)
        return True


def checkpoint_name(transitions: int) -> str:
    return f"step_{transitions:09d}"
