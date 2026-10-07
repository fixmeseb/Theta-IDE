"""
Base agent classes — shared interface and utilities for all RL agents.

Hierarchy:
    BaseAgent (ABC)
    ├── OnlineAgentBase   — rollout buffers, env stepping, GAE, dataset writing
    └── OfflineAgentBase  — interval limit calc, reader.sample(), target networks
"""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig


class BaseAgent(L.LightningModule, ABC):
    """Abstract base class for all RL agents in the BlendRL framework.

    Provides:
        - Unified config traversal (get_cfg)
        - Soft target network updates (_soft_update)
        - Standard interface contract via abstract methods
        - Common environment initialization helpers
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, dict) and not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.automatic_optimization = False

    # ──────────────────────────────────────────────
    # Config Utilities
    # ──────────────────────────────────────────────

    def get_cfg(self, key, default=None):
        """Unified config traversal — searches agent, env, then top-level config.

        Handles Hydra's nested DictConfig structures, dotted keys (e.g. 'cql.cql_alpha'),
        and sub-dictionaries (e.g., cfg.agent.cew, cfg.agent.cql, cfg.agent.blendrl).
        """
        cfg = self.cfg

        def _get_nested(root, k):
            if not isinstance(root, (dict, DictConfig)):
                return None, False
            if k in root:
                return root[k], True
            if "." in k:
                parts = k.split(".")
                curr = root
                for p in parts:
                    if isinstance(curr, (dict, DictConfig)) and p in curr:
                        curr = curr[p]
                    else:
                        return None, False
                return curr, True
            return None, False

        # Search in agent config
        if hasattr(cfg, "agent"):
            val, found = _get_nested(cfg.agent, key)
            if found:
                return val
            # Handle double-nested agent config from Hydra inheritance
            if "agent" in cfg.agent and isinstance(cfg.agent.agent, (dict, DictConfig)):
                val, found = _get_nested(cfg.agent.agent, key)
                if found:
                    return val
            # Search sub-dictionaries in cfg.agent (e.g., cfg.agent.cew, cfg.agent.cql, cfg.agent.blendrl)
            for sub_k, sub_v in cfg.agent.items():
                if isinstance(sub_v, (dict, DictConfig)):
                    val, found = _get_nested(sub_v, key)
                    if found:
                        return val

        # Search in model config
        if hasattr(cfg, "model"):
            val, found = _get_nested(cfg.model, key)
            if found:
                return val
            if isinstance(cfg.model, (dict, DictConfig)):
                for sub_k, sub_v in cfg.model.items():
                    if isinstance(sub_v, (dict, DictConfig)):
                        val, found = _get_nested(sub_v, key)
                        if found:
                            return val

        # Search in env config
        if hasattr(cfg, "env"):
            val, found = _get_nested(cfg.env, key)
            if found:
                return val

        # Search in top-level config
        val, found = _get_nested(cfg, key)
        if found:
            return val

        return default

    def is_hybrid_configured(self) -> bool:
        """Determine whether a hybrid / symbolic-neural BlendRL policy is configured.

        Checks:
        1. Explicit actor_mode ('hybrid', 'logic')
        2. Configured modules list
        3. Model specification is 'blendrl', 'blender', or 'hybrid'
        4. Algorithm or agent name contains 'blendrl'
        """
        if self.get_cfg("actor_mode", "neural") in ("hybrid", "logic"):
            return True
        if bool(self.get_cfg("modules", [])):
            return True

        # Check model config (Tier 3 Composite Model specification)
        model_cfg = getattr(self.cfg, "model", None)
        if isinstance(model_cfg, str) and model_cfg.strip().lower() in ("blendrl", "blender", "hybrid"):
            return True
        if isinstance(model_cfg, (dict, DictConfig)):
            if "blendrl" in model_cfg or "blender" in model_cfg:
                return True
            name = model_cfg.get("name") or model_cfg.get("architecture") or model_cfg.get("type")
            if name and str(name).strip().lower() in ("blendrl", "blender", "hybrid"):
                return True

        # Check algorithm / agent name
        algo = self.get_cfg("algorithm", self.get_cfg("name", ""))
        if "blendrl" in str(algo).lower():
            return True
        agent_cfg = getattr(self.cfg, "agent", None)
        if isinstance(agent_cfg, str) and "blendrl" in agent_cfg.lower():
            return True

        return False

    def resolve_model_name(self, default: str = "mlp") -> str:
        """Resolve the model architecture name from config across all paradigms."""
        model_cfg = getattr(self.cfg, "model", None)
        if isinstance(model_cfg, str) and model_cfg.strip():
            name = model_cfg.strip()
            if name.lower() in ("blendrl", "blender", "hybrid"):
                arch = self.get_cfg("architecture", None)
                if arch and isinstance(arch, str) and arch.strip().lower() not in ("blendrl", "blender", "hybrid"):
                    return arch.strip()
                return default
            return name
        if isinstance(model_cfg, (dict, DictConfig)):
            # If composite model like model: {blendrl: {neural: dueling_resnet, ...}}
            blendrl_cfg = model_cfg.get("blendrl", model_cfg.get("blender", None))
            if isinstance(blendrl_cfg, (dict, DictConfig)):
                neural_arch = blendrl_cfg.get("neural", None)
                if neural_arch:
                    if isinstance(neural_arch, (dict, DictConfig)):
                        keys = [k for k in neural_arch.keys() if not str(k).startswith("_")]
                        if keys:
                            return str(keys[0]).strip()
                    return str(neural_arch).strip()
            name = model_cfg.get("architecture") or model_cfg.get("name") or model_cfg.get("type")
            if name:
                name_str = str(name).strip()
                if name_str.lower() in ("blendrl", "blender", "hybrid"):
                    return default
                return name_str
            # If wrapped under a single key, e.g. {dueling_resnet: {...}}
            keys = [k for k in model_cfg.keys() if not str(k).startswith("_")]
            if len(keys) == 1 and isinstance(model_cfg[keys[0]], (dict, DictConfig)):
                k = str(keys[0]).strip()
                if k.lower() in ("blendrl", "blender", "hybrid"):
                    return default
                return k

        arch = self.get_cfg("architecture", None)
        if arch and isinstance(arch, str):
            arch_str = arch.strip()
            if arch_str.lower() not in ("blendrl", "blender", "hybrid"):
                return arch_str
        return default

    # ──────────────────────────────────────────────
    # Network Utilities
    # ──────────────────────────────────────────────

    def _soft_update(self, model, target_model, tau: float | None = None):
        """Polyak averaging for target network updates."""
        if tau is None:
            tau = self.get_cfg("soft_target_tau", 0.005)
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # ──────────────────────────────────────────────
    # Environment Helpers
    # ──────────────────────────────────────────────

    def _init_env(self, n_envs=None):
        """Initialize the vectorized environment and extract observation/action spaces.

        Args:
            n_envs: Number of parallel environments. Defaults to cfg value for online,
                    1 for offline (evaluation only).

        Returns:
            Observation tensor from env.reset().
        """
        from src.app.core.env_vectorized import VectorizedBaseEnv

        if n_envs is None:
            n_envs = self.get_cfg("num_envs", 4)

        env_name = self.get_cfg("env.name", getattr(getattr(self.cfg, "env", None), "name", None) or "cartpole")
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", env_name))

        self.env = VectorizedBaseEnv.from_name(
            env_name, n_envs=n_envs, mode=algorithm, seed=self.get_cfg("seed", getattr(self.cfg, "seed", 1))
        )

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.observation_space = obs.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()

        return obs

    # ──────────────────────────────────────────────
    # Abstract Interface (enforced contract)
    # ──────────────────────────────────────────────

    @abstractmethod
    def get_action_and_value(self, obs, logic_obs=None, action=None):
        """Compute action, log probability, entropy, and value for given observations.

        Returns:
            Tuple of (action, logprob, entropy, value) — or with blend_entropy for hybrid agents.
        """
        ...

    @abstractmethod
    def get_value(self, obs, logic_obs=None):
        """Compute value estimate for given observations."""
        ...

    def get_action_probs(self, obs, logic_obs=None):
        """Compute action probabilities according to the agent's policy paradigm.

        Subclasses should override this method to define their canonical policy distribution.
        """
        if hasattr(self, "actor") and hasattr(self.actor, "get_action_probs"):
            return self.actor.get_action_probs(obs)
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_action_probs.")

    def get_action(self, obs, logic_obs=None):
        """Select discrete action for given observation (default: argmax of action probs)."""
        probs = self.get_action_probs(obs, logic_obs)
        return torch.argmax(probs, dim=-1)

    def get_blending_weights(self, obs: torch.Tensor, logic_obs: torch.Tensor | None = None) -> torch.Tensor | None:
        """Return blending weights for hybrid/modular architectures if applicable, else None."""
        if (
            hasattr(self, "model")
            and hasattr(self.model, "actor")
            and hasattr(self.model.actor, "to_blender_policy_distribution")
        ):
            if getattr(self, "is_modular", False) and hasattr(self, "_prepare_logic_obs"):
                logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model.actor.to_blender_policy_distribution(obs, logic_obs)
        return None


class OfflineAgentBase(BaseAgent):
    """Base class for all offline RL agents (IQL, CQL, CEW, and their BlendRL variants).

    Provides:
        - Device transfer for train and validation readers
        - Interval-based dataset limit management (on_train_epoch_start)
        - Common offline training epoch tracking
    """

    def on_train_start(self):
        """Preload datasets to agent device on training start."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            if hasattr(datamodule, "reader") and datamodule.reader is not None:
                datamodule.reader.device = self.device
            if hasattr(datamodule, "val_reader") and datamodule.val_reader is not None:
                datamodule.val_reader.device = self.device

    def configure_callbacks(self):
        """Attach model-registered callbacks to the trainer.

        Discovers and instantiates callbacks requested by any model or sub-module
        implementing HasModelCallbacks (e.g. CEWModel or BlenderActorCritic).
        """
        from src.app.core.protocols import HasModelCallbacks, walk_model_modules

        callbacks = []
        for m in walk_model_modules(self):
            if isinstance(m, HasModelCallbacks) or hasattr(m, "get_callbacks"):
                callbacks.extend(m.get_callbacks())

        # Deduplicate callbacks by class
        seen_types = set()
        unique_callbacks = []
        for cb in callbacks:
            cb_type = type(cb)
            if cb_type not in seen_types:
                seen_types.add(cb_type)
                unique_callbacks.append(cb)
        return unique_callbacks

    def on_train_epoch_start(self):
        """Set dataset limit based on current training interval.

        Implements the progressive data exposure schedule defined by
        intervals_count and epochs_per_interval (when intervals_count > 1).
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "reader") and datamodule.reader is not None:
            is_offline_only = getattr(self.cfg.env, "offline_only", False)
            intervals_count = 1 if is_offline_only else self.cfg.get("intervals_count", 1)
            if intervals_count > 1:
                epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
                current_interval = self.current_epoch // epochs_per_interval
                interval_size = self.cfg.total_timesteps // intervals_count
                current_limit = interval_size * (current_interval + 1)
                datamodule.reader.set_limit(min(current_limit, len(datamodule.reader)))
            else:
                datamodule.reader.set_limit(len(datamodule.reader))
        self._handle_optimizer_rebind()

    def _log_offline_transitions(self):
        """Calculate and log the current transition count for offline training."""
        cfg = self.cfg
        is_offline_only = getattr(cfg.env, "offline_only", False)
        intervals_count = 1 if is_offline_only else cfg.get("intervals_count", 1)
        if intervals_count > 1:
            epochs_per_interval = self.get_cfg("epochs_per_interval", 1)
            current_interval = self.current_epoch // epochs_per_interval
            interval_size = cfg.total_timesteps // intervals_count
            current_transitions = interval_size * (current_interval + 1)
        else:
            current_transitions = (
                cfg.total_timesteps
                if hasattr(cfg, "total_timesteps") and isinstance(cfg.total_timesteps, (int, float))
                else len(self.trainer.datamodule.reader)
            )
        self.log("transitions", float(current_transitions), logger=False, prog_bar=True)
        return current_transitions

    def _handle_optimizer_rebind(self) -> None:
        """Check all model components for dynamic topology changes.

        Any model (standalone or inside a composite model) that implements
        DynamicTopologyProtocol (or sets _request_optimizer_rebind = True)
        triggers optimizer rebinding and target-network synchronization here.
        """
        from src.app.core.protocols import walk_model_modules

        candidates = walk_model_modules(self)
        needs_rebind = False
        for m in candidates:
            if hasattr(m, "has_topology_changed") and m.has_topology_changed():
                needs_rebind = True
                break
            elif getattr(m, "_request_optimizer_rebind", False):
                needs_rebind = True
                break

        if not needs_rebind:
            return

        # Clear flag on all candidates
        for m in candidates:
            if hasattr(m, "reset_topology_changed"):
                m.reset_topology_changed()
            elif getattr(m, "_request_optimizer_rebind", False):
                m._request_optimizer_rebind = False

        self._rebind_optimizer()

    def _rebind_optimizer(self) -> None:
        """Rebind optimizer after a model topology change and sync target networks."""
        # Sync target network if present and topology-aware
        for src_attr, tgt_attr in (
            ("q_model", "target_q_model"),
            ("q_network", "target_q_network"),
            ("model", "target_model"),
        ):
            src = getattr(self, src_attr, None)
            tgt = getattr(self, tgt_attr, None)
            if src is not None and tgt is not None and src is not tgt:
                if hasattr(src, "clone_topology_to"):
                    src.clone_topology_to(tgt)
                elif hasattr(tgt, "clone_topology_from"):
                    tgt.clone_topology_from(src)

        # Sync any constituent target modules (e.g. inside BlenderActorCritic)
        src_blender = getattr(self, "model", None)
        tgt_blender = getattr(self, "target_model", None)
        if src_blender is not None and tgt_blender is not None and src_blender is not tgt_blender:
            if hasattr(src_blender, "clone_topology_to"):
                src_blender.clone_topology_to(tgt_blender)
            else:
                src_modules = list(getattr(src_blender, "policy_modules", None) or [])
                tgt_modules = list(getattr(tgt_blender, "policy_modules", None) or [])
                for s, t in zip(src_modules, tgt_modules):
                    if hasattr(s, "clone_topology_to"):
                        s.clone_topology_to(t)
                    elif hasattr(t, "clone_topology_from"):
                        t.clone_topology_from(s)

        # Rebind optimizer
        try:
            new_opt = self.configure_optimizers()
            if isinstance(new_opt, list):
                opts = new_opt
            else:
                opts = [new_opt]
            strategy_opts = getattr(getattr(self, "trainer", None), "strategy", None)
            if strategy_opts is not None and hasattr(strategy_opts, "optimizers"):
                for i, opt in enumerate(opts):
                    if i < len(strategy_opts.optimizers):
                        strategy_opts.optimizers[i] = opt
            if hasattr(self, "opt"):
                self.opt = opts[0] if opts else self.opt
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("_rebind_optimizer failed: %s", e)
