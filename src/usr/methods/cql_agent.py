import math
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.usr.methods.agent_registry import register_agent
from src.usr.methods.base_agent import OfflineAgentBase


@register_agent("cql", "blendrl_cql")
class CQLAgent(OfflineAgentBase):
    """Unified Conservative Q-Learning (CQL) Offline RL Agent.

    Supports pure neural architectures (MLP, Dueling ResNet, Sepsis Transformer),
    logic-guided policies, neuro-fuzzy CEW modules, and hybrid BlendRL mixtures.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()
        self.lr = self.get_cfg("lr", 3e-4)
        self.gamma = float(self.get_cfg("gamma", 0.99))

        self._init_env(n_envs=1)
        env_name = self.get_cfg("env.name", getattr(getattr(self.cfg, "env", None), "name", "cartpole"))
        algorithm = self.get_cfg("algorithm", self.get_cfg("name", env_name))

        # Check if modular/hybrid architecture is configured
        self.is_modular = self.is_hybrid_configured()

        default_rules = self.get_cfg("rules", "default")
        default_reasoner = self.get_cfg("reasoner", "nsfr")
        default_arch = self.get_cfg("architecture", "mlp")

        if self.is_modular:
            from src.app.core.model_registry import build_model

            self.model = build_model(
                "blendrl",
                env=self.env,
                device=self.device,
                rules=self.get_cfg("rules", default_rules),
                actor_mode=self.get_cfg("actor_mode", "hybrid"),
                blender_mode=self.get_cfg("blender_mode", "neural"),
                blend_function=self.get_cfg("blend_function", "softmax"),
                reasoner=self.get_cfg("reasoner", default_reasoner),
                architecture=self.get_cfg("architecture", default_arch),
                modules=self.get_cfg("modules", None),
                cfg=self.cfg,
            )
            self.target_model = build_model(
                "blendrl",
                env=self.env,
                device=self.device,
                rules=self.get_cfg("rules", default_rules),
                actor_mode=self.get_cfg("actor_mode", "hybrid"),
                blender_mode=self.get_cfg("blender_mode", "neural"),
                blend_function=self.get_cfg("blend_function", "softmax"),
                reasoner=self.get_cfg("reasoner", default_reasoner),
                architecture=self.get_cfg("architecture", default_arch),
                modules=self.get_cfg("modules", None),
                cfg=self.cfg,
            )
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            from src.app.core.model_registry import build_model

            model_arch = self.resolve_model_name(default=default_arch)

            is_cew = (
                model_arch == "cew"
                or self.get_cfg("architecture") == "cew"
                or self.get_cfg("model") == "cew"
                or str(self.get_cfg("algorithm", "")).startswith("cew")
            )

            if is_cew:
                obs_dim = (
                    int(np.prod(self.observation_space))
                    if hasattr(self, "observation_space") and self.observation_space
                    else None
                )
                cew_cfg = self.get_cfg("cew", {}) or {}
                cew_kwargs = dict(
                    cql_alpha=float(self.get_cfg("cql_alpha", 1.0)),
                    lr=float(self.get_cfg("lr", 3e-4)),
                    ecm_dthr=float(cew_cfg.get("ecm_dthr", self.get_cfg("ecm_dthr", 0.1))),
                    eps=float(self.get_cfg("eps", 0.1)),
                    kappa=float(self.get_cfg("kappa", 0.6)),
                    fyd="fyd" in str(self.get_cfg("algorithm", ""))
                    or bool(self.get_cfg("fyd", cew_cfg.get("fyd", False))),
                    fyd_top_k=self.get_cfg("fyd_top_k", cew_cfg.get("fyd_top_k", None)),
                    stabilize=bool(self.get_cfg("stabilize", True)),
                )
                self.q_model = build_model(
                    "cew",
                    env=self.env,
                    device=self.device,
                    obs_dim=obs_dim,
                    n_actions=self.n_actions,
                    **cew_kwargs,
                )
                self.target_q_model = build_model(
                    "cew",
                    env=self.env,
                    device=self.device,
                    obs_dim=obs_dim,
                    n_actions=self.n_actions,
                    **cew_kwargs,
                )
                self.q_network = self.q_model
                self.target_q_network = self.target_q_model
            else:
                hidden_sizes = self.get_cfg("hidden_sizes", [256, 256])
                if hidden_sizes is not None:
                    hidden_sizes = list(hidden_sizes)
                from src.app.core.model_registry import build_model

                obs_dim = (
                    self.observation_space[-1]
                    if hasattr(self, "observation_space") and self.observation_space
                    else None
                )
                self.q_network = build_model(
                    model_arch,
                    env=self.env,
                    n_actions=self.n_actions,
                    device=self.device,
                    hidden_sizes=hidden_sizes,
                    obs_dim=obs_dim,
                )
                self.target_q_network = build_model(
                    model_arch,
                    env=self.env,
                    n_actions=self.n_actions,
                    device=self.device,
                    hidden_sizes=hidden_sizes,
                    obs_dim=obs_dim,
                )
                self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _prepare_logic_obs(self, obs, logic_obs=None):
        if logic_obs is not None:
            return logic_obs.to(self.device)
        if obs.ndim == 2:
            return obs.unsqueeze(1).repeat(1, 2, 1).to(self.device)
        return obs.to(self.device)

    def get_q_values(self, obs, logic_obs=None):
        """Compute discrete Q-values across all supported architectures (modular or neural)."""
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model.get_q_values(obs, logic_obs)
        elif hasattr(self, "q_model"):
            return self.q_model.get_q_values(obs)
        else:
            return self.q_network.get_q_values(obs) if hasattr(self.q_network, "get_q_values") else self.q_network(obs)

    def get_action_probs(self, obs, logic_obs=None):
        """Action probabilities for discrete CQL: softmax over Q-values."""
        if self.is_modular and self.get_cfg("use_actor", False) and hasattr(self.model, "actor"):
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            probs, _ = self.model.actor(obs, logic_obs)
            return probs
        q_vals = self.get_q_values(obs, logic_obs)
        return torch.softmax(q_vals, dim=-1)

    def get_action(self, obs, logic_obs=None):
        """Greedy action selection for discrete CQL: argmax over Q-values."""
        if self.is_modular and self.get_cfg("use_actor", False) and hasattr(self.model, "actor"):
            probs = self.get_action_probs(obs, logic_obs)
            return torch.argmax(probs, dim=-1)
        q_vals = self.get_q_values(obs, logic_obs)
        return torch.argmax(q_vals, dim=-1)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            if self.get_cfg("use_actor", False):
                return self.model(obs, logic_obs, action=action)
            q_vals = self.model.get_q_values(obs, logic_obs)
        else:
            q_vals = (
                self.q_network.get_q_values(obs) if hasattr(self.q_network, "get_q_values") else self.q_network(obs)
            )
        probs = torch.softmax(q_vals, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = torch.argmax(q_vals, dim=-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = q_vals.max(dim=-1)[0]
        return action, logprob, entropy, value

    def get_value(self, obs, logic_obs=None):
        if self.is_modular and self.get_cfg("use_actor", False):
            logic_obs = self._prepare_logic_obs(obs, logic_obs)
            return self.model.get_value(obs, logic_obs)
        q_vals = self.get_q_values(obs, logic_obs)
        return q_vals.max(dim=-1)[0]

    def setup(self, stage: str | None = None):
        super().setup(stage)

    def on_train_start(self):
        if hasattr(self.trainer.datamodule, "reader") and self.trainer.datamodule.reader is not None:
            self.trainer.datamodule.reader.device = self.device
        if hasattr(self.trainer.datamodule, "val_reader") and self.trainer.datamodule.val_reader is not None:
            self.trainer.datamodule.val_reader.device = self.device

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        cfg = self.cfg

        if isinstance(batch, dict) and "obs" in batch:
            real_batch = batch
        elif datamodule is not None and getattr(datamodule, "reader", None) is not None:
            if isinstance(batch, torch.Tensor):
                real_batch = datamodule.reader.get_batch(batch, device=self.device)
            else:
                batch_size = self.get_cfg("batch_size", 256)
                real_batch = datamodule.reader.sample(batch_size)
        else:
            raise RuntimeError("CQLAgent requires an active offline dataset reader or batched dictionary.")

        obs = real_batch["obs"].to(self.device, non_blocking=True)
        actions = real_batch["action"].to(self.device, non_blocking=True)
        rewards = real_batch["reward"].to(self.device, non_blocking=True)
        reward_scale = float(self.get_cfg("reward_scale", 1.0))
        if reward_scale != 1.0:
            rewards = rewards * reward_scale

        next_obs = real_batch["next_obs"].to(self.device, non_blocking=True)
        dones = real_batch["done"].to(self.device, non_blocking=True)

        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        gamma = float(self.get_cfg("gamma", getattr(self.cfg.env, "gamma", 0.99)))
        bellman_loss_fn = str(self.get_cfg("bellman_loss", "smooth_l1")).lower()
        pos_action_weight = float(self.get_cfg("pos_action_weight", 1.0))
        if pos_action_weight != 1.0:
            weights = torch.where(
                actions == 1,
                torch.as_tensor(pos_action_weight, device=self.device, dtype=torch.float32),
                torch.ones_like(actions, dtype=torch.float32, device=self.device),
            )
            weights = weights / (weights.mean() + 1e-8)
        else:
            weights = None

        if self.is_modular:
            logic_obs = self._prepare_logic_obs(obs, real_batch.get("logic_obs"))
            next_logic_obs = self._prepare_logic_obs(next_obs, real_batch.get("next_logic_obs"))

            with torch.no_grad():
                online_next_q = self.model.get_q_values(next_obs, next_logic_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_model.get_q_values(next_obs, next_logic_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)

            all_q_values = self.model.get_q_values(obs, logic_obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            if bellman_loss_fn in ["smooth_l1", "huber"]:
                if weights is not None:
                    bellman_loss = (F.smooth_l1_loss(q_action, q_target, beta=1.0, reduction="none") * weights).mean()
                else:
                    bellman_loss = F.smooth_l1_loss(q_action, q_target, beta=1.0)
            else:
                if weights is not None:
                    bellman_loss = (F.mse_loss(q_action, q_target, reduction="none") * weights).mean()
                else:
                    bellman_loss = F.mse_loss(q_action, q_target)
            cql_diff = torch.logsumexp(all_q_values, dim=1) - q_action
            if weights is not None:
                cql_loss = (cql_diff * weights).mean()
            else:
                cql_loss = cql_diff.mean()
            q_loss = bellman_loss + cql_alpha * cql_loss

            use_actor = bool(self.get_cfg("use_actor", False))
            if use_actor:
                probs, weights_act = self.model.actor(obs, logic_obs)
                log_probs = torch.log(probs + 1e-12)
                entropy = -(probs * log_probs).sum(dim=1)
                blend_entropy = (
                    -(weights_act * torch.log(weights_act + 1e-12)).sum(dim=1) if weights_act is not None else None
                )
                ent_coef = self.get_cfg("ent_coef", 0.01)
                blend_ent_coef = self.get_cfg("blend_ent_coef", 0.01)
                blend_entropy_loss = blend_entropy.mean() if isinstance(blend_entropy, torch.Tensor) else 0.0

                actor_obj = (probs * all_q_values.detach()).sum(dim=1).mean()
                actor_loss = -actor_obj - ent_coef * entropy.mean() - blend_ent_coef * blend_entropy_loss
                total_loss = q_loss + actor_loss
            else:
                actor_loss = torch.tensor(0.0, device=self.device)
                blend_entropy = None
                total_loss = q_loss

            opt = getattr(self, "opt", self.optimizers())
            if isinstance(opt, list):
                opt = opt[0]
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()

            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.model, self.target_model, tau=soft_target_tau)
        else:
            blend_entropy = None
            opt = getattr(self, "opt", self.optimizers())
            if isinstance(opt, list):
                opt = opt[0]
            with torch.no_grad():
                online_next_q = self.q_network(next_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_q_network(next_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)

            all_q_values = self.q_network(obs)
            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            if bellman_loss_fn in ["smooth_l1", "huber"]:
                if weights is not None:
                    bellman_loss = (F.smooth_l1_loss(q_action, q_target, beta=1.0, reduction="none") * weights).mean()
                else:
                    bellman_loss = F.smooth_l1_loss(q_action, q_target, beta=1.0)
            else:
                if weights is not None:
                    bellman_loss = (F.mse_loss(q_action, q_target, reduction="none") * weights).mean()
                else:
                    bellman_loss = F.mse_loss(q_action, q_target)
            cql_diff = torch.logsumexp(all_q_values, dim=1) - q_action
            if weights is not None:
                cql_loss = (cql_diff * weights).mean()
            else:
                cql_loss = cql_diff.mean()
            q_loss = bellman_loss + cql_alpha * cql_loss

            opt.zero_grad()
            self.manual_backward(q_loss)
            opt.step()

            soft_target_tau = self.get_cfg("soft_target_tau", 0.005)
            self._soft_update(self.q_network, self.target_q_network, tau=soft_target_tau)
            actor_loss = 0.0

        self._log_offline_transitions()
        log_data = {
            "losses/total_loss": (q_loss + actor_loss).item()
            if isinstance(q_loss + actor_loss, torch.Tensor)
            else q_loss + actor_loss,
            "losses/q_loss": q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss,
            "losses/bellman_loss": bellman_loss.item() if isinstance(bellman_loss, torch.Tensor) else bellman_loss,
            "losses/cql_loss": cql_loss.item() if isinstance(cql_loss, torch.Tensor) else cql_loss,
        }
        if bool(self.get_cfg("use_actor", False)):
            log_data["losses/actor_loss"] = actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            log_data["losses/blend_entropy"] = blend_entropy.mean().item()
        self.log_dict(log_data)

    def on_validation_epoch_start(self):
        self._val_step_losses = []

    def validation_step(self, batch, batch_idx):
        datamodule = getattr(self.trainer, "datamodule", None)
        if isinstance(batch, dict) and "obs" in batch:
            val_batch = batch
        elif datamodule is not None and getattr(datamodule, "val_reader", None) is not None:
            val_batch = datamodule.val_reader.get_batch(batch, device=self.device)
        else:
            return

        obs = val_batch["obs"].to(self.device, non_blocking=True)
        actions = val_batch["action"].to(self.device, non_blocking=True)
        rewards = val_batch["reward"].to(self.device, non_blocking=True)
        reward_scale = float(self.get_cfg("reward_scale", 1.0))
        if reward_scale != 1.0:
            rewards = rewards * reward_scale

        next_obs = val_batch["next_obs"].to(self.device, non_blocking=True)
        dones = val_batch["done"].to(self.device, non_blocking=True)
        cql_alpha = self.get_cfg("cql_alpha", 1.0)
        gamma = float(self.get_cfg("gamma", getattr(self.cfg.env, "gamma", 0.99)))
        bellman_loss_fn = str(self.get_cfg("bellman_loss", "smooth_l1")).lower()
        pos_action_weight = float(self.get_cfg("pos_action_weight", 1.0))
        if pos_action_weight != 1.0:
            val_weights = torch.where(
                actions == 1,
                torch.as_tensor(pos_action_weight, device=self.device, dtype=torch.float32),
                torch.ones_like(actions, dtype=torch.float32, device=self.device),
            )
            val_weights = val_weights / (val_weights.mean() + 1e-8)
        else:
            val_weights = None

        with torch.no_grad():
            if self.is_modular:
                logic_obs = self._prepare_logic_obs(obs, val_batch.get("logic_obs"))
                next_logic_obs = self._prepare_logic_obs(next_obs, val_batch.get("next_logic_obs"))
                online_next_q = self.model.get_q_values(next_obs, next_logic_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_model.get_q_values(next_obs, next_logic_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)
                all_q_values = self.model.get_q_values(obs, logic_obs)
                if self.get_cfg("use_actor", False):
                    probs, _ = self.model.actor(obs, logic_obs)
                    pred_acts = torch.argmax(probs, dim=-1)
                else:
                    pred_acts = torch.argmax(all_q_values, dim=-1)
            else:
                online_next_q = self.q_network(next_obs)
                best_next_action = torch.argmax(online_next_q, dim=1, keepdim=True)
                next_v = self.target_q_network(next_obs).gather(1, best_next_action).squeeze(1)
                q_target = rewards + gamma * next_v * (1 - dones)
                all_q_values = self.q_network(obs)
                pred_acts = torch.argmax(all_q_values, dim=-1)

            q_action = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            if bellman_loss_fn in ["smooth_l1", "huber"]:
                if val_weights is not None:
                    bellman_loss = (
                        F.smooth_l1_loss(q_action, q_target, beta=1.0, reduction="none") * val_weights
                    ).mean()
                else:
                    bellman_loss = F.smooth_l1_loss(q_action, q_target, beta=1.0)
            else:
                if val_weights is not None:
                    bellman_loss = (F.mse_loss(q_action, q_target, reduction="none") * val_weights).mean()
                else:
                    bellman_loss = F.mse_loss(q_action, q_target)
            logsumexp_qvalues = torch.logsumexp(all_q_values, dim=1)
            if val_weights is not None:
                cql_loss = ((logsumexp_qvalues - q_action) * val_weights).mean()
            else:
                cql_loss = (logsumexp_qvalues - q_action).mean()
            val_loss = bellman_loss + cql_alpha * cql_loss
            admin_rate = (pred_acts == 1).float().mean()
            tp = ((pred_acts == 1) & (actions == 1)).sum().float()
            fp = ((pred_acts == 1) & (actions == 0)).sum().float()
            fn = ((pred_acts == 0) & (actions == 1)).sum().float()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            val_f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

        self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/bellman_loss", bellman_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/cql_loss", cql_loss, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/q_mean", all_q_values.mean(), prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/admin_rate", admin_rate, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/f1", val_f1, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/precision", prec, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val/recall", rec, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        if hasattr(self, "_val_step_losses"):
            self._val_step_losses.append(val_loss.detach())
        return val_loss

    def on_validation_epoch_end(self):
        if hasattr(self, "_val_step_losses") and len(self._val_step_losses) > 0:
            losses = torch.stack(self._val_step_losses)
            mean_loss = losses.mean()
            std_loss = losses.std() if len(losses) > 1 else torch.tensor(0.0, device=losses.device)
            robust_loss = mean_loss + 3.0 * std_loss
            self.log("val/robust_loss", robust_loss, prog_bar=True, sync_dist=True)
            self.log("val/loss_std", std_loss, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        weight_decay = float(self.get_cfg("weight_decay", 0.0))
        lr = float(self.get_cfg("lr", 3e-4))
        logic_lr = float(self.get_cfg("logic_lr", lr))
        blender_lr = float(self.get_cfg("blender_lr", lr))
        if self.is_modular:
            param_groups = []
            used_param_ids = set()

            # 1. Logic modules (NSFR / reasoner)
            if hasattr(self.model, "policy_modules"):
                for m, m_type in zip(self.model.policy_modules, self.model.module_types):
                    if m_type != "neural":
                        m_params = [p for p in m.parameters() if p.requires_grad and id(p) not in used_param_ids]
                        if m_params:
                            for p in m_params:
                                used_param_ids.add(id(p))
                            param_groups.append({"params": m_params, "lr": logic_lr, "weight_decay": weight_decay})

            # 2. Blender module
            if hasattr(self.model, "blender") and self.model.blender is not None:
                b_params = [
                    p for p in self.model.blender.parameters() if p.requires_grad and id(p) not in used_param_ids
                ]
                if b_params:
                    for p in b_params:
                        used_param_ids.add(id(p))
                    param_groups.append({"params": b_params, "lr": blender_lr, "weight_decay": weight_decay})

            # 3. All remaining parameters (neural actor, Q-networks, critic)
            remaining_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in used_param_ids]
            if remaining_params:
                param_groups.append({"params": remaining_params, "lr": lr, "weight_decay": weight_decay})

            if not param_groups:
                return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return optim.Adam(param_groups)
        elif hasattr(self, "q_model"):
            params = [p for p in self.q_model.parameters() if p.requires_grad]
            if not params:
                params = [torch.zeros(1, requires_grad=True)]
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        params = [p for p in self.q_network.parameters() if p.requires_grad]
        if not params:
            params = [torch.zeros(1, requires_grad=True)]
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        for attr in ("q_model", "model"):
            m = getattr(self, attr, None)
            if hasattr(m, "extra_state"):
                state = m.extra_state()
                checkpoint["extra_state"] = state
                checkpoint["cew_extra"] = state

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        extra = checkpoint.get("extra_state", checkpoint.get("cew_extra", {}))
        for attr in ("q_model", "model"):
            m = getattr(self, attr, None)
            if hasattr(m, "load_extra_state") and extra:
                m.load_extra_state(extra)

        if hasattr(self, "target_q_model") and self.target_q_model is not None and hasattr(self, "q_model"):
            if hasattr(self.q_model, "clone_topology_to"):
                self.q_model.clone_topology_to(self.target_q_model)
            elif hasattr(self.target_q_model, "clone_topology_from"):
                self.target_q_model.clone_topology_from(self.q_model)
