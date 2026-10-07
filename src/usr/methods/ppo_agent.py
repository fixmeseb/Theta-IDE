import os
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from src.usr.methods.agent_registry import register_agent
from src.usr.methods.base_agent import BaseAgent


@register_agent("ppo", "blendrl")
class PPOAgent(BaseAgent):
    """Unified Proximal Policy Optimization (PPO) Online RL Agent.

    Supports pure neural actor-critic baselines as well as hybrid modular BlendRL policies.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.save_hyperparameters()

        self.lr = self.get_cfg("lr", 3e-4)
        self.logic_lr = self.get_cfg("logic_lr", self.lr)
        self.blender_lr = self.get_cfg("blender_lr", self.lr)

        self.num_envs = self.get_cfg("num_envs", 4)
        self.num_steps = self.get_cfg("num_steps", 128)
        self.update_epochs = self.get_cfg("update_epochs", self.get_cfg("ppo_epochs", 10))
        self.batch_size = self.get_cfg("batch_size", 64)
        self.num_minibatches = self.get_cfg("num_minibatches", None)
        if self.num_minibatches is not None:
            self.batch_size = (self.num_envs * self.num_steps) // self.num_minibatches

        self.gamma = float(self.get_cfg("gamma", 0.99))
        self.gae_lambda = float(self.get_cfg("gae_lambda", 0.95))
        self.clip_coef = self.get_cfg("clip_coef", 0.2)
        self.ent_coef = self.get_cfg("ent_coef", 0.01)
        self.blend_ent_coef = self.get_cfg("blend_ent_coef", 0.01)
        self.vf_coef = self.get_cfg("vf_coef", 0.5)
        self.max_grad_norm = self.get_cfg("max_grad_norm", 0.5)
        self.anneal_lr = self.get_cfg("anneal_lr", True)
        self.norm_adv = self.get_cfg("norm_adv", True)
        self.clip_vloss = self.get_cfg("clip_vloss", True)

        self._init_env(n_envs=self.num_envs)

        self.dataset_writer = None
        if self.get_cfg("save_dataset", False) and self.get_cfg("dataset_path", None):
            from src.app.dataset_utils import DatasetWriter

            env_name = self.get_cfg("env.name", getattr(getattr(self.cfg, "env", None), "name", "env"))
            self.dataset_writer = DatasetWriter(save_dir=self.get_cfg("dataset_path"), env_name=env_name, cfg=self.cfg)

        # Check if modular/hybrid policy is configured
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
                cfg=self.cfg,
            )
            self.logic_shape = (
                (2, self.observation_space[-1]) if len(self.observation_space) == 1 else self.observation_space
            )
            self.register_buffer("logic_obs", torch.zeros((self.num_steps, self.num_envs) + self.logic_shape))
        else:
            from src.app.core.model_registry import build_model

            model_arch = self.resolve_model_name(default=default_arch)
            obs_dim = (
                self.observation_space[-1] if hasattr(self, "observation_space") and self.observation_space else None
            )
            self.model = build_model(
                model_arch,
                env=self.env,
                n_actions=self.n_actions,
                device=self.device,
                hidden_sizes=self.get_cfg("hidden_sizes", [64, 64]),
                obs_dim=obs_dim,
            )

        # Storage for rollouts
        self.register_buffer("obs", torch.zeros((self.num_steps, self.num_envs) + self.observation_space))
        self.register_buffer("actions", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("logprobs", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("rewards", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("terminations", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("truncations", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("values", torch.zeros((self.num_steps, self.num_envs)))

        init_obs = self.env.reset()
        if isinstance(init_obs, tuple):
            init_obs = init_obs[1] if len(init_obs) == 2 and isinstance(init_obs[1], torch.Tensor) else init_obs[0]
        self.next_obs = torch.as_tensor(init_obs, dtype=torch.float32)
        self.next_logic_obs = self._prepare_logic_obs(self.next_obs) if self.is_modular else None
        self.next_done = torch.zeros(self.num_envs)
        self.next_terminated = torch.zeros(self.num_envs)
        self.next_truncated = torch.zeros(self.num_envs)

        self.global_step_count = 0

    def _prepare_logic_obs(self, obs, logic_obs=None):
        if logic_obs is not None:
            return logic_obs.to(self.device)
        if obs.ndim == 2:
            return obs.unsqueeze(1).repeat(1, 2, 1).to(self.device)
        return obs.to(self.device)

    def forward(self, x, logic_x=None):
        if self.is_modular:
            return self.model(x, logic_x)
        return self.model(x)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        if self.is_modular:
            if logic_obs is None and obs.ndim == 2:
                logic_obs = obs.unsqueeze(1).repeat(1, 2, 1)
            return self.model.get_action_and_value(obs, logic_obs, action)
        return self.model.get_action_and_value(obs, action)

    def get_action_probs(self, obs, logic_obs=None):
        if self.is_modular:
            if logic_obs is None and obs.ndim == 2:
                logic_obs = obs.unsqueeze(1).repeat(1, 2, 1)
            probs, _ = self.model.actor(obs, logic_obs)
            return probs
        elif hasattr(self.model, "get_action_probs"):
            return self.model.get_action_probs(obs)
        elif hasattr(self.model, "actor") and hasattr(self.model.actor, "get_action_probs"):
            return self.model.actor.get_action_probs(obs)
        action, _, dist_entropy, _ = self.model.get_action_and_value(obs)
        return torch.softmax(action, dim=-1)

    def get_action(self, obs, logic_obs=None):
        probs = self.get_action_probs(obs, logic_obs)
        return torch.argmax(probs, dim=-1)

    def get_value(self, obs, logic_obs=None):
        if self.is_modular:
            if logic_obs is None and obs.ndim == 2:
                logic_obs = obs.unsqueeze(1).repeat(1, 2, 1)
            return self.model.get_value(obs, logic_obs)
        return self.model.get_value(obs)

    def on_train_epoch_start(self):
        cfg = self.cfg
        if cfg.paradigm == "offline_rl":
            return

        if self.global_step_count >= cfg.total_timesteps:
            self.trainer.should_stop = True
            return

        self.model.eval()
        for step in range(self.num_steps):
            self.global_step_count += self.num_envs
            self.obs[step] = self.next_obs
            if self.is_modular and self.next_logic_obs is not None:
                self.logic_obs[step] = self.next_logic_obs
            self.terminations[step] = self.next_terminated
            self.truncations[step] = self.next_truncated

            with torch.no_grad():
                if self.is_modular:
                    res = self.model.get_action_and_value(
                        self.next_obs.to(self.device),
                        self.next_logic_obs.to(self.device) if self.next_logic_obs is not None else None,
                    )
                else:
                    res = self.model.get_action_and_value(self.next_obs.to(self.device))

                action = getattr(res, "action", res[0])
                logprob = getattr(res, "logprob", res[1])
                value = getattr(res, "value", res[3])
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            step_res = self.env.step(action.cpu().numpy())
            if len(step_res) == 5:
                next_obs, reward, terminated, truncated, infos = step_res
            else:
                next_obs, reward, terminated = step_res[:3]
                truncated = [False] * len(reward)

            next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
            truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

            # Save dataset transitions if configured
            if hasattr(self, "dataset_writer") and self.dataset_writer is not None:
                self.dataset_writer.write(
                    obs=self.next_obs.cpu().numpy(),
                    action=action.cpu().numpy(),
                    reward=reward.cpu().numpy(),
                    done=(terminated | truncated).cpu().numpy(),
                    next_obs=next_obs.cpu().numpy(),
                    logic_obs=self.next_logic_obs.cpu().numpy()
                    if (self.is_modular and self.next_logic_obs is not None)
                    else None,
                    next_logic_obs=self._prepare_logic_obs(next_obs).cpu().numpy() if self.is_modular else None,
                )

            self.rewards[step] = reward.view(-1)
            self.next_obs = next_obs
            if self.is_modular:
                self.next_logic_obs = self._prepare_logic_obs(next_obs)
            self.next_terminated = terminated
            self.next_truncated = truncated
            self.next_done = terminated | truncated

        # Bootstrap value if not done
        with torch.no_grad():
            if self.is_modular:
                next_value = self.model.get_value(
                    self.next_obs.to(self.device),
                    self.next_logic_obs.to(self.device) if self.next_logic_obs is not None else None,
                ).reshape(1, -1)
            else:
                next_value = self.model.get_value(self.next_obs.to(self.device)).reshape(1, -1)

            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - ((self.terminations[t + 1] + self.truncations[t + 1]) > 0).float()
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # Flatten the batch
        self.b_obs = self.obs.reshape((-1,) + self.observation_space)
        if self.is_modular:
            self.b_logic_obs = self.logic_obs.reshape((-1,) + self.logic_shape)
        self.b_logprobs = self.logprobs.reshape(-1)
        self.b_actions = self.actions.reshape(-1)
        self.b_advantages = advantages.reshape(-1)
        self.b_returns = returns.reshape(-1)
        self.b_values = self.values.reshape(-1)

        self.model.train()

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        if cfg.paradigm == "offline_rl":
            return

        b_inds = np.arange(self.b_obs.shape[0])
        np.random.shuffle(b_inds)

        for start in range(0, self.b_obs.shape[0], self.batch_size):
            end = start + self.batch_size
            mb_inds = b_inds[start:end]

            mb_obs = self.b_obs[mb_inds].to(self.device)
            mb_actions = self.b_actions[mb_inds].to(self.device)

            if self.is_modular:
                mb_logic_obs = self.b_logic_obs[mb_inds].to(self.device)
                res = self.model.get_action_and_value(mb_obs, mb_logic_obs, mb_actions.long())
            else:
                res = self.model.get_action_and_value(mb_obs, mb_actions.long())

            _, newlogprob, entropy, newvalue = res[:4]
            blend_entropy = getattr(res, "blend_entropy", None)

            logratio = newlogprob - self.b_logprobs[mb_inds].to(self.device)
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()

            mb_advantages = self.b_advantages[mb_inds].to(self.device)
            if self.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if self.clip_vloss:
                v_loss_unclipped = (newvalue - self.b_returns[mb_inds].to(self.device)) ** 2
                v_clipped = self.b_values[mb_inds].to(self.device) + torch.clamp(
                    newvalue - self.b_values[mb_inds].to(self.device),
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - self.b_returns[mb_inds].to(self.device)) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds].to(self.device)) ** 2).mean()

            entropy_loss = entropy.mean()
            blend_entropy_loss = blend_entropy.mean() if isinstance(blend_entropy, torch.Tensor) else 0.0

            loss = (
                pg_loss
                - self.ent_coef * entropy_loss
                - self.blend_ent_coef * blend_entropy_loss
                + v_loss * self.vf_coef
            )

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            opt.step()

        self.log("transitions", self.global_step_count)
        log_metrics = {
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/total_loss": loss.item(),
        }
        if self.is_modular and isinstance(blend_entropy, torch.Tensor):
            log_metrics["losses/blend_entropy"] = blend_entropy.mean().item()
        self.log_dict(log_metrics)

    def configure_optimizers(self):
        if self.is_modular:
            params = []
            if hasattr(self.model, "actor") and hasattr(self.model.actor, "policy_modules"):
                for m, m_type in zip(self.model.actor.policy_modules, self.model.module_types):
                    lr = self.lr if m_type == "neural" else self.logic_lr
                    params.append({"params": m.parameters(), "lr": lr})
            if (
                hasattr(self.model, "actor")
                and hasattr(self.model.actor, "blender")
                and self.model.actor.blender is not None
            ):
                params.append({"params": self.model.actor.blender.parameters(), "lr": self.blender_lr})
            if hasattr(self.model, "critic"):
                params.append({"params": self.model.critic.parameters(), "lr": self.lr})
            if not params:
                params = self.model.parameters()
            return optim.Adam(params, eps=1e-5)
        return optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)

    def on_fit_end(self):
        if hasattr(self, "dataset_writer") and self.dataset_writer is not None:
            self.dataset_writer.close()
        super().on_fit_end()
