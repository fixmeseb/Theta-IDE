import os
import pickle
import random

# from nudge.env import NudgeBaseEnv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from torch.distributions.categorical import Categorical

PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
for _p in [
    os.path.join(PROJECT_ROOT, "src", "usr", "models"),
    os.path.join(PROJECT_ROOT, "src", "usr", "environments"),
]:
    if _p not in sys.path and os.path.exists(_p):
        sys.path.insert(0, _p)

from nsfr.common import get_nsfr_model
from nsfr.utils.common import load_module
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic, NeuralPPO
from nudge.torch_utils import softor
from src.app.core.factories import get_blender, get_neural_agent
from src.app.core.model_registry import register_model
from src.app.core.types import ActionResult


class BlenderActor(nn.Module):
    """
    BlendeRL actor that combines heterogeneous policy modules.

    Args:
        env: environment
        policy_modules: list of policy modules (NSFR, CEW, Neural, etc.)
        module_types: list of types for each module
        blender: blending policy
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
    """

    def __init__(
        self,
        env,
        policy_modules,
        module_types,
        blender,
        actor_mode,
        blender_mode,
        blend_function,
        device=None,
        explain=False,
    ):
        """
        Initialize a BlendeRL agent.
        """
        super().__init__()
        self.env = env
        self.policy_modules = nn.ModuleList(policy_modules)
        self.module_types = module_types
        self.blender = blender
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.device = device
        self.explain = explain

        # Build mappings for logic-based blender if needed
        self.blender_id_to_pred_indices = self._build_blender_id_dict()

    def _build_blender_id_dict(self):
        """
        Initialize a dictionary that maps blender mode id to predicate indices.
        Returns:
            blender_id_to_pred_indices: dictionary that maps blender mode id to predicate indices
        """
        if self.blender_mode == "neural":
            return {}

        blender_mode_names = [f"agent_{i}" for i in range(len(self.policy_modules))]
        # Compatibility with legacy names if only 2 modules (neural, logic)
        if len(self.policy_modules) == 2:
            blender_mode_names = ["neural_agent", "logic_agent"]

        blender_id_to_pred_indices = {i: [] for i in range(len(blender_mode_names))}

        if hasattr(self.blender, "get_prednames"):
            for j, pred_name in enumerate(self.blender.get_prednames()):
                for i, mode_name in enumerate(blender_mode_names):
                    if mode_name in pred_name:
                        blender_id_to_pred_indices[i].append(j)
        return blender_id_to_pred_indices

    def _map_logic_output(self, q, module):
        action_names = self.env.get_action_meanings()

        if hasattr(module, "prednames"):
            mapped_q = torch.zeros(q.size(0), len(action_names), device=q.device)
            for idx, action_name in enumerate(action_names):
                if action_name in module.prednames:
                    pred_idx = module.prednames.index(action_name)
                    mapped_q[:, idx] = q[:, pred_idx]
        elif q.size(1) == len(action_names):
            mapped_q = q
        else:
            return q

        # Normalize into a valid action probability distribution:
        # If any action predicate is active, normalize proportionally across valid action candidates.
        # If no action predicate is active (sum == 0, meaning logic is silent/unmatched):
        # Default to safe action (e.g. "withhold" / "noop") rather than uniform 50/50 random.
        sum_q = mapped_q.sum(dim=-1, keepdim=True)
        active_mask = sum_q > 1e-6

        if "withhold" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("withhold")] = 1.0
        elif "noop" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("noop")] = 1.0
        else:
            default_probs = torch.full_like(mapped_q, 1.0 / len(action_names))

        normalized_q = torch.where(active_mask, mapped_q / torch.clamp(sum_q, min=1e-6), default_probs)
        return normalized_q

    def get_explanation(self, neural_state, logic_state, action):
        """
        Get the explanation of the blending weights.
        """
        # TODO: Update for heterogeneous modules
        return None, None, self.to_blender_policy_distribution(neural_state, logic_state)[0].detach().cpu().numpy()

    def compute_action_probs_hybrid(self, neural_state, logic_state):
        """
        Compute action probabilities by blending all modules.
        """
        batch_size = neural_state.size(0)
        module_probs = []

        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                probs = module.get_action_probs(neural_state)
            elif m_type == "logic":
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
            else:
                # Custom / plugin submodule (e.g. CEW or other continuous/relational modules)
                sub_inp = (
                    neural_state
                    if (
                        neural_state.ndim == 2
                        and hasattr(module, "n_inputs")
                        and neural_state.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else neural_state)
                )
                if hasattr(module, "get_action_probs"):
                    probs = self._map_logic_output(module.get_action_probs(sub_inp), module)
                else:
                    probs = module(sub_inp)
            module_probs.append(probs)

        # weights size: B * N_modules
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        self.w_policy = weights[0]

        action_probs = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_probs in enumerate(module_probs):
            action_probs += weights[:, i].unsqueeze(1) * m_probs.to(neural_state.device)

        return action_probs, weights

    def compute_action_probs_logic(self, logic_state):
        """
        Compute action probabilities using only non-neural modules.
        """
        # Determine expected neural input size from blender if possible
        n_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
            n_in = self.blender.network[0].in_features
        elif hasattr(self.blender, "fc"):  # logic blender
            n_in = 1  # dummy

        dummy_neural = torch.zeros(logic_state.size(0), n_in).to(logic_state.device)
        weights = self.to_blender_policy_distribution(dummy_neural, logic_state)

        # Zero out neural modules and re-normalize
        for i, m_type in enumerate(self.module_types):
            if m_type == "neural":
                weights[:, i] = 0.0

        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]

        action_probs = torch.zeros(logic_state.size(0), self.env.n_actions, device=logic_state.device)
        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                continue
            elif m_type == "logic":
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
                action_probs += weights[:, i].unsqueeze(1) * probs
            else:
                sub_inp = (
                    dummy_neural
                    if (
                        dummy_neural.ndim == 2
                        and hasattr(module, "n_inputs")
                        and dummy_neural.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else dummy_neural)
                )
                if hasattr(module, "get_action_probs"):
                    probs = self._map_logic_output(module.get_action_probs(sub_inp), module)
                else:
                    probs = module(sub_inp)
                action_probs += weights[:, i].unsqueeze(1) * probs

        return action_probs, weights

    def compute_action_probs_neural(self, neural_state):
        """
        Compute action probabilities using only neural modules.
        """
        # Determine expected logic input size from blender
        l_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
            l_in = self.blender.network[0].in_features

        dummy_logic = torch.zeros(neural_state.size(0), l_in).to(neural_state.device)
        weights = self.to_blender_policy_distribution(neural_state, dummy_logic)

        # Zero out logic modules and re-normalize
        for i, m_type in enumerate(self.module_types):
            if m_type != "neural":
                weights[:, i] = 0.0

        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]

        action_probs = torch.zeros(neural_state.size(0), self.env.n_actions, device=neural_state.device)
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                probs = module.get_action_probs(neural_state)
                action_probs += weights[:, i].unsqueeze(1) * probs

        return action_probs, weights

    def to_blender_policy_distribution(self, neural_state, logic_state):
        """
        Merge policies using the blender function.
        """
        if self.blender_mode == "logic":
            policy_probs = self.blender(logic_state)
            batch_size = policy_probs.size(0)
            mode_probs = []
            n_modes = len(self.policy_modules)
            for i in range(n_modes):
                indices = torch.tensor(self.blender_id_to_pred_indices.get(i, []), device=policy_probs.device)
                if len(indices) == 0:
                    mode_probs.append(torch.zeros(batch_size, 1, device=policy_probs.device))
                    continue
                indices = indices.expand(batch_size, -1)
                gathered = torch.gather(policy_probs, 1, indices)
                merged = softor(gathered, dim=1)
                mode_probs.append(merged)

            probs = torch.stack(mode_probs, dim=1).squeeze(-1)
            logits = torch.logit(probs, eps=0.01)
        else:
            # Neural blender
            if len(neural_state.shape) == 2:  # vector
                logits = self.blender(logic_state)
            else:
                logits = self.blender(neural_state)

        if self.blend_function == "softmax":
            return torch.softmax(logits, dim=1)
        else:
            return F.gumbel_softmax(logits, dim=1, hard=True)

    def forward(self, neural_state, logic_state):
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state)

    def get_q_values(self, neural_state, logic_state=None):
        """Compute Q-values by blending Q-values from all modules."""
        if logic_state is None:
            if neural_state.ndim == 2:
                logic_state = neural_state.unsqueeze(1).repeat(1, 2, 1)
            else:
                logic_state = neural_state
        batch_size = neural_state.size(0)
        module_q_values = []

        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                if hasattr(module, "get_q_values"):
                    try:
                        q = module.get_q_values(neural_state, logic_obs=logic_state)
                    except TypeError:
                        q = module.get_q_values(neural_state)
                elif hasattr(module, "forward"):
                    try:
                        q = module(neural_state, logic_obs=logic_state)
                    except TypeError:
                        q = module(neural_state)  # Assuming forward returns Q-values for Q-networks
                else:
                    q = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
            elif m_type == "logic":
                # logic (NSFR / Neumann)
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(logic_state)
                else:
                    # Logic modules usually return probs, treat as Q-values [0, 1]
                    q = self._map_logic_output(module.get_action_probs(logic_state), module)
            else:
                # Custom / plugin submodule (e.g. CEW or other continuous/relational modules)
                sub_inp = (
                    neural_state
                    if (
                        neural_state.ndim == 2
                        and hasattr(module, "n_inputs")
                        and neural_state.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else neural_state)
                )
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(sub_inp)
                else:
                    q = module(sub_inp)
            module_q_values.append(q)

        weights = self.to_blender_policy_distribution(neural_state, logic_state)

        q_values = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_q in enumerate(module_q_values):
            q_values = q_values + weights[:, i].unsqueeze(1) * m_q.to(neural_state.device)

        return q_values


@register_model("blendrl", "blender")
class BlenderActorCritic(nn.Module):
    """
    BlendeRL actor-critic that supports heterogeneous policy modules.
    """

    def __init__(
        self,
        env,
        rules,
        actor_mode,
        blender_mode,
        blend_function,
        reasoner,
        device,
        architecture=None,
        rng=None,
        explain=False,
        modules=None,
        cfg=None,  # For accessing other agent hyperparams
    ):
        super().__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env
        self.cfg = cfg
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.reasoner = reasoner
        self.architecture = architecture
        self.explain = explain

        hidden_sizes = self.get_cfg("hidden_sizes", [64, 64])
        if hidden_sizes is not None:
            hidden_sizes = list(hidden_sizes)

        self.policy_modules = nn.ModuleList()
        self.module_types = []

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        neural_in_features = obs.shape[-1]

        def _get_val(obj, key, fallback=None):
            if isinstance(obj, dict):
                return obj.get(key, fallback)
            return getattr(obj, key, fallback) if hasattr(obj, key) else fallback

        # 1. Parse modules from argument, modules list, or hierarchical neural/symbolic blocks
        modules_list = modules if modules is not None else self.get_cfg("modules", None)
        symbolic_cfg = self.get_cfg("symbolic", None) or self.get_cfg("symbolic_actor", None)
        neural_cfg = self.get_cfg("neural", None) or self.get_cfg("neural_actor", None)
        blender_cfg = self.get_cfg("blender", None) or self.get_cfg("blender_actor", None)

        if blender_cfg is not None:
            self.blender_mode = _get_val(blender_cfg, "mode", self.blender_mode)
            self.blend_function = _get_val(blender_cfg, "blend_function", self.blend_function)
            blender_reasoner = _get_val(blender_cfg, "reasoner", self.reasoner or "nsfr")
        else:
            blender_reasoner = self.reasoner or "nsfr"

        if modules_list is None and (symbolic_cfg is not None or neural_cfg is not None):
            modules_list = []
            if symbolic_cfg is not None:
                if isinstance(symbolic_cfg, str):
                    s_dict = {"type": self.reasoner or "nsfr", "rules": symbolic_cfg}
                else:
                    s_dict = dict(symbolic_cfg) if hasattr(symbolic_cfg, "items") else {}
                    s_type = _get_val(
                        s_dict, "type", _get_val(s_dict, "name", _get_val(s_dict, "reasoner", self.reasoner or "nsfr"))
                    )
                    s_dict["type"] = s_type
                    if "rules" not in s_dict and s_type in ("nsfr", "neumann"):
                        s_dict["rules"] = self.get_cfg("rules", "default")
                modules_list.append(s_dict)

            if neural_cfg is not None:
                if isinstance(neural_cfg, str):
                    n_dict = {"module_type": "neural", "architecture": neural_cfg}
                else:
                    n_dict = dict(neural_cfg) if hasattr(neural_cfg, "items") else {}
                    arch = _get_val(
                        n_dict,
                        "architecture",
                        _get_val(n_dict, "type", _get_val(n_dict, "name", self.architecture or "dnn")),
                    )
                    n_dict["architecture"] = arch
                    n_dict["module_type"] = "neural"
                modules_list.append(n_dict)

        self.module_cfgs = list(modules_list) if modules_list else []

        blender_rules = None
        if modules_list:
            for m_cfg in modules_list:
                m_type = _get_val(m_cfg, "module_type", _get_val(m_cfg, "type", "neural"))
                if m_type == "nsfr" or m_type == "neumann":
                    m_rules = _get_val(m_cfg, "rules", self.get_cfg("rules", "default"))
                    if self.reasoner == "neumann" or m_type == "neumann":
                        from neumann.common import get_neumann_model

                        m = get_neumann_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    else:
                        m = get_nsfr_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    self.policy_modules.append(m)
                    self.module_types.append("logic")
                    if blender_rules is None:
                        blender_rules = m_rules
                elif m_type not in ("nsfr", "neumann", "neural"):
                    # Dynamically instantiate from MODEL_REGISTRY without hardcoded classes
                    from src.app.core.model_registry import build_model

                    m = build_model(
                        m_cfg,
                        env=env,
                        device=device,
                        obs_dim=int(np.prod(obs.shape[1:])),
                        n_actions=env.n_actions,
                    ).to(device)
                    self.policy_modules.append(m)
                    self.module_types.append(m_type)
                elif m_type == "neural":
                    arch = _get_val(m_cfg, "architecture", _get_val(m_cfg, "type", self.architecture))
                    if arch == "neural" or not arch:
                        arch = self.architecture or "dnn"
                    m_hidden = _get_val(m_cfg, "hidden_sizes", hidden_sizes)
                    m = get_neural_agent(
                        env.name,
                        env.n_actions,
                        device,
                        arch_name=arch,
                        hidden_sizes=m_hidden,
                        num_in_features=neural_in_features,
                    )
                    self.policy_modules.append(m)
                    self.module_types.append("neural")
                    self.architecture = arch
        else:
            # Backward compatibility with 'rules' string
            if isinstance(rules, str) and "," in rules:
                rulesets = [r.strip() for r in rules.split(",")]
            elif isinstance(rules, list):
                rulesets = rules
            else:
                rulesets = [rules]

            # Add Neural module first
            self.policy_modules.append(
                get_neural_agent(
                    env.name,
                    env.n_actions,
                    device,
                    arch_name=self.architecture,
                    hidden_sizes=hidden_sizes,
                    num_in_features=neural_in_features,
                )
            )
            self.module_types.append("neural")

            # Add Logic modules
            for r in rulesets:
                if self.reasoner == "neumann":
                    from neumann.common import get_neumann_model

                    la = get_neumann_model(env.name, r, device=device, train=True, explain=self.explain)
                else:
                    la = get_nsfr_model(env.name, r, device=device, train=True, explain=self.explain)
                self.policy_modules.append(la)
                self.module_types.append("logic")

        out_size = len(self.policy_modules)

        # Use first logic module's rules for blender if logic-based
        if blender_rules is None:
            blender_rules = rulesets[0] if "rulesets" in locals() else (rules if isinstance(rules, str) else rules[0])

        self.blender = get_blender(
            env,
            blender_rules,
            device,
            blender_mode=self.blender_mode,
            reasoner=blender_reasoner,
            train=True,
            explain=self.explain,
            out_size=out_size,
            architecture=self.architecture if self.architecture else "cnn",
        )

        # Load logic critic (MLP)
        mlp_module_path = f"in/envs/{env.name}/mlp.py"
        if os.path.exists(mlp_module_path):
            module = load_module(mlp_module_path)
            mlp_cls = getattr(module, "StandardMLP", getattr(module, "MLP", None))
            if mlp_cls:
                from src.app.core.factories import _safe_instantiate

                self.logic_critic = _safe_instantiate(
                    mlp_cls,
                    device=device,
                    out_size=1,
                    logic=True,
                    hidden_sizes=hidden_sizes,
                    num_in_features=neural_in_features,
                )
            else:
                self.logic_critic = None
        else:
            self.logic_critic = None

        self.actor = BlenderActor(
            env,
            self.policy_modules,
            self.module_types,
            self.blender,
            self.actor_mode,
            self.blender_mode,
            self.blend_function,
            device=device,
        )

    def get_cfg(self, key, default=None):
        """Helper to get a config value from cfg.model, cfg.agent, or cfg (supports dot-paths)."""
        if self.cfg is None:
            return default

        def _get_nested(root, k):
            if root is None:
                return None, False
            if hasattr(root, k):
                val = getattr(root, k)
                if val is not None:
                    return val, True
            if isinstance(root, dict) and k in root:
                return root[k], True
            if "." in k:
                parts = k.split(".")
                curr = root
                for p in parts:
                    if curr is None:
                        return None, False
                    if hasattr(curr, p):
                        curr = getattr(curr, p)
                    elif isinstance(curr, dict) and p in curr:
                        curr = curr[p]
                    else:
                        return None, False
                return curr, True
            return None, False

        for ns in ["model", "agent"]:
            if hasattr(self.cfg, ns):
                val, found = _get_nested(getattr(self.cfg, ns), key)
                if found:
                    return val
            elif isinstance(self.cfg, dict) and ns in self.cfg:
                val, found = _get_nested(self.cfg[ns], key)
                if found:
                    return val

        val, found = _get_nested(self.cfg, key)
        if found:
            return val
        return default

    def forward(self, neural_state, logic_state=None, action=None):
        return self.get_action_and_value(neural_state, logic_state, action=action)

    def get_action_and_value(self, neural_state, logic_state=None, action=None):
        if logic_state is None and neural_state.ndim == 2:
            logic_state = neural_state.unsqueeze(1).repeat(1, 2, 1)
        action_probs, blending_weights = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        blend_dist = Categorical(blending_weights)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        blended_value = self.get_value(neural_state, logic_state, blending_weights=blending_weights)

        return ActionResult(
            action=action,
            logprob=logprob,
            entropy=dist.entropy(),
            value=blended_value,
            aux={
                "blend_entropy": blend_dist.entropy(),
                "blending_weights": blending_weights,
            },
        )

    def get_q_values(self, neural_state, logic_state=None):
        if self.get_cfg("blend_q_values", True):
            return self.actor.get_q_values(neural_state, logic_state)
        # Legacy fallback if blend_q_values is explicitly False
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                if hasattr(module, "get_q_values"):
                    return module.get_q_values(neural_state)
                elif hasattr(module, "actor"):
                    x = neural_state.float().reshape(neural_state.shape[0], -1)
                    hidden = module.network(x)
                    return module.actor(hidden)
                elif hasattr(module, "forward"):
                    return module(neural_state)
        return self.actor.get_q_values(neural_state, logic_state)

    def get_value(self, neural_state, logic_state, blending_weights=None):
        if blending_weights is None:
            _, blending_weights = self.actor(neural_state, logic_state)

        neural_value = self.get_neural_value(neural_state).squeeze(1)
        logic_value = self.get_logic_value(logic_state).squeeze(1)

        # Weighted value blending
        # w_neural_sum * V_neural + w_logic_sum * V_logic
        neural_weight_sum = 0
        logic_weight_sum = 0
        for i, m_type in enumerate(self.module_types):
            if m_type == "neural":
                neural_weight_sum += blending_weights[:, i]
            else:
                logic_weight_sum += blending_weights[:, i]

        blended_value = (neural_weight_sum * neural_value + logic_weight_sum * logic_value).unsqueeze(1)
        return blended_value

    def get_neural_value(self, neural_state):
        # Find the first neural module to use its critic
        for i, m in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                return m.get_value(neural_state)
        return torch.zeros(neural_state.size(0), 1, device=neural_state.device)

    def get_logic_value(self, logic_state):
        if self.logic_critic:
            if hasattr(self.logic_critic, "num_in_features"):
                flat_size = np.prod(logic_state.shape[1:])
                if flat_size != self.logic_critic.num_in_features and logic_state.ndim > 2:
                    if logic_state.shape[-1] == self.logic_critic.num_in_features:
                        logic_state = logic_state[:, 0, :]
            return self.logic_critic(logic_state)
        return torch.zeros(logic_state.size(0), 1, device=logic_state.device)

    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)
