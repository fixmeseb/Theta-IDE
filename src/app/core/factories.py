import logging
import os
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch

from src.app.core.utils import load_module
from src.usr.models.neural.architectures import CNNActor, NeuralBlenderActor, NeuralBlenderMLP
from src.app.pipeline import runtime

logger = logging.getLogger(__name__)


def _safe_instantiate(module_class, **kwargs):
    import inspect

    sig = inspect.signature(module_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return module_class(**kwargs)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return module_class(**valid_kwargs)


def get_neural_agent(
    env_name, n_actions, device, arch_name=None, hidden_sizes=[64, 64], num_in_features=None, **kwargs
):
    if num_in_features is None:
        from src.app.core.env_vectorized import VectorizedBaseEnv

        try:
            temp_env = VectorizedBaseEnv.from_name(env_name, n_envs=1, mode="eval")
            obs = temp_env.reset()
            num_in_features = obs.shape[-1]
            temp_env.close()
        except Exception as e:
            logger.debug("Could not infer num_in_features by instantiating %s: %s", env_name, e)

    if arch_name in ["cross_attention", "cross_attention_transformer", "sepsis_cross_attention"]:
        transformer_module_path = f"{runtime.ENVS_DIR}/{env_name}/transformer.py"
        if not os.path.exists(transformer_module_path):
            raise FileNotFoundError(f"Requested transformer architecture but {transformer_module_path} does not exist.")
        module = load_module(transformer_module_path)
        cls = getattr(module, "CrossAttentionPolicy", None) or getattr(module, "CrossAttentionSepsisPolicy")
        return _safe_instantiate(cls, device=device, out_size=n_actions, num_in_features=num_in_features, **kwargs).to(
            device
        )

    if arch_name in ["transformer", "sepsis_transformer"]:
        transformer_module_path = f"{runtime.ENVS_DIR}/{env_name}/transformer.py"
        if not os.path.exists(transformer_module_path):
            raise FileNotFoundError(f"Requested transformer architecture but {transformer_module_path} does not exist.")
        module = load_module(transformer_module_path)
        cls = getattr(module, "TransformerPolicy", None) or getattr(module, "SepsisTransformerPolicy")
        return _safe_instantiate(cls, device=device, out_size=n_actions, num_in_features=num_in_features, **kwargs).to(
            device
        )

    if arch_name in ["dueling_resnet", "resnet"]:
        mlp_module_path = f"{runtime.ENVS_DIR}/{env_name}/mlp.py"
        if not os.path.exists(mlp_module_path):
            raise FileNotFoundError(f"Requested resnet architecture but {mlp_module_path} does not exist.")
        module = load_module(mlp_module_path)
        cls = getattr(module, "DuelingResNetMLP", None) or getattr(module, "MLP")
        return _safe_instantiate(
            cls, device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features, **kwargs
        ).to(device)

    if arch_name in ["mlp", "dnn", "standard_mlp"]:
        mlp_module_path = f"{runtime.ENVS_DIR}/{env_name}/mlp.py"
        if not os.path.exists(mlp_module_path):
            raise FileNotFoundError(f"Requested MLP architecture but {mlp_module_path} does not exist.")
        module = load_module(mlp_module_path)
        cls = getattr(module, "StandardMLP", None) or getattr(module, "MLP")
        return _safe_instantiate(
            cls, device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features, **kwargs
        ).to(device)

    if arch_name == "cnn":
        return CNNActor(n_actions=n_actions).to(device)

    raise ValueError(f"Unknown architecture '{arch_name}' requested for environment '{env_name}'.")


def get_blender(
    env,
    blender_rules,
    device,
    train=True,
    blender_mode="logic",
    reasoner="nsfr",
    explain=False,
    out_size=2,
    architecture="cnn",
):
    assert blender_mode in ["logic", "neural"]
    if blender_mode == "logic":
        if reasoner == "nsfr":
            from nsfr.common import get_blender_nsfr_model

            return get_blender_nsfr_model(env.name, blender_rules, device, train=train, explain=explain)
        elif reasoner == "neumann":
            from neumann.common import get_blender_neumann_model, get_neumann_model

            return get_blender_neumann_model(env.name, blender_rules, device, train=train, explain=explain)
    if blender_mode == "neural":
        if architecture == "cnn":
            net = NeuralBlenderActor(out_size=out_size)
        else:
            obs = env.reset()
            num_in_features = np.prod(obs.shape[1:])
            net = NeuralBlenderMLP(num_in_features=num_in_features, out_size=out_size)
        net.to(device)
        return net


def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from src.app.core.atari_wrappers import make_atari_env as apply_wrappers

    def make_env(env_id, seed, capture_video, run_name):
        def thunk():
            env = gym.make(env_id)
            return apply_wrappers(env)

        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs


def load_cleanrl_agent(pretrained: bool = False, device: str | torch.device = "cpu", model_path: str | None = None):
    """Load CleanRL CNNActor, optionally restoring weights from a checkpoint path."""
    agent = CNNActor(n_actions=18)
    if pretrained:
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot load pretrained CleanRL weights: model path '{model_path}' does not exist."
            )
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Failed loading CleanRL weights from '{model_path}': {e}")
    agent.to(device)
    return agent


def load_logic_ppo(agent, path):
    new_actor_dic = OrderedDict()
    new_critic_dic = OrderedDict()
    dic = torch.load(path)
    for name, value in dic.items():
        if "actor." in name:
            new_name = name.replace("actor.", "")
            new_actor_dic[new_name] = value
        if "critic." in name:
            new_name = name.replace("critic.", "")
            new_critic_dic[new_name] = value
    agent.logic_actor.load_state_dict(new_actor_dic)
    agent.logic_critic.load_state_dict(new_critic_dic)
    return agent
