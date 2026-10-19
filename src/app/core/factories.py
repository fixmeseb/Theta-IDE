import logging
import os
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch

from src.app.core.utils import load_module

logger = logging.getLogger(__name__)


def _neural_architecture(name):
    """A class from the Hub-installed `neural` model component (src/usr/models/neural/)."""
    try:
        from src.usr.models.neural import architectures
    except ImportError as e:
        raise ImportError(
            f"{name} comes from the 'neural' model component, which is not installed. Install it from the Component Hub."
        ) from e
    return getattr(architectures, name)


def get_neural_agent(env_name, n_actions, device, arch_name=None, hidden_sizes=None, num_in_features=None, **kwargs):
    from src.app.core.model_registry import build_model

    if hidden_sizes is None:
        hidden_sizes = [64, 64]

    return build_model(
        arch_name or "mlp",
        env=env_name,
        n_actions=n_actions,
        device=device,
        hidden_sizes=hidden_sizes,
        num_in_features=num_in_features,
        **kwargs,
    )


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
    """Deprecated legacy helper. Blenders are now managed directly by the BlendRL plugin."""
    try:
        from src.usr.models.blendrl.agents.blender_agent import build_blender_module

        return build_blender_module(
            env=env,
            blender_rules=blender_rules,
            device=device,
            train=train,
            blender_mode=blender_mode,
            reasoner=reasoner,
            explain=explain,
            out_size=out_size,
            architecture=architecture,
        )
    except ImportError:
        pass

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
            net = _neural_architecture("NatureCNNActor")(out_size=out_size)
        else:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            num_in_features = np.prod(obs.shape[1:])
            net = _neural_architecture("StandardMLPActor")(num_in_features=num_in_features, out_size=out_size)
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


def load_cleanrl_agent(
    pretrained: bool = False,
    device: str | torch.device = "cpu",
    model_path: str | None = None,
    n_actions: int = 18,
):
    """Load CleanRL CNNActor, optionally restoring weights from a checkpoint path."""
    agent = _neural_architecture("CNNActor")(n_actions=n_actions)
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
