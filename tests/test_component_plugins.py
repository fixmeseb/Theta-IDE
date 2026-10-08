"""Unit tests for the Component Plugin system, protocols, and model registry."""

from __future__ import annotations

import unittest.mock as mock

import pytest
import torch
import torch.nn as nn

from src.app.core.model_registry import (
    MODEL_REGISTRY,
    build_model,
    get_model_class,
    register_model,
)
from src.app.core.protocols import (
    DynamicTopologyProtocol,
    ExtraStateProtocol,
    HasModelCallbacks,
    walk_model_modules,
)
from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic
from src.usr.models.cew.cew_callback import CEWSelfOrganizationCallback
from src.usr.models.cew.cew_model import CEWModel


class DummyDynamicModel(nn.Module, DynamicTopologyProtocol, ExtraStateProtocol, HasModelCallbacks):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self._changed = False

    def forward(self, x):
        return self.linear(x)

    def has_topology_changed(self) -> bool:
        return self._changed

    def reset_topology_changed(self) -> None:
        self._changed = False

    def clone_topology_to(self, target: nn.Module) -> None:
        target.load_state_dict(self.state_dict())

    def get_callbacks(self) -> list:
        return ["dummy_callback"]

    def extra_state(self) -> dict:
        return {"dummy_key": 42}

    def load_extra_state(self, state: dict) -> None:
        self.dummy_val = state.get("dummy_key")


def test_register_and_build_custom_model():
    @register_model("test_dummy_plugin")
    def _create_dummy(obs_dim=4, n_actions=2, **kwargs):
        return nn.Linear(obs_dim, n_actions)

    assert "test_dummy_plugin" in MODEL_REGISTRY
    m = build_model("test_dummy_plugin", obs_dim=8, n_actions=3)
    assert isinstance(m, nn.Linear)
    assert m.in_features == 8
    assert m.out_features == 3


def test_registered_standard_neural_models():
    from src.usr.models.neural.architectures import CNNActor, MLPQNetwork, NeuralBlenderMLP

    mlp = build_model("mlp", obs_dim=6, n_actions=3)
    assert isinstance(mlp, MLPQNetwork)
    assert mlp.n_actions == 3
    assert mlp.num_in_features == 6

    cnn = build_model("cnn", n_actions=4)
    assert isinstance(cnn, CNNActor)

    blender_mlp = build_model("neural_blender_mlp", obs_dim=8, n_actions=2)
    assert isinstance(blender_mlp, NeuralBlenderMLP)

    from src.usr.models.neural.resnet import DuelingResNetMLP
    resnet = build_model("dueling_resnet", obs_dim=46, n_actions=2)
    assert isinstance(resnet, DuelingResNetMLP)

    from src.usr.models.neural.transformer import CrossAttentionSepsisPolicy, SepsisTransformerPolicy
    transformer = build_model("transformer", obs_dim=46, n_actions=2)
    assert isinstance(transformer, SepsisTransformerPolicy)

    cross_attn = build_model("cross_attention", obs_dim=46, n_actions=2)
    assert isinstance(cross_attn, CrossAttentionSepsisPolicy)


def test_factories_get_neural_agent_delegates_to_build_model():
    from src.app.core.factories import get_neural_agent
    from src.usr.models.neural.architectures import MLPQNetwork

    agent_model = get_neural_agent("cartpole", n_actions=2, device="cpu", arch_name="mlp", num_in_features=4)
    assert isinstance(agent_model, MLPQNetwork)
    assert agent_model.n_actions == 2



def test_cew_implements_protocols():
    cew = CEWModel(n_inputs=4, n_actions=2)

    assert isinstance(cew, DynamicTopologyProtocol)
    assert isinstance(cew, ExtraStateProtocol)
    assert isinstance(cew, HasModelCallbacks)

    # Initial state
    assert not cew.has_topology_changed()
    cew._request_optimizer_rebind = True
    assert cew.has_topology_changed()
    cew.reset_topology_changed()
    assert not cew.has_topology_changed()

    # Callbacks
    cbs = cew.get_callbacks()
    assert len(cbs) == 1
    assert isinstance(cbs[0], CEWSelfOrganizationCallback)

    # Extra state
    state = cew.extra_state()
    assert "rules" in state
    assert "antecedents" in state


def test_walk_model_modules_standalone():
    dummy = DummyDynamicModel()
    modules = walk_model_modules(dummy)
    assert dummy in modules


class DummyComposite:
    def __init__(self, policy_modules):
        self.policy_modules = policy_modules


class DummyAgent:
    def __init__(self, model=None, q_model=None, target_q_model=None):
        self.model = model
        self.q_model = q_model
        self.target_q_model = target_q_model
        self.rebind_called = False

    def _rebind_optimizer(self):
        self.rebind_called = True


def test_walk_model_modules_composite():
    cew = CEWModel(n_inputs=4, n_actions=2)
    dummy_blender = DummyComposite(policy_modules=[cew])
    dummy_agent = DummyAgent(model=dummy_blender)

    discovered = walk_model_modules(dummy_agent)
    assert cew in discovered


def test_base_agent_callback_discovery_without_hardcoding():
    from src.usr.methods.base_agent import OfflineAgentBase

    cew = CEWModel(n_inputs=4, n_actions=2)
    dummy_agent = DummyAgent(q_model=cew)

    callbacks = OfflineAgentBase.configure_callbacks(dummy_agent)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], CEWSelfOrganizationCallback)


def test_base_agent_dynamic_rebind_detection():
    from src.usr.methods.base_agent import OfflineAgentBase

    dynamic_model = DummyDynamicModel()
    target_model = DummyDynamicModel()
    dummy_agent = DummyAgent(q_model=dynamic_model, target_q_model=target_model)

    # Simulate topology change
    dynamic_model._changed = True
    assert dynamic_model.has_topology_changed()

    OfflineAgentBase._handle_optimizer_rebind(dummy_agent)

    # Verified that flag was reset and _rebind_optimizer was called
    assert not dynamic_model.has_topology_changed()
    assert dummy_agent.rebind_called


def test_standard_gymnasium_vector_env_resolution():
    from src.app.core.env_vectorized import StandardGymVectorEnv, VectorizedBaseEnv

    # Resolves any standard Gymnasium environment without needing an in/envs/ folder
    env = VectorizedBaseEnv.from_name("CartPole-v1", n_envs=2)
    assert isinstance(env, StandardGymVectorEnv)
    assert env.n_actions() == 2

    obs = env.reset()
    assert obs.shape == (2, 4)

    next_obs, rewards, term, trunc, infos = env.step(torch.tensor([0, 1]))
    assert next_obs.shape == (2, 4)
    assert len(rewards) == 2
    env.close()


def test_cql_agent_instantiates_with_neural_models():
    from omegaconf import OmegaConf

    from src.usr.methods.cql_agent import CQLAgent
    from src.usr.models.neural.architectures import MLPQNetwork
    from src.usr.models.neural.resnet import DuelingResNetMLP

    # 1. Test CQL with standard MLP
    cfg_mlp = OmegaConf.create({
        "agent": {"name": "cql_mlp", "algorithm": "cql", "lr": 1e-3, "cql_alpha": 1.0},
        "model": "mlp",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    })
    agent_mlp = CQLAgent(cfg_mlp)
    assert isinstance(agent_mlp.q_network, MLPQNetwork)
    assert isinstance(agent_mlp.target_q_network, MLPQNetwork)

    # 2. Test CQL with DuelingResNetMLP
    cfg_resnet = OmegaConf.create({
        "agent": {"name": "cql_resnet", "algorithm": "cql", "lr": 1e-3, "cql_alpha": 1.0},
        "model": "dueling_resnet",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    })
    agent_resnet = CQLAgent(cfg_resnet)
    assert isinstance(agent_resnet.q_network, DuelingResNetMLP)
    assert isinstance(agent_resnet.target_q_network, DuelingResNetMLP)


def test_action_count_harmonization():
    from src.app.core.env_vectorized import ActionCount, VectorizedBaseEnv

    ac = ActionCount(4)
    assert isinstance(ac, int)
    assert ac == 4
    assert callable(ac)
    assert ac() == 4
    assert ac * 2 == 8

    # Tensor initialization with ActionCount dimension
    t = torch.zeros(2, ac)
    assert t.shape == (2, 4)

    # Verify StandardGymVectorEnv exposes n_actions as ActionCount
    env = VectorizedBaseEnv.from_name("CartPole-v1", n_envs=1)
    try:
        assert isinstance(env.n_actions, int)
        assert env.n_actions == 2
        assert env.n_actions() == 2
        assert callable(env.n_actions)
    finally:
        env.close()


def test_paradigm_definition_and_data_modules():
    from omegaconf import OmegaConf

    from src.app.core.paradigm_loader import load_paradigm_definition
    from src.app.pipeline.config import get_known_algorithms, get_known_models

    # Verify known models and algorithms helpers
    models = get_known_models()
    assert {"mlp", "dnn", "dueling_resnet", "transformer", "cross_attention", "cew"} <= models
    algos = get_known_algorithms()
    assert {"cql", "ppo", "iql"} <= algos

    # 1. Online RL paradigm
    online_def = load_paradigm_definition("online_rl")
    assert online_def.data_module_cls is not None
    cfg_online = OmegaConf.create({
        "paradigm": "online_rl",
        "agent": {"name": "ppo", "batch_size": 32},
        "env": {"name": "cartpole", "offline_only": False},
        "seed": 42,
    })
    dm_online = online_def.data_module_cls(cfg_online)
    assert dm_online.train_dataloader() is not None

    # 2. Offline RL paradigm
    offline_def = load_paradigm_definition("offline_rl")
    assert offline_def.data_module_cls is not None

    # 3. Supervised paradigm
    supervised_def = load_paradigm_definition("supervised")
    assert supervised_def.data_module_cls is not None
    cfg_supervised = OmegaConf.create({
        "paradigm": "supervised",
        "env": {"name": "test_env", "offline_only": True},
    })
    dm_supervised = supervised_def.data_module_cls(cfg_supervised)
    assert hasattr(dm_supervised, "setup")


def test_iql_agent_instantiates_with_neural_models():
    from omegaconf import OmegaConf

    from src.usr.methods.iql_agent import IQLAgent
    from src.usr.models.neural.architectures import CNNActor, MLPQNetwork, MLPValueNetwork
    from src.usr.models.neural.resnet import DuelingResNetMLP

    # 1. Test IQL with standard MLP
    cfg_mlp = OmegaConf.create({
        "agent": {"name": "iql_mlp", "algorithm": "iql", "lr": 1e-3, "tau": 0.7, "beta": 3.0},
        "model": "mlp",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    })
    agent_mlp = IQLAgent(cfg_mlp)
    assert isinstance(agent_mlp.q_network, MLPQNetwork)
    assert isinstance(agent_mlp.q_network2, MLPQNetwork)
    assert isinstance(agent_mlp.target_q_network, MLPQNetwork)
    assert isinstance(agent_mlp.target_q_network2, MLPQNetwork)
    assert isinstance(agent_mlp.value_network, MLPValueNetwork)
    assert isinstance(agent_mlp.actor, MLPQNetwork)

    # Test Q computation
    dummy_obs = torch.randn(2, 4)
    q_vals = agent_mlp.get_q_values(dummy_obs)
    assert q_vals.shape == (2, 2)

    # 2. Test IQL with DuelingResNetMLP
    cfg_resnet = OmegaConf.create({
        "agent": {"name": "iql_resnet", "algorithm": "iql", "lr": 1e-3, "tau": 0.7, "beta": 3.0},
        "model": "dueling_resnet",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    })
    agent_resnet = IQLAgent(cfg_resnet)
    assert isinstance(agent_resnet.q_network, DuelingResNetMLP)
    assert isinstance(agent_resnet.target_q_network, DuelingResNetMLP)
    q_vals_resnet = agent_resnet.get_q_values(dummy_obs)
    assert q_vals_resnet.shape == (2, 2)


def test_hybrid_detection_helpers():
    from omegaconf import OmegaConf

    from src.usr.methods.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        def training_step(self, batch, batch_idx):
            pass

        def configure_optimizers(self):
            pass

        def get_action_and_value(self, x, action=None):
            pass

        def get_value(self, x):
            pass

    # 1. Hybrid via algorithm name
    agent1 = DummyAgent(OmegaConf.create({"agent": {"algorithm": "blendrl_ppo"}}))
    assert agent1.is_hybrid_configured() is True
    assert agent1.resolve_model_name() == "mlp"

    # 2. Hybrid via model string
    agent2 = DummyAgent(OmegaConf.create({"agent": {"algorithm": "ppo"}, "model": "blendrl"}))
    assert agent2.is_hybrid_configured() is True
    assert agent2.resolve_model_name() == "mlp"

    # 3. Hybrid via composite model dict
    agent3 = DummyAgent(
        OmegaConf.create({
            "agent": {"algorithm": "cql"},
            "model": {
                "blendrl": {
                    "neural": "dueling_resnet",
                    "symbolic": "cew",
                }
            },
        })
    )
    assert agent3.is_hybrid_configured() is True
    assert agent3.resolve_model_name() == "dueling_resnet"

    # 4. Standard neural model
    agent4 = DummyAgent(OmegaConf.create({"agent": {"algorithm": "cql"}, "model": "dueling_resnet"}))
    assert agent4.is_hybrid_configured() is False
    assert agent4.resolve_model_name() == "dueling_resnet"


def test_agent_initialization_with_plain_dicts():
    from src.usr.methods.cql_agent import CQLAgent
    from src.usr.methods.iql_agent import IQLAgent
    from src.usr.methods.ppo_agent import PPOAgent

    plain_ppo_cfg = {
        "agent": {"algorithm": "ppo", "lr": 1e-4, "batch_size": 32},
        "model": "mlp",
        "env": {"name": "cartpole", "offline_only": False, "n_actions": 2, "obs_dim": 4},
    }
    ppo = PPOAgent(plain_ppo_cfg)
    assert ppo.lr == 1e-4
    assert ppo.batch_size == 32

    plain_cql_cfg = {
        "agent": {"algorithm": "cql", "lr": 2e-4, "gamma": 0.95},
        "model": "mlp",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    }
    cql = CQLAgent(plain_cql_cfg)
    assert cql.lr == 2e-4
    assert cql.gamma == 0.95

    plain_iql_cfg = {
        "agent": {"algorithm": "iql", "lr": 3e-4, "tau": 0.7, "beta": 3.0},
        "model": "mlp",
        "env": {"name": "cartpole", "offline_only": True, "n_actions": 2, "obs_dim": 4},
    }
    iql = IQLAgent(plain_iql_cfg)
    assert iql.lr == 3e-4
    assert iql.tau == 0.7
    assert iql.beta == 3.0


def test_dynamic_build_model_cew_and_blendrl():
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic
    from src.usr.models.cew.cew_model import CEWModel

    m_cew = build_model("cew", n_inputs=3, n_actions=4)
    assert isinstance(m_cew, CEWModel)

    mock_env = mock.MagicMock()
    mock_env.name = "cartpole"
    mock_env.n_actions = 2
    mock_env.reset.return_value = (torch.zeros(1, 4), {})

    m_blend = build_model(
        "blendrl",
        env=mock_env,
        actor_mode="neural",
        modules=[{"module_type": "neural", "architecture": "mlp"}],
    )
    assert isinstance(m_blend, BlenderActorCritic)


def test_load_cleanrl_agent_parameterized_actions():
    from src.app.core.factories import load_cleanrl_agent

    agent_default = load_cleanrl_agent()
    assert agent_default.n_actions == 18

    agent_custom = load_cleanrl_agent(n_actions=4)
    assert agent_custom.n_actions == 4





