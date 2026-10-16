"""Tests for plugin discovery, entry point registration, and component protocols."""
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.app.core.model_registry import MODEL_REGISTRY
from src.app.core.paradigm_loader import _COMPONENT_REGISTRY
from src.app.core.plugin_discovery import discover_entry_points, load_all_plugins
from src.app.core.protocols import (
    AgentPluginProtocol,
    DataModulePluginProtocol,
    EnvironmentPluginProtocol,
    EvaluationHookProtocol,
    TrajectorySamplerProtocol,
)


class DummyDataModule:
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return []

    def val_dataloader(self):
        return []


class DummyTrajectorySampler:
    def sample_trajectory_window(self, batch_size: int, context_len: int):
        return {"obs": torch.zeros((batch_size, context_len, 4))}


class DummyEnvPlugin:
    @classmethod
    def make_env(cls, env_id: str, **kwargs):
        return None

    @classmethod
    def get_default_wrappers(cls):
        return []

    @classmethod
    def get_reward_shaping_hook(cls):
        return None

    @classmethod
    def get_logic_valuation_hook(cls):
        return None


class DummyAgentPlugin:
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def configure_optimizers(self):
        return []


class DummyEvaluationHook:
    def on_eval_start(self, agent, env, stage="val"):
        pass

    def evaluate_step(self, agent, batch_or_transition):
        return {"eval/metric": 1.0}

    def on_eval_end(self):
        return {"eval/mean_metric": 1.0}


def test_runtime_protocols():
    assert isinstance(DummyDataModule(), DataModulePluginProtocol)
    assert isinstance(DummyTrajectorySampler(), TrajectorySamplerProtocol)
    assert isinstance(DummyEnvPlugin(), EnvironmentPluginProtocol)
    assert isinstance(DummyAgentPlugin(), AgentPluginProtocol)
    assert isinstance(DummyEvaluationHook(), EvaluationHookProtocol)


def test_discover_entry_points():
    mock_ep = MagicMock()
    mock_ep.name = "mock_plugin"
    mock_ep.load.return_value = DummyDataModule

    with patch("src.app.core.plugin_discovery.entry_points", return_value=[mock_ep]):
        results = discover_entry_points("theta.datamodules", force_reload=True)
        assert "mock_plugin" in results
        assert results["mock_plugin"] is DummyDataModule


def test_load_all_plugins_populates_registries():
    mock_model_ep = MagicMock()
    mock_model_ep.name = "plugin_test_model"
    mock_model_ep.load.return_value = type("PluginTestModel", (), {})

    mock_comp_ep = MagicMock()
    mock_comp_ep.name = "PluginTestDataModule"
    mock_comp_ep.load.return_value = DummyDataModule

    def mock_eps_fn(group=None):
        if group == "theta.models":
            return [mock_model_ep]
        elif group == "theta.datamodules":
            return [mock_comp_ep]
        return []

    with patch("src.app.core.plugin_discovery.entry_points", side_effect=mock_eps_fn):
        discovered = load_all_plugins(force_reload=True)
        assert "plugin_test_model" in discovered["models"]
        assert "PluginTestDataModule" in discovered["datamodules"]
        assert "plugin_test_model" in MODEL_REGISTRY
        assert "PluginTestDataModule" in _COMPONENT_REGISTRY
