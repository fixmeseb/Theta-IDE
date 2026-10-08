"""Tests for BasePlotter load_metrics and method filtering."""
from pathlib import Path
import pandas as pd
import pytest

from plot.base import BasePlotter


class DummyPlotter(BasePlotter):
    name = "dummy"
    default_cfg = {}

    def plot(self, exp_id, cli_overrides=None, exp_config_name=None):
        pass


def test_load_metrics_filters_by_structured_methods(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plotter = DummyPlotter("dummy")

    group = "test_grp"
    exp_id = "test_exp"

    # Create dummy log dirs for two methods: method_a and method_b
    log_dir_a = tmp_path / "results" / "logs" / group / exp_id / "method_a" / "version_0"
    log_dir_b = tmp_path / "results" / "logs" / group / exp_id / "method_b" / "version_0"
    log_dir_a.mkdir(parents=True)
    log_dir_b.mkdir(parents=True)

    df_a = pd.DataFrame({"step": [0, 1], "loss": [0.5, 0.4]})
    df_b = pd.DataFrame({"step": [0, 1], "loss": [0.8, 0.7]})
    df_a.to_csv(log_dir_a / "metrics.csv", index=False)
    df_b.to_csv(log_dir_b / "metrics.csv", index=False)

    # Mock get_experiment_config to return methods dict containing only method_a
    monkeypatch.setattr(
        plotter,
        "get_experiment_config",
        lambda exp_id, exp_config_name=None: {
            "methods": {
                "method_a": {"agent": "ppo"}
            }
        },
    )

    metrics = plotter.load_metrics(group, exp_id)
    assert "method_a" in metrics
    assert "method_b" not in metrics
    assert "version_0" in metrics["method_a"]


def test_get_active_aliases_and_is_method_active():
    # 1. Structured methods dict
    aliases, has_filter = BasePlotter.get_active_aliases({"methods": {"ppo": {}, "cql_tuned": {}}})
    assert has_filter is True
    assert "ppo" in aliases
    assert "cql_tuned" in aliases
    assert BasePlotter.is_method_active("ppo", aliases, has_filter)
    assert BasePlotter.is_method_active("ppo_0", aliases, has_filter)
    assert BasePlotter.is_method_active("ppo_12", aliases, has_filter)
    assert not BasePlotter.is_method_active("iql", aliases, has_filter)

    # 2. Legacy online_methods
    aliases_legacy, has_filter_legacy = BasePlotter.get_active_aliases({"online_methods": "ppo, blendrl"})
    assert has_filter_legacy is True
    assert "ppo" in aliases_legacy
    assert "blendrl" in aliases_legacy

    # 3. No filter
    aliases_none, has_filter_none = BasePlotter.get_active_aliases({})
    assert has_filter_none is False
    assert BasePlotter.is_method_active("any_method", aliases_none, has_filter_none)

