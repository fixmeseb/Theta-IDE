from unittest.mock import MagicMock, patch

import pytest

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.validation import list_paradigms, validate_experiment_config


class MockCfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.env = type("Env", (), {})()
        
    def get(self, k, default=None):
        return getattr(self, k, default)

@pytest.fixture
def mock_cfg_online():
    cfg = MockCfg(
        paradigm="online_rl",
        methods={"ppo_cp_tuned": {"agent": "ppo", "model": "dnn"}},
        intervals_count=1,
        eval_episodes=100,
        group="",
        experiment_id=""
    )
    cfg.env.name = "cartpole"
    cfg.env.offline_only = False
    return cfg

@pytest.fixture
def mock_cfg_offline():
    cfg = MockCfg(
        paradigm="offline_rl",
        methods={"iql_mimic": {"agent": "iql", "model": "dnn"}},
        intervals_count=1,
        eval_episodes=0,
        group="",
        experiment_id=""
    )
    cfg.env.name = "mimic"
    cfg.env.offline_only = True
    return cfg

@pytest.fixture
def mock_cfg_supervised():
    cfg = MockCfg(
        paradigm="supervised",
        methods={"ep_lstm": {"model": "lstm"}},
        intervals_count=1,
        eval_episodes=0,
        group="early_prediction",
        experiment_id="quick_test"
    )
    cfg.env.name = "mimic"
    cfg.env.offline_only = True
    return cfg







@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_missing_paradigm(mock_yaml, mock_cfg_online):
    mock_cfg_online.__dict__.update({"paradigm": None})
    with pytest.raises(ConfigurationError, match="has no 'paradigm' declared"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_unknown_paradigm(mock_yaml, mock_cfg_online):
    mock_cfg_online.__dict__.update({"paradigm": "unknown_xyz"})
    with pytest.raises(ConfigurationError, match="Unknown paradigm 'unknown_xyz'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_offline_rl_requires_offline_env(mock_yaml, mock_cfg_offline):
    """offline_rl paradigm should fail when env.offline_only is False."""
    mock_cfg_offline.env.offline_only = False
    with pytest.raises(ConfigurationError, match="requires 'env.offline_only'"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_online_rl_requires_live_env(mock_yaml, mock_cfg_online):
    """online_rl paradigm should fail when env.offline_only is True."""
    mock_cfg_online.env.offline_only = True
    with pytest.raises(ConfigurationError, match="requires 'env.offline_only'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_intervals_count_gt1_offline_rl(mock_yaml, mock_cfg_offline):
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
        "intervals_count": 2,
        "eval_episodes": 0,
        "online_methods": "",
        "offline_methods": "iql/mimic",
        "group": "",
        "experiment_id": "",
    })
    with pytest.raises(ConfigurationError, match="does not support intervals_count > 1"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_eval_episodes_offline_rl(mock_yaml, mock_cfg_offline):
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
        "intervals_count": 1,
        "eval_episodes": 100,
        "online_methods": "",
        "offline_methods": "iql/mimic",
        "group": "",
        "experiment_id": "",
    })
    with pytest.raises(ConfigurationError, match="disables simulated gym rollouts"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_missing_methods_online_rl(mock_yaml, mock_cfg_online):
    mock_cfg_online.__dict__.update({
        "paradigm": "online_rl",
        "intervals_count": 1,
        "eval_episodes": 100,
        "methods": {},
        "group": "",
        "experiment_id": "",
    })
    with pytest.raises(ConfigurationError, match="requires 'methods'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_supervised_paradigm_passes_for_ep(mock_yaml, mock_cfg_supervised):
    """supervised paradigm should pass cleanly for EP task config."""
    notices = validate_experiment_config(mock_cfg_supervised, "test_exp")
    assert isinstance(notices, list)


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_successful_validation_online_rl(mock_yaml, mock_cfg_online):
    notices = validate_experiment_config(mock_cfg_online, "test_exp")
    assert isinstance(notices, list)


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={"intervals_count": 5})
def test_explicit_intervals_count_in_raw_yaml(mock_yaml, mock_cfg_offline):
    """intervals_count declared explicitly in raw YAML should be caught even if cfg returns 1."""
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
        "eval_episodes": 0,
        "online_methods": "",
        "offline_methods": "iql/mimic",
        "group": "",
        "experiment_id": "",
    })
    with pytest.raises(ConfigurationError, match="does not support intervals_count > 1"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={"eval_episodes": 5})
def test_explicit_eval_episodes_in_raw_yaml(mock_yaml, mock_cfg_offline):
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
        "intervals_count": 1,
        "online_methods": "",
        "offline_methods": "iql/mimic",
        "group": "",
        "experiment_id": "",
    })
    with pytest.raises(ConfigurationError, match="disables simulated gym rollouts"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


def test_list_paradigms_includes_base_paradigms():
    """list_paradigms() should include all base paradigms from YAML files."""
    paradigms = list_paradigms()
    assert "supervised" in paradigms
    assert "offline_rl" in paradigms
    assert "online_rl" in paradigms


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_cql_forbidden_in_supervised(mock_yaml, mock_cfg_supervised):
    """Supervised paradigm must reject RL agents like cql."""
    mock_cfg_supervised.__dict__["methods"] = {
        "cql_dnn": {"agent": "cql", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="does not use RL agent harnesses"):
        validate_experiment_config(mock_cfg_supervised, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_ppo_forbidden_in_offline_rl(mock_yaml, mock_cfg_offline):
    """Offline RL paradigm must reject online agents like ppo."""
    mock_cfg_offline.__dict__["methods"] = {
        "ppo_dnn": {"agent": "ppo", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="forbidden in paradigm 'offline_rl'"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_disallowed_agent_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject agents not in allowed_agents."""
    mock_cfg_online.__dict__["methods"] = {
        "bad_method": {"agent": "unsupported_algo", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="not allowed in paradigm 'online_rl'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_cql_forbidden_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject offline agents like cql."""
    mock_cfg_online.__dict__["methods"] = {
        "cql_dnn": {"agent": "cql", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="forbidden in paradigm 'online_rl'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_iql_forbidden_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject offline agents like iql."""
    mock_cfg_online.__dict__["methods"] = {
        "iql_dnn": {"agent": "iql", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="forbidden in paradigm 'online_rl'"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_cew_forbidden_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject offline agents like cew."""
    mock_cfg_online.__dict__["methods"] = {
        "cew_dnn": {"agent": "cew", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="(forbidden in paradigm 'online_rl'|not allowed in paradigm 'online_rl')"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_method_missing_model_raises(mock_yaml, mock_cfg_offline):
    """Methods missing required 'model' key must raise ConfigurationError."""
    mock_cfg_offline.__dict__["methods"] = {
        "broken_method": {"agent": "cql"}
    }
    with pytest.raises(ConfigurationError, match="missing required key 'model'"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_offline_rl_method_missing_agent_raises(mock_yaml, mock_cfg_offline):
    """Offline RL methods missing required 'agent' key must raise ConfigurationError."""
    mock_cfg_offline.__dict__["methods"] = {
        "dnn_only": {"model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="requires an agent harness"):
        validate_experiment_config(mock_cfg_offline, "test_exp")

