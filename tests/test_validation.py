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
        eval_episodes=100,
        group="",
        experiment_id=""
    )
    cfg.env.name = "cartpole"
    return cfg

@pytest.fixture
def mock_cfg_offline():
    cfg = MockCfg(
        paradigm="offline_rl",
        methods={"iql_mimic": {"agent": "iql", "model": "dnn"}},
        eval_episodes=0,
        group="",
        experiment_id=""
    )
    cfg.env.name = "mimic"
    return cfg

@pytest.fixture
def mock_cfg_supervised():
    cfg = MockCfg(
        paradigm="supervised",
        methods={"ep_mlp": {"model": "mlp_classifier"}},
        eval_episodes=0,
        group="early_prediction",
        experiment_id="quick_test"
    )
    cfg.env.name = "cartpole"
    return cfg







@pytest.fixture(autouse=True)
def _register_test_agents():
    """Register test agent components with declared supported paradigms."""
    from src.usr.algorithms.agent_registry import AGENT_REGISTRY, register_agent
    from src.usr.algorithms.base_agent import OfflineAgentBase, OnlineAgentBase

    old_registry = AGENT_REGISTRY.copy()

    @register_agent("ppo")
    class MockPPO(OnlineAgentBase):
        def get_action_and_value(self, obs, logic_obs=None, action=None):
            pass

        def get_value(self, obs, logic_obs=None):
            pass

    @register_agent("cql")
    class MockCQL(OfflineAgentBase):
        def get_action_and_value(self, obs, logic_obs=None, action=None):
            pass

        def get_value(self, obs, logic_obs=None):
            pass

    @register_agent("iql")
    class MockIQL(OfflineAgentBase):
        def get_action_and_value(self, obs, logic_obs=None, action=None):
            pass

        def get_value(self, obs, logic_obs=None):
            pass

    @register_agent("cew")
    class MockCEW(OfflineAgentBase):
        def get_action_and_value(self, obs, logic_obs=None, action=None):
            pass

        def get_value(self, obs, logic_obs=None):
            pass

    @register_agent("unsupported_algo")
    class MockUnsupported(OfflineAgentBase):
        def get_action_and_value(self, obs, logic_obs=None, action=None):
            pass

        def get_value(self, obs, logic_obs=None):
            pass

    yield

    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(old_registry)


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
def test_eval_episodes_offline_rl(mock_yaml, mock_cfg_offline):
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
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


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={"eval_episodes": 5})
def test_explicit_eval_episodes_in_raw_yaml(mock_yaml, mock_cfg_offline):
    mock_cfg_offline.__dict__.update({
        "paradigm": "offline_rl",
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
    with pytest.raises(ConfigurationError, match="(not allowed in paradigm 'offline_rl'|forbidden in paradigm 'offline_rl')"):
        validate_experiment_config(mock_cfg_offline, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_disallowed_agent_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject agents not in allowed_agents."""
    mock_cfg_online.__dict__["methods"] = {
        "bad_method": {"agent": "unsupported_algo", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="(not allowed in paradigm 'online_rl'|forbidden in paradigm 'online_rl')"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_cql_forbidden_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject offline agents like cql."""
    mock_cfg_online.__dict__["methods"] = {
        "cql_dnn": {"agent": "cql", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="(not allowed in paradigm 'online_rl'|forbidden in paradigm 'online_rl')"):
        validate_experiment_config(mock_cfg_online, "test_exp")


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_iql_forbidden_in_online_rl(mock_yaml, mock_cfg_online):
    """Online RL paradigm must reject offline agents like iql."""
    mock_cfg_online.__dict__["methods"] = {
        "iql_dnn": {"agent": "iql", "model": "dnn"}
    }
    with pytest.raises(ConfigurationError, match="(not allowed in paradigm 'online_rl'|forbidden in paradigm 'online_rl')"):
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


def test_agent_supported_paradigms_from_installed_manifest(tmp_path, monkeypatch):
    """Component manifests (.theta_component.json) in methods directory define supported paradigms."""
    import json
    from src.app.pipeline.validation import _get_agent_supported_paradigms

    method_dir = tmp_path / "src" / "usr" / "algorithms" / "custom_agent"
    method_dir.mkdir(parents=True, exist_ok=True)
    manifest = method_dir / ".theta_component.json"
    manifest.write_text(json.dumps({"id": "custom_agent", "paradigms": ["offline_rl"]}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    paradigms = _get_agent_supported_paradigms("custom_agent")
    assert paradigms == ["offline_rl"]


def test_find_missing_components_detects_uninstalled():
    """find_missing_components identifies uninstalled agents, models, submodels, and environments."""
    from src.app.pipeline.validation import find_missing_components

    spec = {
        "env": "nonexistent_env_xyz",
        "methods": {
            "test_m": {
                "agent": "nonexistent_agent_abc",
                "model": {
                    "blendrl": {
                        "symbolic": "nsfr",
                        "neural": "dueling_resnet",
                    }
                }
            }
        }
    }
    missing = find_missing_components(spec)
    assert any("environment: nonexistent_env_xyz" in m for m in missing)
    assert any("agent: nonexistent_agent_abc" in m for m in missing)
    assert any("model: blendrl" in m for m in missing)
    assert any("symbolic: nsfr" in m for m in missing)


@patch("src.app.pipeline.validation._load_raw_experiment_yaml", return_value={})
def test_missing_components_raises_configuration_error(mock_yaml, mock_cfg_online):
    """validate_experiment_config must raise ConfigurationError when components are missing."""
    mock_cfg_online.__dict__["methods"] = {
        "bad_model_method": {"agent": "ppo", "model": "nonexistent_model_123"}
    }
    with pytest.raises(ConfigurationError, match="references missing component"):
        validate_experiment_config(mock_cfg_online, "test_exp")


