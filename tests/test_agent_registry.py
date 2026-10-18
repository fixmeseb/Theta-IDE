import pytest

from src.usr.algorithms.agent_registry import (
    AGENT_REGISTRY,
    ALGORITHM_REGISTRY,
    auto_discover,
    get_agent_class,
    get_algorithm_class,
    list_registered_agents,
    list_registered_algorithms,
    register_agent,
    register_algorithm,
)


@pytest.fixture(autouse=True)
def clean_registry():
    auto_discover()
    original = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()
    yield
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(original)


def test_register_agent_single():
    @register_agent("my_algo")
    class DummyAgent:
        pass

    assert "my_algo" in AGENT_REGISTRY
    assert AGENT_REGISTRY["my_algo"] is DummyAgent


def test_register_agent_multiple():
    @register_agent("algo1", "algo2")
    class DummyAgent:
        pass

    assert "algo1" in AGENT_REGISTRY
    assert "algo2" in AGENT_REGISTRY


def test_get_agent_class_exact_match():
    @register_agent("exact")
    class DummyAgent:
        pass

    cls = get_agent_class("exact")
    assert cls is DummyAgent


def test_get_agent_class_longest_prefix():
    @register_agent("base")
    class BaseAgent:
        pass

    @register_agent("base_advanced")
    class AdvancedAgent:
        pass

    cls1 = get_agent_class("base_advanced_suffix")
    assert cls1 is AdvancedAgent

    cls2 = get_agent_class("base_other")
    assert cls2 is BaseAgent


def test_get_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent algorithm"):
        get_agent_class("unknown_algo")


def test_list_registered_agents():
    @register_agent("z_algo", "a_algo")
    class DummyAgent:
        pass

    agents = list_registered_agents()
    assert isinstance(agents, list)
    assert "a_algo" in agents
    assert "z_algo" in agents


def test_algorithm_aliases():
    @register_algorithm("custom_algo")
    class CustomAlgo:
        pass

    assert "custom_algo" in ALGORITHM_REGISTRY
    assert get_algorithm_class("custom_algo") is CustomAlgo
    algos = list_registered_algorithms()
    assert "custom_algo" in algos

