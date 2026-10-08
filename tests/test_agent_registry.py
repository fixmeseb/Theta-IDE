import pytest

from src.usr.methods.agent_registry import AGENT_REGISTRY, get_agent_class, list_registered_agents, register_agent


@pytest.fixture(autouse=True)
def clean_registry():
    from src.usr.methods.agent_registry import auto_discover

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
