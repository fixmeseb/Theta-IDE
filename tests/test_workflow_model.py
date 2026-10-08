"""Unit tests for the string diagram workflow data model and presets."""

import pytest
from src.app.pipeline.workflow.model import (
    Port,
    PortDirection,
    PortType,
    WorkflowGraph,
    WorkflowNode,
    WorkflowString,
)
from src.app.pipeline.workflow.presets import (
    BUILTIN_PRESETS,
    create_model_distillation_preset,
    create_online_vs_offline_preset,
    create_sepsis_reciprocal_preset,
    create_transfer_learning_preset,
    get_preset,
)
from src.app.pipeline.workflow.templates import (
    make_offline_rl_node,
    make_online_rl_node,
    make_supervised_node,
)


def test_port_type_colors_and_names():
    assert PortType.DATASET.color == "#06b6d4"
    assert PortType.CHECKPOINT.color == "#a855f7"
    assert PortType.VALUE_ESTIMATOR.color == "#f59e0b"
    assert PortType.METRICS.color == "#10b981"
    assert "Dataset" in PortType.DATASET.display_name


def test_type_compatibility():
    assert WorkflowGraph.is_type_compatible(PortType.DATASET, PortType.DATASET)
    assert not WorkflowGraph.is_type_compatible(PortType.DATASET, PortType.CHECKPOINT)
    assert WorkflowGraph.is_type_compatible(PortType.GENERIC, PortType.CHECKPOINT)
    assert WorkflowGraph.is_type_compatible(PortType.CHECKPOINT, PortType.GENERIC)


def test_graph_connect_and_validate():
    graph = WorkflowGraph(id="test_g", name="Test Graph")
    n1 = make_online_rl_node("n1", "Online RL")
    n2 = make_offline_rl_node("n2", "Offline RL")

    graph.add_node(n1)
    graph.add_node(n2)

    # Connecting dataset output to dataset input should succeed
    wire = graph.connect("n1", "dataset", "n2", "dataset")
    assert wire.source_node_id == "n1"
    assert wire.target_node_id == "n2"
    assert wire.port_type == PortType.DATASET
    assert len(graph.strings) == 1

    # Validation should pass
    issues = graph.validate()
    assert len(issues) == 0

    # Incompatible connection should raise TypeError
    with pytest.raises(TypeError):
        graph.connect("n1", "checkpoint", "n2", "dataset")


def test_graph_serialization_yaml():
    preset = create_online_vs_offline_preset()
    yaml_str = preset.to_yaml()
    assert "online_ppo" in yaml_str
    assert "offline_iql" in yaml_str

    restored = WorkflowGraph.from_yaml(yaml_str)
    assert len(restored.nodes) == 3
    assert len(restored.strings) == 3
    assert restored.validate() == []


def test_topological_sort_levels():
    preset = create_online_vs_offline_preset()
    levels = preset.topological_levels()
    assert len(levels) >= 2
    # First level must contain online_ppo
    assert "online_ppo" in levels[0]
    # offline_iql must depend on online_ppo
    assert any("offline_iql" in lvl for lvl in levels[1:])


def test_all_builtin_presets():
    for pid, factory in BUILTIN_PRESETS.items():
        g = factory()
        assert isinstance(g, WorkflowGraph)
        assert len(g.nodes) >= 2
        assert len(g.strings) >= 1
        # All builtin forward connections should be type valid
        issues = g.validate()
        assert len(issues) == 0, f"Preset {pid} had validation errors: {issues}"
