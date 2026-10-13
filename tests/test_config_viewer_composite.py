"""Tests for ConfigViewer composite subcomponents and dynamic hierarchies."""
import shutil
from pathlib import Path
import pytest
import yaml

from PyQt6.QtWidgets import QApplication
from frontend.config_viewer import ConfigViewer, config_tree


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_config_viewer_loads_cql_benchmark(qapp, tmp_path):
    # Copy mimic cql_benchmark.yaml to tmp_path
    src_file = Path("in/config/experiment/mimic/cql_benchmark.yaml")
    assert src_file.is_file()

    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    # Check raw_data loaded
    data = viewer.raw_data
    methods = data.get("methods", {})
    assert "cql_dnn" in methods
    assert "cql_blendrl_cew_dueling_resnet" in methods

    # Verify tensorboard is False or not injected as true
    assert data.get("tensorboard") is not True

    # Test candidate parameters for cql_dnn (pure DNN)
    tree = config_tree()
    from frontend.config_viewer import get_method_candidate_hyperparameters
    dnn_cand = get_method_candidate_hyperparameters("cql", "dnn", tree)
    assert "blend_ent_coef" not in dnn_cand
    assert "hidden_sizes" in dnn_cand

    # Test candidate parameters for blendrl
    blend_cand = get_method_candidate_hyperparameters("cql", "blendrl", tree)
    assert "blend_ent_coef" in blend_cand

    # Test subcomponent addition and editing
    method_name = "cql_blendrl_human_rigid"
    # This method has symbolic, but not neural yet
    viewer._add_method_subcomponent(method_name, "neural")
    rigid_spec = viewer.raw_data["methods"][method_name]
    if isinstance(rigid_spec.get("model"), dict):
        assert "neural" in rigid_spec["model"]["blendrl"]
    else:
        assert "neural" in rigid_spec

    # Edit parameter on symbolic subcomponent
    viewer._on_method_subcomponent_param_edited(method_name, "symbolic", "rules", "custom_rules")
    if isinstance(rigid_spec.get("model"), dict):
        sym_dict = rigid_spec["model"]["blendrl"]["symbolic"]
    else:
        sym_dict = rigid_spec["symbolic"]
    assert sym_dict["rules"] == "custom_rules"

    # Remove parameter
    viewer._remove_method_subcomponent_param(method_name, "symbolic", "rules")
    assert "rules" not in sym_dict

    # Save to disk and check YAML
    viewer.save_to_disk()
    saved_yaml = yaml.safe_load(dest_file.read_text(encoding="utf-8"))
    assert "cql_blendrl_cew_dueling_resnet" in saved_yaml["methods"]
    assert "tensorboard" not in saved_yaml or saved_yaml["tensorboard"] is not True


def test_config_viewer_loads_cartpole_final(qapp, tmp_path):
    src_file = Path("in/config/experiment/cartpole/final_cartpole.yaml")
    assert src_file.is_file()

    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    tree = config_tree()
    from frontend.config_viewer import get_method_candidate_hyperparameters

    # ppo_final_cp has agent ppo, model dnn
    dnn_cand = get_method_candidate_hyperparameters("ppo", "dnn", tree)
    assert "blend_ent_coef" not in dnn_cand
    assert "clip_coef" in dnn_cand

    # ppo_blendrl_final_cp has agent ppo, model blendrl
    blend_cand = get_method_candidate_hyperparameters("ppo", "blendrl", tree)
    assert "blend_ent_coef" in blend_cand
    assert "logic_lr" in blend_cand


def test_scoped_candidate_hyperparameters(qapp):
    from frontend.config_viewer import (
        config_tree,
        get_agent_candidate_hyperparameters,
        get_model_candidate_hyperparameters,
        get_training_candidate_hyperparameters,
    )
    tree = config_tree()

    # Agent candidates
    cql_agent_cand = get_agent_candidate_hyperparameters("cql", tree)
    assert "cql_alpha" in cql_agent_cand
    assert "hidden_sizes" not in cql_agent_cand
    assert "blend_ent_coef" not in cql_agent_cand
    assert "lr" not in cql_agent_cand

    ppo_agent_cand = get_agent_candidate_hyperparameters("ppo", tree)
    assert "clip_coef" in ppo_agent_cand
    assert "cql_alpha" not in ppo_agent_cand

    # Model candidates
    dnn_model_cand = get_model_candidate_hyperparameters("dnn", tree)
    assert "hidden_sizes" in dnn_model_cand
    assert "cql_alpha" not in dnn_model_cand
    assert "blend_ent_coef" not in dnn_model_cand

    blend_model_cand = get_model_candidate_hyperparameters("blendrl", tree)
    assert "blend_ent_coef" in blend_model_cand
    assert "blend_q_values" in blend_model_cand
    assert "logic_lr" in blend_model_cand
    assert "cql_alpha" not in blend_model_cand

    # Training candidates
    train_cand = get_training_candidate_hyperparameters()
    assert "lr" in train_cand
    assert "batch_size" in train_cand
    assert "gamma" in train_cand
    assert "epochs_per_interval" in train_cand


def test_indented_hierarchy_blocks_in_gui(qapp, tmp_path):
    from frontend.config_viewer import IndentedHierarchyBlock

    src_file = Path("in/config/experiment/mimic/cql_benchmark.yaml")
    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    # Find hierarchy blocks rendered in the widget tree
    blocks = viewer.findChildren(IndentedHierarchyBlock)
    assert len(blocks) > 0, "Expected IndentedHierarchyBlock instances in rendered GUI"

    # Verify subtle vertical border is styled
    first_block = blocks[0]
    assert "border-left: 2px solid #504945" in first_block.styleSheet()


def test_add_and_remove_parameters_in_hierarchy(qapp, tmp_path):
    src_file = Path("in/config/experiment/mimic/cql_benchmark.yaml")
    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    method_name = "cql_dnn"
    # Add agent parameter
    viewer._add_method_param(method_name, "cql_alpha", 2.5)
    assert viewer.raw_data["methods"][method_name]["cql_alpha"] == 2.5

    # Add model parameter
    viewer._add_method_param(method_name, "hidden_sizes", [128, 128])
    assert viewer.raw_data["methods"][method_name]["hidden_sizes"] == [128, 128]

    # Remove parameters
    viewer._remove_method_param(method_name, "cql_alpha")
    assert "cql_alpha" not in viewer.raw_data["methods"][method_name]

    viewer._remove_method_param(method_name, "hidden_sizes")
    assert "hidden_sizes" not in viewer.raw_data["methods"][method_name]

