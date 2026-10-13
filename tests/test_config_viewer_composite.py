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


def test_top_box_defaults_hidden_until_added(qapp, tmp_path):
    src_file = Path("in/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # None of the 4 flags are in final_cartpole.yaml -> none should be rendered
    assert viewer.chk_tb is None
    assert viewer.chk_save_ds is None
    assert viewer.chk_recover is None
    assert viewer.chk_no_plot is None

    # Add option via raw_data or adder
    viewer.raw_data["tensorboard"] = True
    viewer._render_boxes()

    # Now tensorboard is rendered as a visible checkbox
    assert viewer.chk_tb is not None
    assert viewer.chk_tb.isChecked() is True
    # The others remain unrendered
    assert viewer.chk_save_ds is None
    assert viewer.chk_recover is None
    assert viewer.chk_no_plot is None

    # Remove option
    viewer.raw_data.pop("tensorboard", None)
    viewer._render_boxes()
    assert viewer.chk_tb is None


def test_method_card_duplicate_and_plus_icons(qapp, tmp_path):
    from PyQt6.QtWidgets import QToolButton
    src_file = Path("in/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # 1. Verify old add method dropdown and button are completely removed
    assert not hasattr(viewer, "combo_new_method")
    assert not hasattr(viewer, "btn_add_method")

    # 2. Check that method cards have duplicate and plus buttons
    tool_buttons = viewer.findChildren(QToolButton)
    copy_buttons = [b for b in tool_buttons if "Duplicate method" in b.toolTip()]
    plus_buttons = [b for b in tool_buttons if "Add new method" in b.toolTip() or "Add method" in b.toolTip()]
    assert len(copy_buttons) >= 2, "Expected duplicate buttons on method cards"
    assert len(plus_buttons) >= 2, "Expected plus buttons on method cards"

    # 3. Test duplicate_method
    initial_methods = list(viewer.raw_data["methods"].keys())
    assert "ppo_final_cp" in initial_methods
    dup_name = viewer.duplicate_method("ppo_final_cp")
    assert dup_name == "ppo_final_cp_copy"
    assert dup_name in viewer.raw_data["methods"]
    assert viewer.raw_data["methods"][dup_name]["agent"] == viewer.raw_data["methods"]["ppo_final_cp"]["agent"]

    # Duplicate again generates incremental suffix
    dup2 = viewer.duplicate_method("ppo_final_cp")
    assert dup2 == "ppo_final_cp_copy_2"
    assert dup2 in viewer.raw_data["methods"]


def test_add_new_method_flow_empty_and_invalid(qapp, tmp_path):
    from frontend.config_viewer import InlineTitleEditor
    src_file = Path("in/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # Start adding new method
    viewer._start_add_new_method()
    assert viewer._pending_new_method is True

    # Find the editor for the new method
    editors = [e for e in viewer.findChildren(InlineTitleEditor) if e.objectName() == "newMethodEditor"]
    assert len(editors) == 1
    new_editor = editors[0]
    assert new_editor.text() == ""
    assert new_editor.placeholderText() == "Method name..."

    # Validate name rules:
    existing_keys = set(viewer.raw_data["methods"].keys())
    # Empty name
    assert viewer._validate_method_name("", existing_keys) == "Method name cannot be empty."
    assert viewer._validate_method_name("   ", existing_keys) == "Method name cannot be empty."
    # Reserved name 'params'
    assert "params" in viewer._validate_method_name("params", existing_keys)
    # Duplicate name
    assert "already exists" in viewer._validate_method_name("ppo_final_cp", existing_keys)
    # Invalid characters (spaces, punctuation)
    assert "invalid characters" in viewer._validate_method_name("invalid name with spaces", existing_keys)
    assert "invalid characters" in viewer._validate_method_name("bad/name", existing_keys)
    assert "invalid characters" in viewer._validate_method_name("bad:name", existing_keys)

    # Cancel pending method: empty name creates no method
    viewer._cancel_new_method()
    qapp.processEvents()
    assert viewer._pending_new_method is False
    assert len(viewer.findChildren(InlineTitleEditor, "newMethodEditor")) == 0


def test_add_new_method_valid_empty_method(qapp, tmp_path):
    from frontend.widgets import ComboBox
    from frontend.config_viewer import IndentedHierarchyBlock
    src_file = Path("in/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # Start adding new method
    viewer._start_add_new_method()
    # Add valid method name
    viewer.raw_data.setdefault("methods", {})["custom_empty_method"] = {}
    viewer._pending_new_method = False
    viewer._render_boxes()

    assert "custom_empty_method" in viewer.raw_data["methods"]
    assert viewer.raw_data["methods"]["custom_empty_method"] == {}

    # Inspect the rendered card for custom_empty_method
    combos = viewer.findChildren(ComboBox)
    # Check that placeholders "Select agent..." and "Select model..." are used with -1 index
    agent_combos = [c for c in combos if c.placeholderText() == "Select agent..."]
    model_combos = [c for c in combos if c.placeholderText() == "Select model..."]
    assert len(agent_combos) >= 1
    assert len(model_combos) >= 1
    # For custom_empty_method, nothing is selected
    assert any(c.currentIndex() == -1 for c in agent_combos)
    assert any(c.currentIndex() == -1 for c in model_combos)


