"""Unit tests for PluginScaffolder service and ScaffoldPluginDialog GUI."""
from pathlib import Path
import pytest

from PyQt6.QtWidgets import QApplication
from frontend.scaffold_dialog import ScaffoldPluginDialog
from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


def test_scaffold_model_standard(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(
        name="simple_mlp",
        kind="model",
        is_composite=False,
        author="Unit Tester",
    )
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "simple_mlp_model.py").is_file()
    assert (res.target_dir / "__init__.py").is_file()
    assert (res.target_dir / ".theta_component.json").is_file()
    assert (res.target_dir / "pyproject.toml").is_file()
    assert (tmp_path / "in" / "config" / "model" / "simple_mlp.yaml").is_file()


def test_scaffold_model_composite(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(
        name="custom_composite",
        kind="model",
        is_composite=True,
    )
    res = scaffolder.scaffold(opts)
    assert res.success is True
    code = (res.target_dir / "custom_composite_model.py").read_text(encoding="utf-8")
    assert "get_model_slots" in code
    assert "ModelSlot" in code
    assert "compute_auxiliary_losses" in code
    assert "configure_optimizer_groups" in code


def test_scaffold_agent(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="soft_actor_critic", kind="agent")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "soft_actor_critic_agent.py").is_file()
    assert (tmp_path / "in" / "config" / "agent" / "soft_actor_critic.yaml").is_file()


def test_scaffold_environment(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="quadrotor_sim", kind="env")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "env.py").is_file()
    assert (tmp_path / "in" / "config" / "env" / "quadrotor_sim.yaml").is_file()
    assert (tmp_path / "in" / "rules" / "quadrotor_sim" / "default.rules").is_file()


def test_scaffold_datamodule(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="trajectory_streamer", kind="datamodule")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "trajectory_streamer_data_module.py").is_file()


def test_scaffold_evaluator(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="reachability_checker", kind="evaluator")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "reachability_checker_evaluator.py").is_file()


def test_scaffold_invalid_kind(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="invalid_comp", kind="unsupported_kind")
    res = scaffolder.scaffold(opts)
    assert res.success is False
    assert "Invalid component kind" in res.error


def test_scaffold_dialog_gui(qapp, tmp_path):
    dialog = ScaffoldPluginDialog(workspace_root=tmp_path)
    dialog.edit_name.setText("test_dialog_model")
    dialog.chk_composite.setChecked(True)

    # Trigger scaffolding
    dialog._on_scaffold_clicked()

    assert dialog.scaffold_result is not None
    assert dialog.scaffold_result.success is True
    assert (dialog.scaffold_result.target_dir / "test_dialog_model_model.py").is_file()
    dialog.deleteLater()
