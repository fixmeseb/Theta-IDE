"""Unit tests for component actions: opening code in editor and uninstalling/deleting components."""
from pathlib import Path
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from frontend.component_actions import (
    find_component_assets,
    DeleteComponentDialog,
)
from frontend.config_viewer import ConfigViewer, SUBTLE_ICON_COLORS
from frontend.components_panel import ComponentsPanel
from frontend.sidetabs import svg_icon


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_svg_icons_exist_and_render(qapp):
    """Test that pencil.svg and trash.svg icons exist and load properly."""
    pencil = svg_icon("pencil", SUBTLE_ICON_COLORS)
    assert not pencil.isNull()

    trash = svg_icon("trash", SUBTLE_ICON_COLORS)
    assert not trash.isNull()


def test_find_component_assets_multi_config_suite(tmp_path):
    """Test resolving a multi-config component suite like Atari."""
    # Setup mock workspace
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "in" / "config" / "env" / "atari").mkdir(parents=True)
    (ws / "in" / "envs" / "atari").mkdir(parents=True)
    (ws / "src").mkdir()

    # Create dummy configs
    c1 = ws / "in" / "config" / "env" / "atari" / "pong.yaml"
    c2 = ws / "in" / "config" / "env" / "atari" / "breakout.yaml"
    c1.write_text("env:\n  name: pong\n")
    c2.write_text("env:\n  name: breakout\n")

    # Create dummy env code
    env_py = ws / "in" / "envs" / "atari" / "env_vectorized.py"
    env_py.write_text("# atari code")

    # Create manifest with installed_from to simulate a Hub-installed package
    meta_file = ws / "in" / "envs" / "atari" / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "atari-env",
        "name": "Atari 2600 Environment Suite",
        "kind": "env",
        "version": "1.0.0",
        "installed_from": "https://raw.githubusercontent.com/theta-hub/packages/atari.zip",
        "target_path": "in/envs/atari",
        "config_paths": [
            "in/config/env/atari/pong.yaml",
            "in/config/env/atari/breakout.yaml",
        ]
    }))

    assets = find_component_assets(c1, "env/atari/pong.yaml", workspace_dir=ws)
    assert assets["is_component"] is True
    assert assets["is_hub_installed"] is True
    assert assets["action_type"] == "uninstall"
    assert assets["comp_name"] == "Atari 2600 Environment Suite"
    assert assets["primary_code_file"] == env_py
    assert ws / "in" / "envs" / "atari" in assets["files_to_delete"]
    assert c1 in assets["files_to_delete"]
    assert c2 in assets["files_to_delete"]


def test_find_component_assets_local_scaffolded_component(tmp_path):
    """Test resolving a locally scaffolded component (not installed from Hub)."""
    ws = tmp_path / "workspace_local"
    ws.mkdir()
    (ws / "in" / "config" / "model").mkdir(parents=True)
    (ws / "src" / "usr" / "models" / "myfirstmlp").mkdir(parents=True)

    cfg = ws / "in" / "config" / "model" / "myfirstmlp.yaml"
    cfg.write_text("type: myfirstmlp\n")

    code_py = ws / "src" / "usr" / "models" / "myfirstmlp" / "myfirstmlp_model.py"
    code_py.write_text("# myfirstmlp code")

    # Local manifest without installed_from
    meta_file = ws / "src" / "usr" / "models" / "myfirstmlp" / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "myfirstmlp",
        "name": "MyfirstmlpModel",
        "kind": "model",
        "version": "0.1.0",
        "target_path": "src/usr/models/myfirstmlp",
        "config_path": "in/config/model/myfirstmlp.yaml",
    }))

    assets = find_component_assets(cfg, "model/myfirstmlp.yaml", workspace_dir=ws)
    assert assets["is_component"] is True
    assert assets["is_hub_installed"] is False
    assert assets["action_type"] == "delete_component"
    assert assets["primary_code_file"] == code_py
    assert ws / "src" / "usr" / "models" / "myfirstmlp" in assets["files_to_delete"]
    assert cfg in assets["files_to_delete"]


def test_find_component_assets_loose_config(tmp_path):
    """Test resolving a standalone config not tied to any component code."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "in" / "config" / "agent").mkdir(parents=True)
    (ws / "src").mkdir()

    cfg = ws / "in" / "config" / "agent" / "my_custom_loose_agent.yaml"
    cfg.write_text("agent: custom\n")

    assets = find_component_assets(cfg, "agent/my_custom_loose_agent.yaml", workspace_dir=ws)
    assert assets["is_component"] is False
    assert assets["is_hub_installed"] is False
    assert assets["action_type"] == "delete_config"
    assert assets["files_to_delete"] == [cfg]


def test_delete_component_dialog(qapp, tmp_path):
    """Test DeleteComponentDialog interactions."""
    files = [tmp_path / "foo.yaml", tmp_path / "src" / "foo"]

    # Local component: should say "Delete Component"
    dlg_local = DeleteComponentDialog("myfirstmlp", files, is_component=True, is_hub_installed=False)
    assert "Delete Component" in dlg_local.windowTitle()
    assert dlg_local.dont_ask_again is False
    dlg_local.chk_dont_ask.setChecked(True)
    dlg_local._on_confirm()
    assert dlg_local.dont_ask_again is True
    dlg_local.deleteLater()

    # Hub component: should say "Uninstall Component"
    dlg_hub = DeleteComponentDialog("Atari", files, is_component=True, is_hub_installed=True)
    assert "Uninstall Component" in dlg_hub.windowTitle()
    dlg_hub.deleteLater()


def test_config_viewer_component_buttons(qapp, tmp_path):
    """Test btn_edit_code and btn_delete_component layout and visibility in ConfigViewer."""
    viewer = ConfigViewer()
    assert hasattr(viewer, "btn_edit_code")
    assert hasattr(viewer, "btn_delete_component")

    # In default mode (not component mode), buttons are hidden
    cfg = tmp_path / "test.yaml"
    cfg.write_text("key: value\n")
    viewer.load_file(cfg, "test.yaml")
    assert viewer.btn_edit_code.isHidden()
    assert viewer.btn_delete_component.isHidden()

    # In component mode, buttons are shown when a file is loaded
    viewer.is_component_mode = True
    viewer.load_file(cfg, "test.yaml")
    assert not viewer.btn_edit_code.isHidden()
    assert not viewer.btn_delete_component.isHidden()

    # Verify relative layout: pencil is to the left of trash
    hb_layout = viewer.header_bar.layout()
    idx_pencil = -1
    idx_trash = -1
    for i in range(hb_layout.count()):
        w = hb_layout.itemAt(i).widget()
        if w is viewer.btn_edit_code:
            idx_pencil = i
        elif w is viewer.btn_delete_component:
            idx_trash = i

    assert idx_pencil != -1
    assert idx_trash != -1
    assert idx_pencil < idx_trash
    assert idx_trash == idx_pencil + 1

    viewer.deleteLater()


def test_components_panel_integration(qapp, tmp_path, monkeypatch):
    """Test ComponentsPanel wiring for editor opening and component deletion."""
    # Setup mock workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg_dir = ws / "in" / "config" / "model"
    cfg_dir.mkdir(parents=True)
    code_dir = ws / "src" / "usr" / "models" / "sample_mlp"
    code_dir.mkdir(parents=True)

    cfg = cfg_dir / "sample_mlp.yaml"
    cfg.write_text("type: sample_mlp\n")

    code_file = code_dir / "sample_mlp_model.py"
    code_file.write_text("# sample model")

    meta_file = code_dir / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "sample_mlp",
        "name": "SampleMLP",
        "kind": "model",
        "version": "0.1.0",
        "target_path": "src/usr/models/sample_mlp",
        "config_path": "in/config/model/sample_mlp.yaml"
    }))

    panel = ComponentsPanel()
    # Point tree to test root
    panel.components_tree.root_dir = ws / "in" / "config"
    panel.on_component_selected(cfg, "model/sample_mlp.yaml")

    # Assert viewer is in component mode and buttons are visible
    assert panel.viewer.is_component_mode is True
    assert not panel.viewer.btn_edit_code.isHidden()
    assert not panel.viewer.btn_delete_component.isHidden()

    # Test open in editor
    opened_targets = []
    monkeypatch.setattr("frontend.component_actions.open_in_editor", lambda p: opened_targets.append(p) or True)
    panel.open_active_component_in_editor()
    assert len(opened_targets) == 1
    assert opened_targets[0] == code_file

    # Test delete component (mock dialog acceptance)
    from PyQt6.QtWidgets import QDialog
    monkeypatch.setattr(DeleteComponentDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    panel.prompt_delete_active_component()

    # Verify both config and source files were deleted
    assert not cfg.exists()
    assert not code_dir.exists()

    panel.deleteLater()
