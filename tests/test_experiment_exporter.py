"""Tests for ExperimentExporter service and HubInstaller roundtrip."""
import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from frontend.hub.installer import HubInstaller
from frontend.hub.models import AuthorInfo, HubComponent, ReleaseInfo
from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions, ExportResult


def test_export_experiment_success(tmp_path):
    workspace_root = Path(__file__).resolve().parents[1]
    exporter = ExperimentExporter(workspace_root=workspace_root)

    out_zip = tmp_path / "test_cartpole_export.zip"
    opts = ExportOptions(
        include_checkpoints=False,
        include_environment=True,
        include_models=True,
        include_rules=True,
        output_path=out_zip,
        author="Tester",
        version="1.2.3",
        description="Test export of CartPole experiment",
    )

    result = exporter.export("cartpole/quick_test", options=opts)
    assert result.success is True
    assert result.archive_path == out_zip
    assert out_zip.is_file()

    # Verify zip archive contents
    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert "README.md" in namelist
        assert "in/config/experiment/cartpole/quick_test.yaml" in namelist
        assert "in/config/experiment/cartpole/_base.yaml" in namelist
        assert "in/config/env/cartpole.yaml" in namelist

        # Inspect manifest
        manifest_raw = zf.read(".theta_component.json").decode("utf-8")
        manifest = json.loads(manifest_raw)
        assert manifest["kind"] == "experiment"
        assert manifest["version"] == "1.2.3"
        assert manifest["author"]["name"] == "Tester"
        assert manifest["environment"] == "cartpole"
        assert "ppo" in manifest["agents"]


def test_export_nonexistent_experiment(tmp_path):
    workspace_root = Path(__file__).resolve().parents[1]
    exporter = ExperimentExporter(workspace_root=workspace_root)
    result = exporter.export("nonexistent_group/nonexistent_exp", options=ExportOptions(output_path=tmp_path / "dummy.zip"))
    assert result.success is False
    assert "not found" in result.error.lower()


def test_hub_installer_roundtrip(tmp_path):
    workspace_root = Path(__file__).resolve().parents[1]
    exporter = ExperimentExporter(workspace_root=workspace_root)

    pkg_zip = tmp_path / "cartpole_pkg.zip"
    res = exporter.export(
        "cartpole/quick_test",
        options=ExportOptions(output_path=pkg_zip, version="2.0.0"),
    )
    assert res.success is True

    # Setup isolated mock workspace
    mock_workspace = tmp_path / "target_workspace"
    mock_data = tmp_path / "target_data"
    mock_workspace.mkdir(parents=True)
    mock_data.mkdir(parents=True)

    installer = HubInstaller(workspace_dir=mock_workspace, data_dir=mock_data)

    comp = HubComponent(
        id="cartpole_quick_test",
        name="cartpole/quick_test",
        kind="experiment",
        version="2.0.0",
        description="Roundtrip test",
        releases={
            "2.0.0": ReleaseInfo(
                tag="v2.0.0",
                url=f"file://{pkg_zip.resolve()}",
                sha256="",
            )
        },
    )

    installed = installer.install(comp)
    assert installed is True

    # Check that experiment yaml was unpacked to target workspace
    dest_exp_yaml = mock_workspace / "in" / "config" / "experiment" / "cartpole" / "quick_test.yaml"
    assert dest_exp_yaml.is_file()

    # Check that env config was unpacked to target workspace
    dest_env_yaml = mock_workspace / "in" / "config" / "env" / "cartpole.yaml"
    assert dest_env_yaml.is_file()

    # Check installer detection
    is_inst, ver = installer.check_installed(comp)
    assert is_inst is True
    assert ver == "2.0.0"
