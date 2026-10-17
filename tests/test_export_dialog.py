"""Unit tests for ExportExperimentDialog."""
from pathlib import Path
import pytest
import yaml

from PyQt6.QtWidgets import QApplication
from frontend.export_dialog import ExportExperimentDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


def test_export_dialog_initialization(qapp, tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )

    assert dialog.experiment_id == "tabular_classification/quick_test"
    assert dialog.radio_zip.isChecked()
    assert not dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".zip")
    assert dialog.edit_version.text() == "1.0.0"
    dialog.deleteLater()


def test_export_dialog_format_toggle(qapp, tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )

    # Switch to YAML format
    dialog.radio_yaml.setChecked(True)
    assert dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".yaml")

    # Switch back to ZIP
    dialog.radio_zip.setChecked(True)
    assert not dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".zip")
    dialog.deleteLater()


def test_export_dialog_export_zip(qapp, tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    out_zip = tmp_path / "exported_cartpole.zip"

    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )
    dialog.edit_dest.setText(str(out_zip))
    dialog.edit_version.setText("1.0.5")
    dialog.chk_checkpoints.setChecked(False)

    dialog._on_export_clicked()

    assert dialog.export_result is not None
    assert dialog.export_result.success is True
    assert out_zip.is_file()
    dialog.deleteLater()


def test_export_dialog_export_yaml(qapp, tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    out_yaml = tmp_path / "exported_cartpole.yaml"

    sample_yaml = "total_timesteps: 5000\nseed: 99\n"
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
        raw_yaml_text=sample_yaml,
    )
    dialog.radio_yaml.setChecked(True)
    dialog.edit_dest.setText(str(out_yaml))

    dialog._on_export_clicked()

    assert out_yaml.is_file()
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert data["total_timesteps"] == 5000
    assert data["seed"] == 99
    dialog.deleteLater()
