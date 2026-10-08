"""Tests for universal ContractLogger and engine resolution."""
import csv
import json
from pathlib import Path
import tempfile
import pytest

from src.app.core.contract_logger import ContractLogger
from src.app.pipeline.engine import DEFAULT_ENGINE_SCRIPT, resolve_engine


class TestContractLogger:
    def test_log_creates_csv_with_headers(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ContractLogger(log_dir=log_dir, version="version_0")

        logger.log({"eval/reward": 100.0, "losses/total_loss": 0.5}, step=1, transitions=256)
        logger.log({"eval/reward": 150.0, "losses/total_loss": 0.3}, step=2, transitions=512)
        logger.close()

        csv_file = log_dir / "version_0" / "metrics.csv"
        assert csv_file.exists()

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert reader[0]["step"] == "1"
            assert reader[0]["transitions"] == "256"
            assert reader[0]["eval/reward"] == "100.0"
            assert reader[0]["losses/total_loss"] == "0.5"

            assert reader[1]["step"] == "2"
            assert reader[1]["transitions"] == "512"
            assert reader[1]["eval/reward"] == "150.0"

    def test_log_dynamic_column_expansion(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ContractLogger(log_dir=log_dir, version="version_0")

        # Step 1: only reward
        logger.log({"eval/reward": 10.0}, step=1, transitions=100)
        # Step 2: introduces new metric column
        logger.log({"eval/reward": 20.0, "new_metric/accuracy": 0.95}, step=2, transitions=200)
        logger.close()

        csv_file = log_dir / "version_0" / "metrics.csv"
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert "new_metric/accuracy" in reader[0]
            assert reader[0]["new_metric/accuracy"] == ""
            assert reader[1]["new_metric/accuracy"] == "0.95"

    def test_save_runtime_metadata(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ContractLogger(log_dir=log_dir, version="version_0")
        meta_path = logger.save_runtime_metadata(config={"agent": "sb3_ppo"}, training_time_seconds=12.5)
        logger.close()

        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["training_time_seconds"] == 12.5
        assert data["config"]["agent"] == "sb3_ppo"
        assert "timestamp" in data

    def test_convenience_hierarchy_init(self, tmp_path):
        logger = ContractLogger(
            group="csc510",
            experiment_id="test_exp",
            method_name="sb3/ppo_test",
            base_dir=tmp_path,
        )
        logger.log_eval(reward_mean=200.0, reward_std=5.0, step=1, transitions=1000)
        logger.close()

        expected_dir = tmp_path / "csc510" / "test_exp" / "sb3_ppo_test" / "version_0"
        assert (expected_dir / "metrics.csv").exists()


class TestEngineResolution:
    def test_default_returns_train_py(self):
        script, py_bin = resolve_engine(method_cfg={})
        assert script == DEFAULT_ENGINE_SCRIPT
        assert py_bin is None

    def test_custom_script_or_entrypoint(self):
        script, py_bin = resolve_engine(method_cfg={"entrypoint": "custom/run.py"})
        assert script == "custom/run.py"

        script2, _ = resolve_engine(method_cfg={"runner": "custom/runner.py"})
        assert script2 == "custom/runner.py"

    def test_known_engine_mapping(self):
        script_sb3, _ = resolve_engine(method_cfg={"engine": "sb3"})
        assert script_sb3 == "src/usr/methods/sb3_runner.py"

        script_cleanrl, _ = resolve_engine(method_cfg={"engine": "cleanrl"})
        assert script_cleanrl == "src/usr/methods/cleanrl_runner.py"

        script_torch, _ = resolve_engine(method_cfg={"engine": "pytorch"})
        assert script_torch == DEFAULT_ENGINE_SCRIPT

    def test_custom_venv_or_python(self, tmp_path):
        fake_venv = tmp_path / "my_venv"
        bin_dir = fake_venv / "bin"
        bin_dir.mkdir(parents=True)
        py_bin = bin_dir / "python3"
        py_bin.touch()

        script, resolved_py = resolve_engine(method_cfg={"engine": "sb3", "venv": str(fake_venv)})
        assert script == "src/usr/methods/sb3_runner.py"
        assert resolved_py == str(py_bin)

    def test_resolve_from_cfg_agent(self):
        class FakeCfg:
            agent = {"engine": "sb3"}

        script, py_bin = resolve_engine(method_cfg={}, cfg=FakeCfg())
        assert script == "src/usr/methods/sb3_runner.py"
        assert py_bin is None
