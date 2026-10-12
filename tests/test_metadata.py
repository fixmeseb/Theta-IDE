import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.app.core.metadata import collect_run_metadata, get_git_info, save_git_diff


def test_get_git_info_success():
    def mock_check_output(args, **kwargs):
        if "rev-parse" in args and "HEAD" in args and "--abbrev-ref" not in args:
            return b"12345abcde\n"
        elif "--abbrev-ref" in args:
            return b"main\n"
        elif "status" in args:
            return b" M some_file.py\n"
        raise subprocess.CalledProcessError(1, cmd=args)

    with patch("subprocess.check_output", side_effect=mock_check_output):
        commit, branch, dirty = get_git_info()
        assert commit == "12345abcde"
        assert branch == "main"
        assert dirty is True


def test_get_git_info_failure():
    with patch("subprocess.check_output", side_effect=Exception("Git not found")):
        commit, branch, dirty = get_git_info()
        assert commit is None
        assert branch is None
        assert dirty is None


def test_save_git_diff_dirty(tmp_path):
    def mock_check_output(args, **kwargs):
        if "diff" in args:
            return b"diff --git a/file b/file\n"
        return b""

    with patch("subprocess.check_output", side_effect=mock_check_output):
        diff_name = save_git_diff(str(tmp_path), git_dirty=True)
        assert diff_name == "git_patch.diff"
        assert (tmp_path / "git_patch.diff").exists()
        assert (tmp_path / "git_patch.diff").read_text() == "diff --git a/file b/file\n"


def test_save_git_diff_clean(tmp_path):
    diff_name = save_git_diff(str(tmp_path), git_dirty=False)
    assert diff_name is None
    assert not (tmp_path / "git_patch.diff").exists()


def test_collect_run_metadata_no_cfg():
    with patch("src.app.core.metadata.get_git_info", return_value=("hash", "dev", False)):
        meta = collect_run_metadata()
        assert meta["git_commit"] == "hash"
        assert meta["git_branch"] == "dev"
        assert meta["git_dirty"] is False
        assert meta["seed"] is None
        assert "system" in meta
        assert "python" in meta["system"]


def test_collect_run_metadata_with_cfg():
    cfg = MagicMock()
    cfg.seed = 42
    cfg.description = "Test note"
    with patch("src.app.core.metadata.get_git_info", return_value=("hash", "dev", True)):
        meta = collect_run_metadata(cfg)
        assert meta["seed"] == 42
        assert meta["description"] == "Test note"
        assert meta["git_dirty"] is True
        assert meta["git_diff_path"] == "git_patch.diff"
