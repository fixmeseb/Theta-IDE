"""Unit tests for formal manifest schema validation."""
import json
from pathlib import Path
import pytest

from src.app.core.manifest_schema import (
    COMPONENT_MANIFEST_SCHEMA,
    validate_manifest_data,
    validate_manifest_file,
)


def test_valid_manifest():
    valid = {
        "id": "my_custom_model",
        "name": "My Custom Policy",
        "kind": "model",
        "version": "1.0.0",
        "description": "Valid model component",
        "author": {"name": "Lab Researcher"},
        "compatibility": {"theta": ">=2.0.0"},
    }
    is_valid, err = validate_manifest_data(valid)
    assert is_valid is True
    assert err is None


def test_missing_required_fields():
    invalid = {
        "id": "missing_kind",
        "name": "Missing Kind",
        # missing kind
        "version": "1.0.0",
    }
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "kind" in err.lower()


def test_invalid_kind_enum():
    invalid = {
        "id": "bad_kind",
        "name": "Bad Kind",
        "kind": "not_a_real_kind",
        "version": "1.0.0",
    }
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "kind" in err.lower()


def test_invalid_semver():
    invalid = {
        "id": "bad_ver",
        "name": "Bad Version",
        "kind": "model",
        "version": "invalid-version-string",
    }
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "version" in err.lower()


def test_validate_manifest_file(tmp_path):
    manifest_path = tmp_path / ".theta_component.json"
    manifest_path.write_text(
        json.dumps({
            "id": "file_test",
            "name": "File Test",
            "kind": "env",
            "version": "0.2.1",
        }),
        encoding="utf-8",
    )
    is_valid, err = validate_manifest_file(manifest_path)
    assert is_valid is True
    assert err is None


def test_validate_nonexistent_manifest_file(tmp_path):
    is_valid, err = validate_manifest_file(tmp_path / "missing.json")
    assert is_valid is False
    assert "not found" in err.lower()
