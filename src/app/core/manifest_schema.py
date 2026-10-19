"""Formal JSON Schema definition and validator for .theta_component.json manifests."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

COMPONENT_MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ThetaComponentManifest",
    "description": "Manifest metadata describing an installable Theta-IDE research component or plugin.",
    "type": "object",
    "required": ["id", "name", "kind", "version"],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Unique identifier for the component (alphanumeric, dashes, underscores).",
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable display title.",
        },
        "kind": {
            "type": "string",
            "enum": [
                "model",
                "method",
                "agent",
                "env",
                "datamodule",
                "data",
                "evaluator",
                "eval",
                "plotter",
                "plot",
                "experiment",
                "plugin",
                "suite",
                "package",
            ],
            "description": "Component archetype.",
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+.*$",
            "description": "Semantic version string (e.g. 1.0.0 or 0.1.0-alpha).",
        },
        "description": {
            "type": "string",
            "description": "Detailed explanation of the component's functionality.",
        },
        "author": {
            "type": ["object", "string"],
            "description": "Author or research lab information.",
        },
        "target_path": {
            "type": "string",
            "description": "Relative workspace destination directory.",
        },
        "config_path": {
            "type": "string",
            "description": "Primary YAML configuration path in in/config/.",
        },
        "config_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of configuration YAML files bundled with this component.",
        },
        "compatibility": {
            "type": "object",
            "description": "Software version constraints (e.g. theta, torch, python).",
        },
        "dependencies": {
            "type": "object",
            "properties": {
                "pip": {"type": "array", "items": {"type": "string"}},
                "components": {"type": "array", "items": {"type": "string"}},
            },
        },
        "created_at": {
            "type": "string",
            "description": "ISO-8601 timestamp of package creation.",
        },
        "paradigms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Learning paradigms compatible with this component (e.g. ['online_rl', 'offline_rl', 'supervised']).",
        },
    },
}

SEMVER_REGEX = re.compile(r"^\d+\.\d+\.\d+.*$")
ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\./]+$")
VALID_KINDS = {
    "model",
    "method",
    "agent",
    "env",
    "datamodule",
    "data",
    "evaluator",
    "eval",
    "plotter",
    "plot",
    "experiment",
    "plugin",
    "suite",
    "package",
}


def validate_manifest_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate component manifest dictionary against schema.

    Uses `jsonschema` if available, falling back to a pure-Python schema validator.

    Returns:
        (True, None) if valid.
        (False, error_message) if invalid.
    """
    if not isinstance(data, dict):
        return False, "Manifest must be a JSON object dictionary."

    # Try standard jsonschema library first
    try:
        import jsonschema
        try:
            jsonschema.validate(instance=data, schema=COMPONENT_MANIFEST_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, f"Manifest schema validation error at '{'.'.join(str(p) for p in err.path)}': {err.message}"
    except ImportError:
        pass

    # Pure Python fallback validation
    for req in ("id", "name", "kind", "version"):
        if req not in data:
            return False, f"Missing required manifest field: '{req}'"
        if not data[req]:
            return False, f"Required field '{req}' cannot be empty."

    comp_id = str(data["id"])
    if not ID_REGEX.match(comp_id):
        return False, f"Invalid 'id': '{comp_id}'. Must contain alphanumeric characters, underscores, dashes, or slashes."

    kind = str(data["kind"]).lower()
    if kind not in VALID_KINDS:
        return False, f"Invalid 'kind': '{kind}'. Allowed kinds: {sorted(VALID_KINDS)}"

    ver = str(data["version"])
    if not SEMVER_REGEX.match(ver):
        return False, f"Invalid 'version': '{ver}'. Must adhere to semantic versioning (e.g. 1.0.0)."

    return True, None


def validate_manifest_file(manifest_path: str | Path) -> Tuple[bool, Optional[str]]:
    """Validate a .theta_component.json file from disk."""
    path = Path(manifest_path)
    if not path.is_file():
        return False, f"Manifest file not found: {path}"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse JSON manifest '{path}': {exc}"

    return validate_manifest_data(data)
