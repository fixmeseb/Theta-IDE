"""Unit tests for backend hot-reloading service."""
import sys
import types
from unittest.mock import MagicMock, patch
import pytest

from src.app.core.hot_reload import reload_component_modules


def test_reload_component_modules_empty():
    # If no modules match prefix, returns empty list without error
    reloaded = reload_component_modules(target_prefix="nonexistent.prefix.test.")
    assert reloaded == []


def test_reload_component_modules_reloads_matching():
    # Create fake dynamic module in sys.modules
    fake_mod_name = "src.usr.models.test_fake_mod"
    fake_mod = types.ModuleType(fake_mod_name)
    fake_mod.VALUE = 42
    sys.modules[fake_mod_name] = fake_mod

    try:
        with patch("importlib.reload", return_value=fake_mod) as mock_reload:
            reloaded = reload_component_modules(target_prefix="src.usr.models.test_fake_mod")
            assert fake_mod_name in reloaded
            mock_reload.assert_called_once_with(fake_mod)
    finally:
        sys.modules.pop(fake_mod_name, None)
