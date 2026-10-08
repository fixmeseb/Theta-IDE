"""Tests for the method style registry (src/method_registry.py)."""

import pytest

from src.usr.methods.method_style_registry import (
    METHOD_STYLE,
    clean_label,
    get_canonical_method_name,
    get_method_aliases,
    get_style,
    get_style_info,
)


class TestGetStyle:
    """Tests for style lookup with exact, prefix, and fallback matching."""

    def test_exact_match_known_method(self):
        style = get_style("ppo")
        assert style["label"] is not None
        assert "color" in style
        assert "marker" in style
        assert "linestyle" in style

    def test_prefix_match_with_suffix(self):
        """ppo_cp_tuned should resolve via prefix match to ppo."""
        style = get_style("ppo_cp_tuned")
        assert style["label"] is not None

    def test_unknown_method_returns_default_with_name(self):
        """Unknown methods return a default style with label set to the input name."""
        style = get_style("completely_unknown_method_xyz")
        assert style["label"] == "completely_unknown_method_xyz"
        assert style["marker"] == "o"  # default marker

    def test_slash_normalization(self):
        """Slash-separated names should be normalized to underscore."""
        style = get_style("cql/dueling_resnet")
        assert style["label"] == "Dueling ResNet"

    def test_blendrl_style_exists(self):
        """BlendRL is registered as an architecture."""
        style = get_style("blendrl")
        assert "BlendRL" in style["label"]

    def test_iql_style_exists(self):
        style = get_style("iql")
        assert style["label"] is not None

    def test_cql_style_exists(self):
        style = get_style("cql")
        assert style["label"] is not None

    def test_cql_blendrl_cew_fyd_dueling_resnet_style(self):
        style = get_style("cql_blendrl_cew_fyd_dueling_resnet")
        assert style["label"] == "BlendRL CEW+FYD+ResNet"
        assert style["color"] == "#bcbd22"
        assert style["marker"] == "p"


class TestCleanLabel:
    """Tests for the clean_label display name helper."""

    def test_returns_string(self):
        result = clean_label("ppo_cp_tuned")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_known_method_has_readable_label(self):
        assert clean_label("cql") == "DNN"

    def test_unknown_method_uses_raw_name(self):
        assert clean_label("nonexistent_xyz") == "nonexistent_xyz"


class TestGetStyleInfo:
    """Tests for the (color, linestyle, marker) tuple accessor."""

    def test_returns_three_tuple(self):
        color, linestyle, marker = get_style_info("ppo")
        assert color is not None
        assert linestyle in ("-", "--", "-.", ":")
        assert isinstance(marker, str)


class TestGetCanonicalMethodName:
    """Tests for alias resolution to canonical names."""

    def test_cql_aliases_to_cql_dnn(self):
        assert get_canonical_method_name("cql") == "cql_dnn"

    def test_dnn_aliases_to_cql_dnn(self):
        assert get_canonical_method_name("dnn") == "cql_dnn"

    def test_ppo_aliases_to_ppo_dnn(self):
        assert get_canonical_method_name("ppo") == "ppo_dnn"

    def test_slash_to_underscore(self):
        result = get_canonical_method_name("cql/dueling_resnet")
        assert result == "cql_dueling_resnet"

    def test_blendrl_cql_cew_fyd_aliases(self):
        result = get_canonical_method_name("blendrl_cql_cew_fyd_dueling_resnet")
        assert result == "cql_blendrl_cew_fyd_dueling_resnet"

    def test_unknown_passes_through(self):
        assert get_canonical_method_name("my_custom_agent") == "my_custom_agent"


class TestGetMethodAliases:
    """Tests for alias set generation."""

    def test_cql_has_multiple_aliases(self):
        aliases = get_method_aliases("cql")
        assert "cql" in aliases
        assert "cql_dnn" in aliases
        assert "dnn" in aliases

    def test_blendrl_cql_bidirectional(self):
        """Both orderings (blendrl_cql_ and cql_blendrl_) should appear."""
        aliases = get_method_aliases("blendrl_cql_human_neural")
        assert "cql_blendrl_human_neural" in aliases
        assert "blendrl_cql_human_neural" in aliases

    def test_returns_set(self):
        result = get_method_aliases("ppo")
        assert isinstance(result, set)


class TestMethodStyleRegistry:
    """Tests for the METHOD_STYLE dictionary itself."""

    def test_all_entries_have_required_keys(self):
        required_keys = {"label", "color", "marker", "linestyle"}
        for name, style in METHOD_STYLE.items():
            missing = required_keys - set(style.keys())
            assert not missing, f"Method '{name}' missing keys: {missing}"

    def test_no_none_colors(self):
        for name, style in METHOD_STYLE.items():
            assert style["color"] is not None, f"Method '{name}' has None color"

    def test_registry_is_not_empty(self):
        assert len(METHOD_STYLE) > 0
