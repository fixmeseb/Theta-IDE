"""Unified method style registry for the NeSyRL project.

Single source of truth for method display names, colors, line styles, and markers.
Both the plotting system (plot/base.py) and EP evaluation (eval.py) import from here.

To add a new architecture, add ONE entry to METHOD_STYLE below.
"""

import re
from typing import Optional

METHOD_STYLE = {
    # Clean harness/architecture keys
    "cql": {"label": "DNN", "color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "cql_dnn": {"label": "DNN", "color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "dnn": {"label": "DNN", "color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "cql_dueling_resnet": {"label": "Dueling ResNet", "color": "#08519c", "marker": "D", "linestyle": "-"},
    "dueling_resnet": {"label": "Dueling ResNet", "color": "#08519c", "marker": "D", "linestyle": "-"},
    "cql_transformer": {"label": "Transformer", "color": "#e377c2", "marker": "p", "linestyle": "-"},
    "transformer": {"label": "Transformer", "color": "#e377c2", "marker": "p", "linestyle": "-"},
    "lstm": {"label": "LSTM", "color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "ep_lstm": {"label": "EP LSTM", "color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "ep_transformer": {"label": "EP Transformer", "color": "#e377c2", "marker": "p", "linestyle": "-"},
    "cql_blendrl_human_neural": {
        "label": "BlendRL (MLP, Human, MLP)",
        "color": "#fdbf6f",
        "marker": "s",
        "linestyle": "-",
    },
    "blendrl_cql_human_neural": {
        "label": "BlendRL (MLP, Human, MLP)",
        "color": "#fdbf6f",
        "marker": "s",
        "linestyle": "-",
    },
    "cql_blendrl_human_neural_logic": {
        "label": "BlendRL (MLP, Human, Human)",
        "color": "#b2df8a",
        "marker": "^",
        "linestyle": "--",
    },
    "blendrl_cql_human_neural_logic": {
        "label": "BlendRL (MLP, Human, Human)",
        "color": "#b2df8a",
        "marker": "^",
        "linestyle": "--",
    },
    "cql_blendrl_human_dueling_resnet": {
        "label": "BlendRL (ResNet, Human, MLP)",
        "color": "#41b6c4",
        "marker": "o",
        "linestyle": "-",
    },
    "blendrl_cql_human_dueling_resnet": {
        "label": "BlendRL (ResNet, Human, MLP)",
        "color": "#41b6c4",
        "marker": "o",
        "linestyle": "-",
    },
    "cql_blendrl_human_dueling_resnet_logic": {
        "label": "BlendRL (Dueling ResNet, Human, Human)",
        "color": "#bc80bd",
        "marker": "D",
        "linestyle": "-.",
    },
    "blendrl_cql_human_dueling_resnet_logic": {
        "label": "BlendRL (Dueling ResNet, Human, Human)",
        "color": "#bc80bd",
        "marker": "D",
        "linestyle": "-.",
    },
    "cql_blendrl_human_dueling_resnet_rigid": {
        "label": "BlendRL ResNet+Rigid",
        "color": "#d95f02",
        "marker": "^",
        "linestyle": "-",
    },
    "blendrl_cql_human_dueling_resnet_rigid": {
        "label": "BlendRL ResNet+Rigid",
        "color": "#d95f02",
        "marker": "^",
        "linestyle": "-",
    },
    "cql_blendrl_human_dueling_resnet_continuous": {
        "label": "BlendRL ResNet+Soft",
        "color": "#018571",
        "marker": "v",
        "linestyle": "-",
    },
    "blendrl_cql_human_dueling_resnet_continuous": {
        "label": "BlendRL ResNet+Soft",
        "color": "#018571",
        "marker": "v",
        "linestyle": "-",
    },
    "cql_blendrl_human_rigid": {"label": "BlendRL Rigid Logic", "color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_continuous": {
        "label": "BlendRL Soft Logic",
        "color": "#17becf",
        "marker": "v",
        "linestyle": "-",
    },
    "cql_blendrl_human_transformer": {
        "label": "BlendRL Human+Transformer",
        "color": "#8c564b",
        "marker": "h",
        "linestyle": "-",
    },
    "blendrl_cql_human_transformer": {
        "label": "BlendRL Human+Transformer",
        "color": "#8c564b",
        "marker": "h",
        "linestyle": "-",
    },
    "cql_blendrl_cross_attention": {
        "label": "BlendRL Cross-Attention",
        "color": "#9467bd",
        "marker": "*",
        "linestyle": "-",
    },
    "blendrl_cql_cross_attention": {
        "label": "BlendRL Cross-Attention",
        "color": "#9467bd",
        "marker": "*",
        "linestyle": "-",
    },
    "cql_blendrl_human_cew": {"label": "BlendRL Human+CEW", "color": "#74c476", "marker": "^", "linestyle": "--"},
    "blendrl_cql_human_cew": {"label": "BlendRL Human+CEW", "color": "#74c476", "marker": "^", "linestyle": "--"},
    "cql_blendrl_cew_dueling_resnet": {
        "label": "BlendRL CEW+ResNet",
        "color": "#2ca02c",
        "marker": "h",
        "linestyle": "-",
    },
    "blendrl_cql_cew_dueling_resnet": {
        "label": "BlendRL CEW+ResNet",
        "color": "#2ca02c",
        "marker": "h",
        "linestyle": "-",
    },
    "cql_blendrl_cew_fyd_dueling_resnet": {
        "label": "BlendRL CEW+FYD+ResNet",
        "color": "#bcbd22",
        "marker": "p",
        "linestyle": "-",
    },
    "blendrl_cql_cew_fyd_dueling_resnet": {
        "label": "BlendRL CEW+FYD+ResNet",
        "color": "#bcbd22",
        "marker": "p",
        "linestyle": "-",
    },
    "cql_blendrl_human_cew_dueling_resnet": {
        "label": "BlendRL Human+CEW+ResNet",
        "color": "#6a3d9a",
        "marker": "p",
        "linestyle": "-",
    },
    "blendrl_cql_human_cew_dueling_resnet": {
        "label": "BlendRL Human+CEW+ResNet",
        "color": "#6a3d9a",
        "marker": "p",
        "linestyle": "-",
    },
    "cql_blendrl_cew_only": {"label": "BlendRL CEW Only", "color": "#d62728", "marker": "D", "linestyle": "-"},
    "blendrl_cql_cew_only": {"label": "BlendRL CEW Only", "color": "#d62728", "marker": "D", "linestyle": "-"},
    "iql": {"label": "IQL (Neural)", "color": "#08519c", "marker": "d", "linestyle": "-"},
    "iql_dnn": {"label": "IQL (Neural)", "color": "#1f77b4", "marker": "d", "linestyle": "-"},
    "iql_blendrl": {"label": "BlendRL IQL", "color": "#d62728", "marker": "s", "linestyle": "-"},
    "iql_blendrl_human_neural": {"label": "BlendRL Human+Neural", "color": "#d62728", "marker": "s", "linestyle": "-"},
    "ppo": {"label": "PPO (Neural)", "color": "#1f77b4", "marker": "o", "linestyle": "--"},
    "ppo_dnn": {"label": "PPO (Neural)", "color": "black", "marker": "o", "linestyle": "--"},
    "ppo_blendrl": {"label": "BlendRL (Logic+Neural)", "color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "ppo_blendrl_human_neural": {"label": "BlendRL Human+Neural", "color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "blendrl": {"label": "BlendRL (Logic+Neural)", "color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "cew_base": {"label": "CEW", "color": "#e7298a", "marker": "h", "linestyle": "-"},
    "cew_fyd": {"label": "CEW+FYD", "color": "#bcbd22", "marker": "p", "linestyle": "-"},
    "clinician": {"label": "Clinician (Dataset)", "color": "#756bb1", "marker": "X", "linestyle": "-"},
}

_DEFAULT_STYLE = {"label": None, "color": None, "marker": "o", "linestyle": "-"}


def get_style(name: str, style_override: dict | None = None) -> dict:
    """Look up style by exact match, canonical name, or longest prefix match.
    If style_override is provided, its fields take precedence.
    """
    raw = str(name)
    normalized = raw.replace("/", "_")
    canon = get_canonical_method_name(normalized)

    base = None
    for cand in [canon, normalized, raw]:
        if cand in METHOD_STYLE:
            base = dict(METHOD_STYLE[cand])
            break

    if base is None:
        # Prefix match: longest key that is a prefix of candidate wins
        for cand in [canon, normalized, raw]:
            for key in sorted(METHOD_STYLE.keys(), key=len, reverse=True):
                if cand.startswith(key + "_") or cand == key:
                    base = dict(METHOD_STYLE[key])
                    break
            if base is not None:
                break

    if base is None:
        base = {**_DEFAULT_STYLE, "label": name}

    if style_override and isinstance(style_override, dict):
        for k in ("label", "color", "marker", "linestyle"):
            if style_override.get(k) is not None:
                base[k] = style_override[k]

    return base


def clean_label(name: str, style_override: dict | None = None) -> str:
    """Return human-readable display label for a method name."""
    return str(get_style(name, style_override=style_override)["label"])


def get_style_info(name: str, style_override: dict | None = None) -> tuple[str | None, str, str]:
    """Return (color, linestyle, marker) tuple for matplotlib plotting."""
    s = get_style(name, style_override=style_override)
    return s["color"], s["linestyle"], s["marker"]


# Method-name aliases, each stated once; get_canonical_method_name and get_method_aliases both read them.
# Base methods and their short names. Each canonical name is also known in "algo/model" form (e.g. "cql/dnn").
_SHORT_NAMES = {
    "cql_dnn": ("cql", "dnn"),
    "cql_dueling_resnet": ("dueling_resnet",),
    "cql_transformer": ("transformer",),
    "iql_dnn": ("iql",),
    "ppo_dnn": ("ppo",),
}
# Historical display names.
_DISPLAY_NAMES = {
    "cql (standard mlp)": "cql_dnn",
    "cql (dueling resnet)": "cql_dueling_resnet",
    "cql (transformer)": "cql_transformer",
}
# BlendRL methods registered as "<algo>_blendrl_<variant>" but also written "blendrl_<algo>_<variant>".
_BLENDRL_REORDERED = (
    "cql_blendrl_human_neural",
    "cql_blendrl_human_transformer",
    "cql_blendrl_human_cew",
    "cql_blendrl_cew_only",
    "cql_blendrl_cew_dueling_resnet",
    "cql_blendrl_cew_fyd_dueling_resnet",
    "iql_blendrl_human_neural",
    "ppo_blendrl_human_neural",
)


def _blendrl_first(canonical: str) -> str:
    """'cql_blendrl_human_neural' -> 'blendrl_cql_human_neural'."""
    algo, rest = canonical.split("_blendrl_", 1)
    return f"blendrl_{algo}_{rest}"


_ALIASES = {
    **{_blendrl_first(canonical): canonical for canonical in _BLENDRL_REORDERED},
    **{short: canonical for canonical, shorts in _SHORT_NAMES.items() for short in shorts},
    **_DISPLAY_NAMES,
    # Two canonical names were also listed as keys; that only matters for upper-case input ("CQL_TRANSFORMER").
    "cql_dueling_resnet": "cql_dueling_resnet",
    "cql_transformer": "cql_transformer",
}


def get_canonical_method_name(name: str) -> str:
    """Map method aliases and historical display names to canonical registered name."""
    s = str(name).replace("/", "_")
    if s in _ALIASES:
        return _ALIASES[s]
    s_lower = s.lower()
    if s_lower in _ALIASES:
        return _ALIASES[s_lower]
    return s


def get_method_aliases(name: str) -> set:
    """Return all known aliases for a given method name."""
    canon = get_canonical_method_name(name)
    raw = str(name).replace("/", "_")
    aliases = {name, canon, raw}
    if "cql_blendrl_" in canon:
        aliases.add(canon.replace("cql_blendrl_", "blendrl_cql_"))
    elif "blendrl_cql_" in canon:
        aliases.add(canon.replace("blendrl_cql_", "cql_blendrl_"))
    if canon in _SHORT_NAMES:
        aliases.update(_SHORT_NAMES[canon])
        aliases.add(canon.replace("_", "/", 1))
    return aliases
