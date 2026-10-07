#!/usr/bin/env python3
"""
plot/blend_common.py — Shared utilities, checkpoint discovery, and data extraction for BlendRL plotters.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from plot.base import clean_label, get_canonical_method_name
from src.usr.eval.pyrenees_evaluator import PyreneesEvaluator

KNOWN_PYRENEES_PROBLEMS = [
    "problem",
    "ex132(w)",
    "ex132a(w)",
    "ex152a(w)",
    "ex212(w)",
    "ex242(w)",
    "ex252(w)",
    "ex252a(w)",
    "exc137(w)",
    "exp426d(w)",
    "exp426e(w)",
]


def discover_blendrl_checkpoints(exp_id: str, group: str, clean_exp: str) -> dict:
    """Discovers only modular BlendRL checkpoints that output blending weights."""
    ckpt_root = Path("results/checkpoints") / group / clean_exp
    if not ckpt_root.exists():
        ckpt_root = Path("results/checkpoints") / clean_exp
    if not ckpt_root.exists():
        return {}

    discovered = {}
    for entry in sorted(ckpt_root.rglob("best_model*.ckpt")):
        rel_parts = entry.relative_to(ckpt_root).parts
        parent_dir_name = rel_parts[0]

        if "blendrl" not in parent_dir_name.lower():
            continue

        detected_dataset = None
        detected_method = parent_dir_name

        if parent_dir_name in KNOWN_PYRENEES_PROBLEMS and len(rel_parts) > 2:
            detected_dataset = parent_dir_name
            detected_method = rel_parts[1]
        else:
            for prob in KNOWN_PYRENEES_PROBLEMS:
                prob_clean = prob.replace("(", "_").replace(")", "_").rstrip("_")
                if parent_dir_name.endswith(f"_{prob}"):
                    detected_dataset = prob
                    detected_method = parent_dir_name[: -len(f"_{prob}")]
                    break
                elif parent_dir_name.endswith(f"_{prob_clean}"):
                    detected_dataset = prob
                    detected_method = parent_dir_name[: -len(f"_{prob_clean}")].rstrip("_")
                    break

        canon_method = get_canonical_method_name(detected_method)
        key = f"{canon_method}_{detected_dataset}" if detected_dataset else canon_method
        if key not in discovered or entry.name == "best_model.ckpt":
            discovered[key] = {
                "path": entry,
                "method": canon_method,
                "dataset": detected_dataset or "problem",
                "dir_name": parent_dir_name,
            }
    return discovered


def load_modular_agent(path: Path):
    from src.usr.methods.cql_agent import CQLAgent
    from src.usr.methods.iql_agent import IQLAgent

    classes = [CQLAgent, IQLAgent]
    try:
        from src.usr.methods.cew_agent import CEWAgent

        classes.insert(1, CEWAgent)
    except ImportError:
        pass

    for cls in classes:
        try:
            ag = cls.load_from_checkpoint(str(path), map_location="cpu", weights_only=False)
            ag.eval()
            if hasattr(ag, "is_modular") and ag.is_modular:
                return ag
        except Exception:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location="cpu", weights_only=False, strict=False)
                ag.eval()
                if hasattr(ag, "is_modular") and ag.is_modular:
                    return ag
            except Exception:
                continue
    return None


def extract_model_routing_data(discovered: dict, sample_size: int = 4000) -> dict:
    """Extracts weights, tiers, states, and OOD scores for all discovered BlendRL checkpoints."""
    evaluator = PyreneesEvaluator()
    authority_rows = []
    tier_rows = []
    ood_handover_rows = []
    state_space_data = {}

    for key, meta in sorted(discovered.items()):
        prob_name = meta["dataset"]
        agent = load_modular_agent(meta["path"])
        if agent is None:
            continue

        clean_path = Path(f"in/datasets/pyrenees/per_problem/{prob_name}/clean.npz")
        if not clean_path.exists() and prob_name == "problem":
            clean_path = Path("in/datasets/pyrenees/pyrenees_clean.npz")
        if not clean_path.exists():
            continue

        data = np.load(clean_path, allow_pickle=True)
        states = np.vstack(data["states"]).astype(np.float32)
        n_states = len(states)
        if n_states == 0:
            continue

        obs_t = torch.tensor(states, dtype=torch.float32)
        pad = torch.zeros((len(obs_t), 3), dtype=torch.float32)
        obs_aug = torch.cat([obs_t, pad], dim=-1) if obs_t.shape[-1] == 123 else obs_t
        logic_obs = obs_aug.unsqueeze(1).repeat(1, 2, 1)

        with torch.no_grad():
            if hasattr(agent, "get_blending_weights"):
                weights = agent.get_blending_weights(obs_aug, logic_obs)
            elif hasattr(agent.model, "actor") and hasattr(agent.model.actor, "to_blender_policy_distribution"):
                weights = agent.model.actor.to_blender_policy_distribution(obs_aug, logic_obs)
            else:
                _, weights = agent.model.actor(obs_aug, logic_obs)

        if weights is None:
            continue

        module_types = getattr(
            agent.model,
            "module_types",
            getattr(getattr(agent.model, "actor", None), "module_types", ["logic", "neural"]),
        )
        logic_idx = module_types.index("logic") if "logic" in module_types else 0
        neural_idx = module_types.index("neural") if "neural" in module_types else (1 if len(module_types) > 1 else 0)

        w_logic = (
            weights[:, logic_idx].detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights[:, logic_idx]
        )
        w_neural = (
            weights[:, neural_idx].detach().cpu().numpy()
            if isinstance(weights, torch.Tensor)
            else weights[:, neural_idx]
        )

        gmm_path = Path(f"in/datasets/pyrenees/per_problem/{prob_name}/gmm_scaler.npz")
        if not gmm_path.exists():
            gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")
        tiers = evaluator._compute_gmm_tiers(states, gmm_path)

        # 1. Authority
        pure_log_pct = float(np.mean(w_logic >= 0.90) * 100.0)
        pure_neu_pct = float(np.mean(w_neural >= 0.90) * 100.0)
        mixed_pct = float(np.mean((w_neural > 0.10) & (w_neural < 0.90)) * 100.0)

        authority_rows.append(
            {
                "Model": prob_name,
                "Method": clean_label(meta["method"]),
                "N_States": n_states,
                "Mean Logic Weight": float(np.mean(w_logic)),
                "Mean Neural Weight": float(np.mean(w_neural)),
                "Pure Logic % (>=0.90)": pure_log_pct,
                "Pure Neural % (>=0.90)": pure_neu_pct,
                "Mixed % (0.10-0.90)": mixed_pct,
            }
        )

        # 2. Tiers
        for t_label, t_val in [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]:
            m = tiers == t_val
            n_t = int(m.sum())
            if n_t > 0:
                tier_rows.append(
                    {
                        "Model": prob_name,
                        "Method": clean_label(meta["method"]),
                        "Tier": t_label,
                        "N_States": n_t,
                        "Mean Logic Weight": float(np.mean(w_logic[m])),
                        "Mean Neural Weight": float(np.mean(w_neural[m])),
                        "Pure Logic % (>=0.90)": float(np.mean(w_logic[m] >= 0.90) * 100.0),
                        "Pure Neural % (>=0.90)": float(np.mean(w_neural[m] >= 0.90) * 100.0),
                    }
                )

        # 3. OOD Distance
        sub_n = min(n_states, sample_size)
        sub_idx = np.random.default_rng(42).choice(n_states, size=sub_n, replace=False)
        sub_states = states[sub_idx]
        sub_w_logic = w_logic[sub_idx]
        sub_w_neural = w_neural[sub_idx]

        nbrs = NearestNeighbors(n_neighbors=20, metric="euclidean").fit(sub_states)
        distances, _ = nbrs.kneighbors(sub_states)
        ood_score = distances[:, -1]

        n_bins = 5
        bin_labels = [f"Q{i + 1}" for i in range(n_bins)]
        try:
            ood_bins = pd.qcut(ood_score, q=n_bins, labels=bin_labels)
            for b_name in bin_labels:
                m_b = ood_bins == b_name
                if m_b.sum() > 0:
                    ood_handover_rows.append(
                        {
                            "Model": prob_name,
                            "Method": clean_label(meta["method"]),
                            "OOD_Quantile": b_name,
                            "Mean_OOD_Distance": float(ood_score[m_b].mean()),
                            "Mean_Logic_Weight": float(sub_w_logic[m_b].mean()),
                            "Mean_Neural_Weight": float(sub_w_neural[m_b].mean()),
                            "Pure_Logic_Pct": float((sub_w_logic[m_b] >= 0.90).mean() * 100.0),
                            "Pure_Neural_Pct": float((sub_w_neural[m_b] >= 0.90).mean() * 100.0),
                        }
                    )
        except Exception:
            pass

        if prob_name in ["problem", "ex152a(w)", "ex252a(w)", "exp426e(w)"] or len(state_space_data) == 0:
            state_space_data[prob_name] = {
                "states": sub_states,
                "w_logic": sub_w_logic,
                "w_neural": sub_w_neural,
                "tiers": tiers[sub_idx],
                "ood_score": ood_score,
            }

    return {
        "authority_rows": authority_rows,
        "tier_rows": tier_rows,
        "ood_handover_rows": ood_handover_rows,
        "state_space_data": state_space_data,
    }
