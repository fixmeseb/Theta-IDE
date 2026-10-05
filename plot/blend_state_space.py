#!/usr/bin/env python3
"""
plot/blend_state_space.py — BlendRL 2D State Space Decision Boundary Plotter.

Visualizes decision boundaries in 2D via Competency Simplex, Global PCA, and t-SNE Manifold.
"""

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from plot.base import BasePlotter
from plot.blend_common import discover_blendrl_checkpoints, extract_model_routing_data
from src.app.pipeline import runtime


class BlendStateSpacePlotter(BasePlotter):
    def __init__(self):
        super().__init__("blend_state_space")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        discovered = discover_blendrl_checkpoints(exp_id, group, clean_exp)
        if not discovered:
            print(f"Notice [blend_state_space]: No modular BlendRL checkpoints found for '{clean_exp}'")
            return

        data = extract_model_routing_data(discovered, sample_size=4000)
        state_space_data = data["state_space_data"]

        if not state_space_data:
            print("Notice [blend_state_space]: No state space data to plot.")
            return

        self._plot_state_space_boundaries(state_space_data, output_dir, clean_exp)

    def _plot_state_space_boundaries(self, state_space_data: dict, output_dir: Path, clean_exp: str):
        target_name = "problem" if "problem" in state_space_data else list(state_space_data.keys())[0]
        data = state_space_data[target_name]
        sub_states = data["states"]
        sub_w = data["w_logic"]

        # 1. Pedagogical Competency Simplex (GMM Posteriors)
        gmm_path = Path(f"{runtime.DATASETS_DIR}/pyrenees/per_problem/{target_name}/gmm_scaler.npz")
        if not gmm_path.exists():
            gmm_path = Path(f"{runtime.DATASETS_DIR}/pyrenees/pyrenees_gmm_scaler.npz")

        have_gmm = False
        if gmm_path.exists():
            try:
                gdata = np.load(gmm_path, allow_pickle=True)
                means = gdata["means"]
                precisions = gdata["precisions"]
                log_dets = gdata["log_dets"]
                log_weights = gdata["log_weights"]
                feat_idx = gdata["feature_indices"]

                x_feat = sub_states[:, feat_idx]
                d = x_feat.shape[-1]
                const = 0.5 * d * np.log(2.0 * np.pi)
                log_probs = []
                for k in range(3):
                    diff = x_feat - means[k]
                    maha = np.sum(diff * (diff @ precisions[k]), axis=-1)
                    log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
                    log_probs.append(log_p)
                posteriors = np.exp(np.stack(log_probs, -1) - np.max(np.stack(log_probs, -1), axis=-1, keepdims=True))
                posteriors /= posteriors.sum(axis=-1, keepdims=True)

                x_mastery = posteriors[:, 2] - posteriors[:, 0]
                y_med = posteriors[:, 1]
                have_gmm = True
            except Exception:
                have_gmm = False

        # 2. PCA
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(sub_states)

        # 3. t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=35, n_iter=1000)
        x_tsne = tsne.fit_transform(sub_states)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        cmap = LinearSegmentedColormap.from_list("NeuralLogic", ["#2b5c8f", "#9970ab", "#d95f02"])

        # Panel 1: Competency Simplex
        if have_gmm:
            sc1 = axes[0].scatter(
                x_mastery, y_med, c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none"
            )
            axes[0].set_title("A. Pedagogical Competency Continuum", fontsize=11, fontweight="bold")
            axes[0].set_xlabel(
                "Competency Continuum [P(High) − P(Low)]\n(← Struggling  |  Mastery →)", fontsize=10, fontweight="bold"
            )
            axes[0].set_ylabel("Intermediate Uncertainty [P(Med)]", fontsize=10, fontweight="bold")
        else:
            x_feat_raw = sub_states[:, 84] if sub_states.shape[1] > 84 else sub_states[:, 0]
            y_feat_raw = sub_states[:, 72] if sub_states.shape[1] > 72 else sub_states[:, 1]
            sc1 = axes[0].scatter(
                x_feat_raw, y_feat_raw, c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none"
            )
            axes[0].set_title("A. Feature Decision Space", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Steps Since Last Error", fontsize=10, fontweight="bold")
            axes[0].set_ylabel("Cumulative Accuracy", fontsize=10, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.3)

        # Panel 2: PCA
        sc2 = axes[1].scatter(
            x_pca[:, 0], x_pca[:, 1], c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none"
        )
        axes[1].set_title(
            f"B. Global State PCA (Expl. Var: {pca.explained_variance_ratio_.sum() * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel(
            f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=10, fontweight="bold"
        )
        axes[1].set_ylabel(
            f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=10, fontweight="bold"
        )
        axes[1].grid(True, linestyle="--", alpha=0.3)

        # Panel 3: t-SNE
        sc3 = axes[2].scatter(
            x_tsne[:, 0], x_tsne[:, 1], c=sub_w, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.65, s=18, edgecolors="none"
        )
        axes[2].set_title("C. Non-Linear Manifold (t-SNE)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
        axes[2].set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.3)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc1, cax=cbar_ax)
        cbar.set_label("Logic Authority Weight ($w_{\\mathrm{logic}}$)", fontsize=10, fontweight="bold")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0.0 (Pure Neural)", "0.5 (Blended)", "1.0 (Pure Logic)"])

        fig.suptitle(
            f"BlendRL Neuro-Symbolic State Space Gating Boundaries ({clean_exp} - {target_name})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig.subplots_adjust(right=0.90, wspace=0.25)

        plot_path = output_dir / "blend_routing_state_space.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot: {plot_path}")
