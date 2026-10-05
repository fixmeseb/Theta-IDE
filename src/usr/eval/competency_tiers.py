"""Three-tier competency model shared by the Pyrenees preprocessing scripts.

Steps are scored, split into Low / Med / High tiers at two score percentiles, and one Gaussian is fitted
per tier. scripts/preprocess_pyrenees_per_problem.py, scripts/run_mrmr_and_recluster.py and
scripts/fit_gmm_competency.py each score steps differently but share the sampling and fitting here.
"""
import numpy as np


def sample_rows(matrix, columns, sample_size=100000, seed=42):
    """A reproducible random subset of rows (at most `sample_size`), restricted to `columns`."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(matrix), size=min(sample_size, len(matrix)), replace=False)
    return matrix[idx][:, columns]


def fit_tiered_gaussians(X, scores, p_low=35, p_high=92):
    """Label rows of X as tier 0/1/2 at the p_low/p_high percentiles of `scores` and fit a Gaussian per tier.

    Returns means, covariances (regularised by 1e-4 * I), precisions, log-determinants, log-weights,
    weights (the share of rows in each tier) and the tier labels. A tier with no rows gets a zero mean,
    identity covariance and weight 1/3.
    """
    p_l = np.percentile(scores, p_low)
    p_h = np.percentile(scores, p_high)
    labels = np.zeros(len(scores), dtype=int)
    labels[scores > p_l] = 1
    labels[scores > p_h] = 2

    n_feats = X.shape[1]
    means = np.zeros((3, n_feats))
    covariances = np.zeros((3, n_feats, n_feats))
    weights = np.zeros(3)
    for k in range(3):
        X_k = X[labels == k]
        if len(X_k) == 0:
            means[k] = np.zeros(n_feats)
            covariances[k] = np.eye(n_feats)
            weights[k] = 1.0 / 3.0
        else:
            means[k] = X_k.mean(axis=0)
            covariances[k] = np.cov(X_k, rowvar=False) + 1e-4 * np.eye(n_feats)
            weights[k] = len(X_k) / len(X)

    return {
        "means": means,
        "covariances": covariances,
        "precisions": np.array([np.linalg.inv(c) for c in covariances]),
        "log_dets": np.array([np.linalg.slogdet(c)[1] for c in covariances]),
        "log_weights": np.log(weights + 1e-12),
        "weights": weights,
        "labels": labels,
    }
