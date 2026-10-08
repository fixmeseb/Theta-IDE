#!/usr/bin/env python3
"""
in/datasets/prepare_tabular.py

Materializes real datasets for the unsupervised paradigm, which needs an offline
dataset but has no access to MIMIC. Everything here ships inside scikit-learn,
so nothing is downloaded and the result is reproducible on any machine.

Writes to in/datasets/<name>/<name>.npz, each holding:
    features  float32  (n_samples, n_features)        flat form
    sequences float32  (n_samples, steps, per_step)    sequence form, when the
                                                       feature count factorises
    labels    int64    (n_samples,)                    held back from training;
                                                       for scoring a latent space

Usage:
    python in/datasets/prepare_tabular.py            # every dataset
    python in/datasets/prepare_tabular.py digits     # just one
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = PROJECT_ROOT / "in" / "datasets"

# name -> (sklearn loader, sequence shape or None)
DATASETS = {
    "breast_cancer": ("load_breast_cancer", None),
    "digits": ("load_digits", (8, 8)),
}


def build(name: str) -> Path:
    import sklearn.datasets as skd

    loader_name, seq_shape = DATASETS[name]
    bunch = getattr(skd, loader_name)()
    features = np.asarray(bunch.data, dtype=np.float32)
    labels = np.asarray(bunch.target, dtype=np.int64)

    # Standardise once, here: an autoencoder on raw clinical ranges is dominated
    # by whichever feature happens to have the largest units.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features = (features - mean) / std

    arrays = {"features": features, "labels": labels}
    if seq_shape is not None:
        steps, per_step = seq_shape
        if steps * per_step == features.shape[1]:
            arrays["sequences"] = features.reshape(-1, steps, per_step)

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(out_path, **arrays)

    shapes = ", ".join(f"{k}{v.shape}" for k, v in arrays.items())
    try:
        shown = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_path  # An output directory outside the project is still fine.
    print(f"wrote {shown}  ({shapes})")
    return out_path


def main(argv: list[str]) -> int:
    requested = argv[1:] or sorted(DATASETS)
    unknown = [n for n in requested if n not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(sorted(DATASETS))}")
        return 1
    for name in requested:
        build(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
