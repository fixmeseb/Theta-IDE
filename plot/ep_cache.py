import os
import pickle
from pathlib import Path
from src.app.pipeline import runtime


def get_ep_eval_data(exp_id, cfg, group, output_dir):
    cache_path = Path(output_dir) / "ep_eval_cache.pkl"
    remake = cfg.get("remake", False)

    ckpt_dir = Path(runtime.CHECKPOINTS_DIR) / group / exp_id
    if not ckpt_dir.exists():
        matches = list(Path(runtime.CHECKPOINTS_DIR).glob(f"**/{exp_id}"))
        if matches:
            ckpt_dir = matches[0]

    cache_stale = False
    if cache_path.exists() and ckpt_dir.exists():
        cache_mtime = cache_path.stat().st_mtime
        for ckpt_file in ckpt_dir.rglob("*.ckpt"):
            if ckpt_file.stat().st_mtime > cache_mtime:
                cache_stale = True
                break

    if cache_path.exists() and not remake and not cache_stale:
        print(f"Loading cached EP evaluation data from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from src.usr.eval.early_prediction.eval_logic import compute_ep_eval_data

    if not ckpt_dir.exists():
        print(f"Error: Could not find checkpoint directory for {exp_id}")
        return None

    print("Computing EP evaluation data (this may take a while)...")
    data = compute_ep_eval_data(
        checkpoint_root=str(ckpt_dir),
        dataset_path=cfg.get("dataset_path", None),
        ep_ckpt_root=cfg.get("ep_ckpt_root", f"{runtime.CHECKPOINTS_DIR}/early_prediction"),
        n_splits=cfg.get("n_splits", 20),
        use_volatility=cfg.get("use_volatility", True),
    )

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data
