#!/usr/bin/env python3
"""
in/datasets/prepare_mayo_cchs.py

Processes raw Mayo and CCHS CSV trajectories into:
1. Unified .npz trajectory archives for VectorizedEnv (`mayo.npz`, `cchs.npz`).
2. Chunked .pkl transition datasets for offline RL training (`in/datasets/mayo/cql/`, `in/datasets/cchs/cql/`).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.app.dataset_utils import DatasetWriter

FEATURE_NAMES = [
    "HeartRate",
    "RespiratoryRate",
    "PulseOx",
    "FIO2",
    "SystolicBP",
    "MAP",
    "Temperature",
    "Bands",
    "BUN",
    "Lactate",
    "Platelet",
    "Creatinine",
    "BiliRubin",
    "WBC",
]

ACTION_COSTS = {
    0: 0.0,    # withhold / no intervention
    1: 0.01,   # oxygen control
    2: 0.1,    # antibiotic / anti-infection
    3: 0.2,    # vasopressor
}


def process_cchs(csv_path: Path, out_dir: Path):
    print(f"=== Processing CCHS dataset from {csv_path} ===")
    df = pd.read_csv(csv_path)

    # Sort by visit and time
    time_col = "MinutesFromArrival" if "MinutesFromArrival" in df.columns else "HoursFromArrival"
    df = df.sort_values(by=["VisitIdentifier", time_col]).reset_index(drop=True)

    grouped = df.groupby("VisitIdentifier", sort=False)
    n_visits = len(grouped)
    print(f"Found {n_visits:,} patient trajectories, {len(df):,} total transitions.")

    # Storage for NPZ
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    traj_ptrs = [0]
    shock_flags = []
    visit_ids = []

    # DatasetWriter for offline RL PKL chunks
    cql_dir = out_dir / "cql"
    writer = DatasetWriter(save_dir=cql_dir, chunk_size=100000, env_name="cchs")

    total_transitions = 0
    curr_ptr = 0

    for vid, group in grouped:
        T = len(group)
        if T == 0:
            continue

        states = group[FEATURE_NAMES].values.astype(np.float32)
        actions = group["Action"].values.astype(np.int64)

        # Determine shock outcome for this patient
        has_shock = bool(group["Shock"].max() == 1)
        base_outcome_reward = (1.0 / T) if not has_shock else (-1.0 / T)

        visit_rewards = np.zeros(T, dtype=np.float32)
        visit_dones = np.zeros(T, dtype=bool)
        visit_dones[-1] = True

        for t in range(T):
            act = int(actions[t])
            cost = ACTION_COSTS.get(act, 0.0)
            r = float(base_outcome_reward - (cost / T))
            visit_rewards[t] = r

            obs = states[t]
            logic_obs = np.stack([obs, obs], axis=0)
            if t == T - 1:
                next_obs = obs.copy()
                next_logic_obs = logic_obs.copy()
                done = True
            else:
                next_obs = states[t + 1]
                next_logic_obs = np.stack([next_obs, next_obs], axis=0)
                done = False

            writer.add(
                obs=obs,
                logic_obs=logic_obs,
                action=act,
                reward=r,
                next_obs=next_obs,
                next_logic_obs=next_logic_obs,
                done=done,
            )
            total_transitions += 1

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(visit_rewards)
        all_dones.append(visit_dones)

        curr_ptr += T
        traj_ptrs.append(curr_ptr)
        shock_flags.append(1 if has_shock else 0)
        visit_ids.append(str(vid))

    writer.close()

    # Save NPZ archive
    npz_path = out_dir / "cchs.npz"
    np.savez_compressed(
        npz_path,
        states=np.concatenate(all_states, axis=0),
        actions=np.concatenate(all_actions, axis=0),
        rewards=np.concatenate(all_rewards, axis=0),
        dones=np.concatenate(all_dones, axis=0),
        traj_ptrs=np.array(traj_ptrs, dtype=np.int64),
        shock=np.array(shock_flags, dtype=np.int32),
        vids=np.array(visit_ids),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved CCHS NPZ archive: {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Saved CCHS PKL chunks: {cql_dir} ({total_transitions:,} transitions)\n")


def process_mayo(csv_path: Path, out_dir: Path):
    print(f"=== Processing Mayo dataset from {csv_path} ===")
    df = pd.read_csv(csv_path)

    # Sort by visit and time
    time_col = "MinutesFromArrival" if "MinutesFromArrival" in df.columns else "HoursFromArrival"
    df = df.sort_values(by=["VisitIdentifier", time_col]).reset_index(drop=True)

    grouped = df.groupby("VisitIdentifier", sort=False)
    n_visits = len(grouped)
    print(f"Found {n_visits:,} patient trajectories, {len(df):,} total transitions.")

    # Storage for NPZ
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    traj_ptrs = [0]
    shock_flags = []
    death_flags = []
    visit_ids = []

    # DatasetWriter for offline RL PKL chunks
    cql_dir = out_dir / "cql"
    writer = DatasetWriter(save_dir=cql_dir, chunk_size=100000, env_name="mayo")

    total_transitions = 0
    curr_ptr = 0

    has_reward_col = "reward" in df.columns

    for vid, group in grouped:
        T = len(group)
        if T == 0:
            continue

        states = group[FEATURE_NAMES].values.astype(np.float32)
        actions = group["Action"].values.astype(np.int64)

        has_shock = bool(group["Shock"].max() == 1) if "Shock" in group else False
        has_death = bool(group["Death"].max() == 1) if "Death" in group else False

        if has_reward_col:
            # Mayo has clinical TQN / stage severity reward computed directly in CSV
            visit_rewards = group["reward"].values.astype(np.float32)
        else:
            base_outcome_reward = (1.0 / T) if not has_shock else (-1.0 / T)
            visit_rewards = np.zeros(T, dtype=np.float32)
            for t in range(T):
                act = int(actions[t])
                cost = ACTION_COSTS.get(act, 0.0)
                visit_rewards[t] = float(base_outcome_reward - (cost / T))

        visit_dones = np.zeros(T, dtype=bool)
        visit_dones[-1] = True

        for t in range(T):
            act = int(actions[t])
            r = float(visit_rewards[t])

            obs = states[t]
            logic_obs = np.stack([obs, obs], axis=0)
            if t == T - 1:
                next_obs = obs.copy()
                next_logic_obs = logic_obs.copy()
                done = True
            else:
                next_obs = states[t + 1]
                next_logic_obs = np.stack([next_obs, next_obs], axis=0)
                done = False

            writer.add(
                obs=obs,
                logic_obs=logic_obs,
                action=act,
                reward=r,
                next_obs=next_obs,
                next_logic_obs=next_logic_obs,
                done=done,
            )
            total_transitions += 1

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(visit_rewards)
        all_dones.append(visit_dones)

        curr_ptr += T
        traj_ptrs.append(curr_ptr)
        shock_flags.append(1 if has_shock else 0)
        death_flags.append(1 if has_death else 0)
        visit_ids.append(str(vid))

    writer.close()

    # Save NPZ archive
    npz_path = out_dir / "mayo.npz"
    np.savez_compressed(
        npz_path,
        states=np.concatenate(all_states, axis=0),
        actions=np.concatenate(all_actions, axis=0),
        rewards=np.concatenate(all_rewards, axis=0),
        dones=np.concatenate(all_dones, axis=0),
        traj_ptrs=np.array(traj_ptrs, dtype=np.int64),
        shock=np.array(shock_flags, dtype=np.int32),
        death=np.array(death_flags, dtype=np.int32),
        vids=np.array(visit_ids),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved Mayo NPZ archive: {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Saved Mayo PKL chunks: {cql_dir} ({total_transitions:,} transitions)\n")


def main():
    cchs_csv = PROJECT_ROOT / "in/datasets/cchs/CCHS_lstm_filtered_dataframe_md.csv"
    cchs_out = PROJECT_ROOT / "in/datasets/cchs"

    mayo_csv = PROJECT_ROOT / "in/datasets/mayo/Mayo_lstm_filtered_dataframe_md.csv"
    mayo_out = PROJECT_ROOT / "in/datasets/mayo"

    if cchs_csv.exists():
        process_cchs(cchs_csv, cchs_out)
    else:
        print(f"Warning: CCHS CSV not found at {cchs_csv}")

    if mayo_csv.exists():
        process_mayo(mayo_csv, mayo_out)
    else:
        print(f"Warning: Mayo CSV not found at {mayo_csv}")

    print("All dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
