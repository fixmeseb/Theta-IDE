"""Universal Stable-Baselines3 Runner for Theta-IDE.

Executes SB3 algorithms (PPO, DQN, A2C, SAC) as a decoupled black-box engine,
logging training telemetry to metrics.csv and saving checkpoints adhering to the
Theta-IDE execution contract.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.app.core.contract_logger import ContractLogger


@hydra.main(version_base=None, config_path="../../in/config", config_name="config")
def main(cfg: DictConfig):
    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import A2C, DQN, PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print(
            "\n[Error] Stable-Baselines3 is not installed in the active environment.\n"
            "Please run: pip install stable-baselines3 shimmy\n"
            "or install the SB3 plugin via the Theta Hub.\n"
        )
        sys.exit(1)

    try:
        import gymnasium as gym
    except ImportError:
        print("\n[Error] Gymnasium is required for SB3. Run: pip install gymnasium\n")
        sys.exit(1)

    group = str(cfg.get("group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp"))
    agent_cfg = cfg.get("agent", {})
    agent_name = str(agent_cfg.get("name", "sb3_agent"))
    algo_name = str(agent_cfg.get("algorithm", "PPO")).upper()
    policy_type = str(agent_cfg.get("policy", "MlpPolicy"))
    total_timesteps = int(cfg.get("total_timesteps", agent_cfg.get("total_timesteps", 50000)))

    # Determine environment ID
    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("env_id") or env_cfg.get("name", "CartPole-v1")
    if env_id == "cartpole":
        env_id = "CartPole-v1"
    elif env_id == "mountaincar":
        env_id = "MountainCar-v0"

    print(f"\n=== [SB3 Engine] Initializing {algo_name} on {env_id} ===")
    print(f"Group: {group} | Experiment: {exp_id} | Total Timesteps: {total_timesteps}")

    # Initialize ContractLogger
    logger = ContractLogger(
        group=group,
        experiment_id=exp_id,
        method_name=agent_name,
        base_dir="results/logs",
    )

    ckpt_dir = Path("results/checkpoints") / group / exp_id / agent_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Custom SB3 Callback to bridge telemetry into Theta-IDE metrics.csv
    class ThetaMetricsCallback(BaseCallback):
        def __init__(self, contract_logger: ContractLogger, eval_freq: int = 1000):
            super().__init__()
            self.contract_logger = contract_logger
            self.eval_freq = eval_freq
            self.last_eval_step = 0

        def _on_step(self) -> bool:
            # Check episodic rewards from monitor or locals
            if self.n_calls % self.eval_freq == 0:
                infos = self.locals.get("infos", [])
                ep_rewards = []
                for info in infos:
                    if "episode" in info:
                        ep_rewards.append(info["episode"]["r"])

                metrics = {}
                if ep_rewards:
                    mean_r = float(sum(ep_rewards) / len(ep_rewards))
                    metrics["eval/reward"] = mean_r
                    metrics["train/reward"] = mean_r

                # Extract loss metrics from SB3 logger if available
                if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                    for k, v in self.model.logger.name_to_value.items():
                        if "loss" in k:
                            metrics[f"losses/{k}"] = float(v)

                self.contract_logger.log(
                    metrics,
                    step=self.n_calls,
                    transitions=self.num_timesteps,
                )
            return True

    # Build environment with Monitor wrapper for episode stats
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(gym.make(env_id))

    # Resolve algorithm class
    algo_map = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "SAC": SAC}
    AlgoClass = algo_map.get(algo_name, PPO)

    # Hyperparameters
    algo_kwargs = {}
    if "learning_rate" in agent_cfg:
        algo_kwargs["learning_rate"] = float(agent_cfg["learning_rate"])
    if "gamma" in agent_cfg:
        algo_kwargs["gamma"] = float(agent_cfg["gamma"])
    if "batch_size" in agent_cfg:
        algo_kwargs["batch_size"] = int(agent_cfg["batch_size"])

    model = AlgoClass(policy_type, env, verbose=1, **algo_kwargs)

    cb = ThetaMetricsCallback(contract_logger=logger, eval_freq=1000)

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=cb)
    finally:
        end_time = time.time()
        # Save final models
        model.save(str(ckpt_dir / "best_model.zip"))
        model.save(str(ckpt_dir / "latest_model.zip"))
        logger.save_runtime_metadata(
            config=OmegaConf.to_container(cfg, resolve=True),
            training_time_seconds=end_time - start_time,
        )
        logger.close()
        env.close()

    print(f"\n[SB3 Engine Complete] Total time: {end_time - start_time:.2f}s")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Telemetry logged to: {logger.metrics_csv_path}")


if __name__ == "__main__":
    main()
