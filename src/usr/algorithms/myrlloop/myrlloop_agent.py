"""Custom MyrlloopAgent reinforcement learning algorithm."""
from typing import Any, Dict, List
import lightning as L
import torch
import torch.optim as optim

from src.app.core.model_registry import build_model
from src.app.core.protocols import AuxiliaryLossProtocol, OptimizerGroupsProtocol
from src.usr.algorithms.agent_registry import register_agent, register_algorithm


@register_algorithm("myrlloop")
class MyrlloopAgent(L.LightningModule):
    """Pluggable RL agent extending PyTorch Lightning."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        """Declare which learning paradigms this algorithm supports."""
        return ['online_rl', 'offline_rl']

    def __init__(self, cfg: Any, env: Any = None, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.cfg = cfg
        self.env = env
        self.lr = float(getattr(cfg, "lr", 0.0003))

        # Model is decoupled and instantiated dynamically via build_model
        model_spec = getattr(cfg, "model", "dnn")
        self.model = build_model(model_spec, env=env)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        obs, action, reward, next_obs, done = batch[:5]
        q_pred = self.model(obs)
        loss = torch.nn.functional.mse_loss(q_pred, reward)

        # Collect optional auxiliary losses from the model
        if isinstance(self.model, AuxiliaryLossProtocol):
            aux_losses = self.model.compute_auxiliary_losses(batch, q_pred)
            for k, v in aux_losses.items():
                loss = loss + v
                self.log(f"train/{k}", v)

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        # Query custom optimizer parameter groups if model provides them
        if isinstance(self.model, OptimizerGroupsProtocol):
            groups = self.model.configure_optimizer_groups(base_lr=self.lr)
            return optim.Adam(groups)
        return optim.Adam(self.parameters(), lr=self.lr)
