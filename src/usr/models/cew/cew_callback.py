"""CEW self-organization callback.

Registered automatically when model: cew is selected. Fires the
self-organization pipeline at interval boundaries and sets the
_request_optimizer_rebind flag so OfflineAgentBase can react.
"""

from __future__ import annotations

from typing import Any, Optional

import lightning as L

from src.usr.models.cew.cew_model import CEWModel


class CEWSelfOrganizationCallback(L.Callback):
    """Runs CEW self-organization for all CEWModel instances in the agent's model.

    Supports both standalone usage (agent holds a CEWModel directly as
    `q_model`) and BlendRL composite usage (agent holds a BlenderActorCritic
    whose policy_modules list contains one or more CEWModel instances).
    """

    def __init__(self, model: Any | None = None):
        super().__init__()
        self.model = model

    def _collect_cew_models(self, pl_module: L.LightningModule) -> list[CEWModel]:
        """Find all CEWModel instances attached to the agent via protocols."""
        if self.model is not None and isinstance(self.model, CEWModel):
            return [self.model]
        from src.app.core.protocols import walk_model_modules

        return [m for m in walk_model_modules(pl_module) if isinstance(m, CEWModel)]

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epochs_per_interval = pl_module.get_cfg("epochs_per_interval", 1) if hasattr(pl_module, "get_cfg") else 1
        if pl_module.current_epoch % epochs_per_interval != 0:
            return

        cew_models = self._collect_cew_models(pl_module)
        if not cew_models:
            return

        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "reader") or datamodule.reader is None:
            return

        sample_size = min(len(datamodule.reader), 10_000)
        if sample_size == 0:
            return

        obs = datamodule.reader.sample(sample_size)["obs"]

        for cew_model in cew_models:
            changed = cew_model.self_organize(obs)
            if changed:
                # Flag is already set on cew_model by self_organize().
                # OfflineAgentBase.on_train_epoch_start picks it up after super() returns.
                pass
