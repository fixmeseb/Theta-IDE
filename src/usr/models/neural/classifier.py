"""Feed-forward classifiers for supervised learning on tabular data.

The registry's `dnn` is MLPQNetwork, an RL Q-network built around n_actions and
a duelling head, so it is the wrong shape for a labelled dataset even though the
name suggests otherwise. These are plain classifiers: features in, logits out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.app.core.model_registry import register_model


@register_model("mlp_classifier", "tabular_classifier", "classifier")
class MLPClassifier(nn.Module):
    """Multi-layer perceptron over a flat feature vector.

    `n_classes=1` emits a single logit per sample, which is what the shipped
    ClassificationEvalProtocol expects for a binary task; a larger value emits
    one logit per class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        n_classes: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(str(activation).lower(), nn.ReLU)

        sizes = list(hidden_sizes) or [128, 64]
        layers: list[nn.Module] = []
        last = input_dim
        for size in sizes:
            layers += [nn.Linear(last, size), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, n_classes)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        """Logits, shaped (B,) for a binary task and (B, n_classes) otherwise."""
        logits = self.head(self.body(x.reshape(x.size(0), -1)))
        return logits.squeeze(-1) if self.n_classes == 1 else logits
