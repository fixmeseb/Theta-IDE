import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.app.core.model_registry import register_model
from src.app.core.types import ActionResult


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@register_model("neural_blender_actor")
class NeuralBlenderActor(nn.Module):
    def __init__(self, out_size=2):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, out_size), std=0.01)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return logits

    def get_action_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


@register_model("neural_blender_mlp")
class NeuralBlenderMLP(nn.Module):
    def __init__(self, num_in_features, out_size=2, hidden_sizes=[256, 256]):
        super().__init__()
        self.num_in_features = num_in_features
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        act_fn = nn.GELU if max_width >= 128 else nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            last_size = size
        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(last_size, out_size), std=0.01)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        hidden = self.network(x)
        logits = self.actor(hidden)
        return logits


@register_model("cnn", "nature_cnn", "cnn_actor")
class CNNActor(nn.Module):
    def __init__(self, n_actions=18):
        super().__init__()
        self.n_actions = n_actions
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=self.critic(hidden),
        )

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

    def get_q_values(self, x):
        hidden = self.network(x / 255.0)
        return self.actor(hidden)


@register_model("value_network", "cnn_value_network")
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


@register_model("mlp", "dnn", "mlpqnetwork")
class MLPQNetwork(nn.Module):
    def __init__(self, n_actions, num_in_features=4, hidden_sizes=[64, 64], dueling=None, activation=None, dropout=0.0):
        super().__init__()
        self.n_actions = n_actions
        self.num_in_features = num_in_features
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        self.dueling = (max_width >= 128) if dueling is None else dueling

        if activation is None:
            act_fn = nn.GELU if max_width >= 128 else nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        self.network = nn.Sequential(*layers)

        if self.dueling:
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(last_size, max(32, last_size // 2))),
                act_fn(),
                layer_init(nn.Linear(max(32, last_size // 2), 1), std=1.0),
            )
            self.advantage_head = nn.Sequential(
                layer_init(nn.Linear(last_size, max(32, last_size // 2))),
                act_fn(),
                layer_init(nn.Linear(max(32, last_size // 2), n_actions), std=0.01),
            )
        else:
            self.head = layer_init(nn.Linear(last_size, n_actions), std=1.0)
            self.value_head = layer_init(nn.Linear(last_size, 1), std=1.0)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        if self.dueling:
            val = self.value_head(feat)
            adv = self.advantage_head(feat)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.head(feat)

    def get_q_values(self, x):
        return self.forward(x)

    def get_action_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_value(self, x, logic_state=None):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        return self.value_head(feat)

    def get_action_and_value(self, x, action=None):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        val = self.value_head(feat)
        if self.dueling:
            adv = self.advantage_head(feat)
            logits = val + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            logits = self.head(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=val,
        )

    def act(self, x, logic_state=None, epsilon=0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)


@register_model("mlp_value_network")
class MLPValueNetwork(nn.Module):
    def __init__(self, num_in_features=4, hidden_sizes=[64, 64], activation=None, dropout=0.0):
        super().__init__()
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        if activation is None:
            act_fn = nn.GELU if max_width >= 128 else nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        layers.append(layer_init(nn.Linear(last_size, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        return self.network(x)


@register_model("q_network", "cnn_q_network")
class QNetwork(nn.Module):
    def __init__(self, n_actions=18):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# Standard architecture aliases (decoupled from BlendRL legacy naming)
NatureCNN = NeuralBlenderActor
StandardMLP = NeuralBlenderMLP
