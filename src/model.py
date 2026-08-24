"""
model.py

Fixed MLP architecture used for every optimizer / seed combination:

    30 -> Linear -> 32 -> ReLU -> Linear -> 16 -> ReLU -> Linear -> 1 (logit)

Architecture is held constant across all experiments; only the optimizer
and the random seed vary.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int = 30, h1: int = 32, h2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # returns raw logits (no sigmoid)


def set_seed(seed: int) -> None:
    """Seed torch (CPU + CUDA) for reproducible initialization / data order."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(seed: int, in_features: int = 30) -> MLP:
    """Construct a freshly-initialized model for a given seed."""
    set_seed(seed)
    return MLP(in_features=in_features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model(seed=0)
    print(m)
    print("Total parameters:", count_parameters(m))