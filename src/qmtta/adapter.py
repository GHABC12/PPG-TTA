from __future__ import annotations

import torch
from torch import nn


class ResidualBottleneckAdapter(nn.Module):
    """Near-identity residual bottleneck adapter for frozen PPG features."""

    def __init__(self, feature_dim: int = 512, bottleneck_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        if feature_dim < 1 or bottleneck_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid adapter dimensions or dropout")
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        # Initially f(h) = h, so the source model is reproduced before TTA.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.dropout(self.activation(self.down(h))))
