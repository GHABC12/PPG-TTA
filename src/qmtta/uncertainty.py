from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def mc_dropout_variance(adapter: nn.Module, head: nn.Module, h: torch.Tensor,
                        samples: int = 8) -> torch.Tensor:
    """Prediction variance in the source head's normalized target units."""
    if samples < 2:
        raise ValueError("MC dropout requires at least two samples")
    drops = [m for m in adapter.modules() if isinstance(m, nn.Dropout)]
    if not drops:
        raise ValueError("adapter has no dropout modules")
    states = [m.training for m in drops]
    try:
        for m in drops:
            m.train(True)
        pred = torch.stack([head(adapter(h)).reshape(-1) for _ in range(samples)])
    finally:
        for m, state in zip(drops, states):
            m.train(state)
    return pred.var(dim=0, unbiased=False)
