from __future__ import annotations

import torch


def patient_prototype_loss(z: torch.Tensor) -> torch.Tensor:
    """Within-case prototype compactness loss over unlabeled features."""
    if z.ndim != 2 or z.shape[0] < 2:
        return z.new_zeros(())
    prototype = z.mean(dim=0, keepdim=True)
    return (z - prototype).square().mean()
