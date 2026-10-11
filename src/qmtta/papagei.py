from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class PaPaGeiEncoder(nn.Module):
    """Adapter for the already-installed official PaPaGei implementation.

    We expose the 512-dimensional pooled backbone latent (the last tuple item),
    not the pretraining projection head. The backbone source is not copied here.
    """

    def __init__(self, repo: str | Path, checkpoint: str | Path, device: str = "cpu"):
        super().__init__()
        repo = Path(repo).resolve()
        sys.path.insert(0, str(repo))
        from models.resnet import ResNet1DMoE

        self.network = ResNet1DMoE(
            in_channels=1, base_filters=32, kernel_size=3, stride=2,
            groups=1, n_block=18, n_classes=512, n_experts=3,
        )
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        result = self.network.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {result}")
        self.network.to(device)
        self.embedding_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.network(x)
        return outputs[-1]

    def freeze_all(self) -> None:
        for p in self.network.parameters():
            p.requires_grad_(False)

    def train_last_blocks(self, n_blocks: int = 1) -> list[nn.Parameter]:
        if n_blocks < 1 or n_blocks > len(self.network.basicblock_list):
            raise ValueError("n_blocks must be between 1 and the encoder block count")
        self.freeze_all()
        for block in self.network.basicblock_list[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]


class StandardizedEncoder(nn.Module):
    """Apply the training-fold feature statistics used by the source heads."""

    def __init__(self, encoder: nn.Module, mean, std):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32).reshape(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return (z - self.feature_mean) / self.feature_std.clamp_min(1e-6)

    def train_last_blocks(self, n_blocks: int = 1):
        return self.encoder.train_last_blocks(n_blocks)
