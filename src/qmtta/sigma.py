from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class SigmaPPGEncoder(nn.Module):
    """Feature adapter around the pinned official SIGMA-PPG implementation."""

    def __init__(self, repo: str | Path, checkpoint: str | Path,
                 input_size: int = 1000, patch_size: int = 100,
                 device: str = "cpu"):
        super().__init__()
        repo = Path(repo).resolve()
        sys.path.insert(0, str(repo))
        from downstream.model_select import select_model

        self.patch_size = patch_size
        self.input_size = input_size
        self.network, self.use_patches = select_model(
            backbone="sigma_ppg_pro", num_classes=1, in_chans=1,
            pretrained=True, checkpoint_path=str(checkpoint),
            freeze_backbone_flag=False, device=torch.device(device),
            patch_size=patch_size, input_size=input_size)
        # The upstream selector resizes pos_embed to patch_count, but the
        # transformer prepends a CLS token and therefore requires patch_count+1.
        # Restore the original pretrained CLS vector and interpolate patch
        # positions separately so the patch projection remains weight-compatible.
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", "").replace("student.", ""): v for k, v in state.items()}
        raw_pos = state.get("pos_embed")
        if raw_pos is not None:
            n_patches = input_size // patch_size
            patch_pos = F.interpolate(raw_pos[:, 1:, :].transpose(1, 2),
                                      size=n_patches, mode="linear", align_corners=False).transpose(1, 2)
            pos = torch.cat([raw_pos[:, :1, :], patch_pos], dim=1)
            self.network.pos_embed = nn.Parameter(pos)
        self.network.to(device)
        self.embedding_dim = 360

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != self.input_size:
            raise ValueError(f"expected (B,1,{self.input_size}) Sigma input, got {tuple(x.shape)}")
        patches = rearrange(x, "b c (n t) -> b c n t", t=self.patch_size)
        return self.network.forward_features(patches)

    def train_last_blocks(self, n_blocks: int = 1):
        if n_blocks < 1 or n_blocks > len(self.network.blocks):
            raise ValueError("n_blocks must be between 1 and the Sigma block count")
        for p in self.network.parameters():
            p.requires_grad_(False)
        for block in self.network.blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        return [p for p in self.network.parameters() if p.requires_grad]
