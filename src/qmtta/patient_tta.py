from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .adapter import ResidualBottleneckAdapter
from .patient_calibration import patient_prototype_loss
from .uncertainty import mc_dropout_variance


class PatientAdapterTTA:
    """Case-episodic unlabeled adapter updates; labels are absent from this API."""

    MODES = {
        "consistency_only": (1.0, 0.0, False),
        "patient_only": (0.0, 0.1, False),
        "uncertainty_weighted": (1.0, 0.0, True),
        "full_patta": (1.0, 0.1, True),
    }

    def __init__(self, head: nn.Module, *, feature_dim: int = 512, bottleneck_dim: int = 32,
                 dropout: float = 0.1, lr: float = 1e-4, steps: int = 5,
                 mode: str = "full_patta", mc_samples: int = 8,
                 uncertainty_scale: float = 1.0, identity_weight: float = 0.01):
        if mode not in self.MODES:
            raise ValueError(f"unknown PATTA mode: {mode}")
        if lr <= 0 or steps < 1 or uncertainty_scale <= 0 or identity_weight < 0:
            raise ValueError("invalid PATTA hyperparameters")
        self.head = head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.adapter = ResidualBottleneckAdapter(feature_dim, bottleneck_dim, dropout)
        self.adapter.to(next(head.parameters()).device)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=0.0)
        self.steps = steps
        self.mc_samples = mc_samples
        self.uncertainty_scale = uncertainty_scale
        self.identity_weight = identity_weight
        self.lambda_consistency, self.lambda_patient, self.use_uncertainty = self.MODES[mode]

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        self.adapter.eval()
        with torch.no_grad():
            return self.head(self.adapter(h)).reshape(-1)

    def update(self, h: torch.Tensor, h_view: torch.Tensor) -> dict:
        """Update only adapter on unlabeled base/augmented feature pairs."""
        h, h_view = h.detach(), h_view.detach()
        if h.ndim != 2 or h.shape != h_view.shape or h.shape[0] < 2:
            return {"steps": 0, "loss": 0.0, "consistency": 0.0,
                    "patient": 0.0, "identity": 0.0, "mc_variance": 0.0,
                    "mean_weight": 1.0, "adapter_delta_rms": 0.0}
        self.adapter.eval()
        with torch.no_grad():
            if self.use_uncertainty:
                variance = mc_dropout_variance(self.adapter, self.head, h, self.mc_samples)
                weight = torch.exp(-variance / self.uncertainty_scale)
            else:
                variance = h.new_zeros(h.shape[0]); weight = h.new_ones(h.shape[0])
            weight = weight.detach()
        last = {}
        for _ in range(self.steps):
            self.adapter.eval()
            z = self.adapter(h)
            z_view = self.adapter(h_view)
            p = self.head(z).reshape(-1)
            p_view = self.head(z_view).reshape(-1)
            cons_each = F.smooth_l1_loss(p, p_view, reduction="none")
            loss_cons = (weight * cons_each).sum() / weight.sum().clamp_min(1e-8)
            loss_patient = patient_prototype_loss(z)
            loss_identity = (z - h).square().mean()
            loss = (self.lambda_consistency * loss_cons +
                    self.lambda_patient * loss_patient +
                    self.identity_weight * loss_identity)
            if not torch.isfinite(loss):
                self.adapter = ResidualBottleneckAdapter(
                    h.shape[1], self.adapter.down.out_features,
                    self.adapter.dropout.p).to(h.device)
                self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.optimizer.param_groups[0]["lr"])
                return {"steps": 0, "loss": float("nan"), "consistency": float("nan"),
                        "patient": float("nan"), "identity": float("nan"),
                        "mc_variance": float(variance.mean()), "mean_weight": float(weight.mean()),
                        "adapter_delta_rms": float("nan")}
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.optimizer.step()
            last = {"steps": 1, "loss": float(loss.detach()),
                    "consistency": float(loss_cons.detach()),
                    "patient": float(loss_patient.detach()),
                    "identity": float(loss_identity.detach()),
                    "mc_variance": float(variance.mean()),
                    "mean_weight": float(weight.mean()),
                    "adapter_delta_rms": float((self.adapter(h) - h).square().mean().sqrt().detach())}
        last["steps"] = self.steps
        return last
