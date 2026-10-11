from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EpisodicQualityMorphTTA:
    """Quality-gated, episodic feature-extractor adaptation.

    `y_true` is deliberately not an argument: target SpO2 labels cannot enter
    the update path. Call reset() before each case to prevent case carry-over.
    """

    def __init__(self, student: nn.Module, teacher: nn.Module,
                 spo2_head: nn.Module, morph_head: nn.Module,
                 *, lr: float = 1e-5, lambda_morph: float = 0.1,
                 lambda_reg: float = 1e-4, quality_threshold: float = 0.5,
                 quality_mode: str = "gated",
                 uncertainty_mode: str = "off", mc_samples: int = 8,
                 uncertainty_threshold: float = float("inf"),
                 uncertainty_scale: float = 1.0,
                 tail_blocks: int = 1):
        self.student, self.teacher = student, teacher
        self.spo2_head, self.morph_head = spo2_head, morph_head
        self.lambda_morph, self.lambda_reg = lambda_morph, lambda_reg
        self.quality_threshold = quality_threshold
        if uncertainty_mode not in {"off", "gated", "weighted"}:
            raise ValueError("uncertainty_mode must be off, gated, or weighted")
        if mc_samples < 2 or uncertainty_scale <= 0:
            raise ValueError("MC samples must be >= 2 and uncertainty scale must be positive")
        self.uncertainty_mode = uncertainty_mode
        self.mc_samples = mc_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_scale = uncertainty_scale
        if quality_mode not in {"gated", "ungated"}:
            raise ValueError("quality_mode must be 'gated' or 'ungated'")
        self.quality_mode = quality_mode
        self.teacher.eval(); self.spo2_head.eval(); self.morph_head.eval()
        self.dropout_modules = [m for m in self.teacher.modules()
                                if isinstance(m, (nn.Dropout, nn.Dropout1d,
                                                  nn.Dropout2d, nn.Dropout3d))]
        if uncertainty_mode != "off" and not self.dropout_modules:
            raise ValueError("MC-dropout uncertainty requested but teacher has no dropout modules")
        for model in (self.teacher, self.spo2_head, self.morph_head):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.parameters = self.student.train_last_blocks(tail_blocks)
        self.student.eval()  # keep dropout and BatchNorm state fixed during TTA
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=0.0)
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.student.load_state_dict(self.teacher.state_dict())
        self.theta0 = [p.detach().clone() for p in self.parameters]
        self.optimizer.state.clear()

    def step(self, x: torch.Tensor, quality: torch.Tensor,
             target_morph: torch.Tensor, morph_valid: torch.Tensor) -> dict:
        q = quality.detach().clamp(0, 1)
        uncertainty_active = torch.ones_like(q, dtype=torch.bool)
        uncertainty_weight = torch.ones_like(q)
        with torch.no_grad():
            if self.uncertainty_mode == "off":
                y_teacher = self.spo2_head(self.teacher(x)).flatten()
                uncertainty = torch.zeros_like(y_teacher)
            else:
                states = [module.training for module in self.dropout_modules]
                try:
                    for module in self.dropout_modules:
                        module.train(True)
                    samples = torch.stack([
                        self.spo2_head(self.teacher(x)).flatten()
                        for _ in range(self.mc_samples)
                    ], dim=0)
                finally:
                    for module, was_training in zip(self.dropout_modules, states):
                        module.train(was_training)
                y_teacher = samples.mean(dim=0)
                uncertainty = samples.var(dim=0, unbiased=False)
                if self.uncertainty_mode == "gated":
                    uncertainty_active = uncertainty <= self.uncertainty_threshold
                elif self.uncertainty_mode == "weighted":
                    uncertainty_weight = torch.exp(-uncertainty / self.uncertainty_scale)
        if self.quality_mode == "gated":
            quality_active = q >= self.quality_threshold
        else:
            quality_active = torch.ones_like(q, dtype=torch.bool)
        active = quality_active & uncertainty_active
        diagnostics = {"batch_size": int(len(q)),
                       "quality_eligible": int(quality_active.sum()),
                       "uncertainty_eligible": int(uncertainty_active.sum()),
                       "selected_for_update": int(active.sum())}
        if not bool(active.any()):
            return {"updated": 0, "loss": 0.0, "cons": 0.0, "morph": 0.0,
                    "reg": 0.0, "uncertainty_mean": float(uncertainty.mean()),
                    **diagnostics}
        x = x[active]
        q = q[active]
        if self.quality_mode == "ungated":
            # Remove both the hard quality gate and quality weighting for a
            # clean consistency-only ablation.
            q = torch.ones_like(q)
        q = q * uncertainty_weight[active]
        if self.lambda_morph > 0:
            target_morph = target_morph[active]
            morph_valid = morph_valid[active].float()
        q = q / q.sum().clamp_min(1e-8)

        y_teacher = y_teacher[active]
        # A small additive perturbation on already z-scored PPG windows.
        x_aug = x + 0.005 * torch.randn_like(x)
        z_student = self.student(x_aug)
        y_student = self.spo2_head(z_student).flatten()
        cons_each = F.smooth_l1_loss(y_student, y_teacher, reduction="none")
        loss_cons = (q * cons_each).sum()

        if self.lambda_morph > 0:
            morph_pred = self.morph_head(self.student(x))
            morph_each = F.smooth_l1_loss(morph_pred, target_morph, reduction="none")
            morph_weight = q[:, None] * morph_valid
            loss_morph = (morph_weight * morph_each).sum() / morph_weight.sum().clamp_min(1e-8)
        else:
            loss_morph = y_student.new_zeros(())
        loss_reg = torch.stack([(p - p0).square().mean()
                                for p, p0 in zip(self.parameters, self.theta0)]).mean()
        loss = loss_cons + self.lambda_morph * loss_morph + self.lambda_reg * loss_reg
        if not torch.isfinite(loss):
            self.reset()
            return {"updated": 0, "loss": float("nan"), "cons": float("nan"),
                    "morph": float("nan"), "reg": float("nan")}
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        return {"updated": int(len(x)), "loss": float(loss.detach()),
                "cons": float(loss_cons.detach()), "morph": float(loss_morph.detach()),
                "reg": float(loss_reg.detach()),
                "uncertainty_mean": float(uncertainty.mean()), **diagnostics}
