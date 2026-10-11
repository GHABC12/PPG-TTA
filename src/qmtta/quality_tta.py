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
                 tail_blocks: int = 1):
        self.student, self.teacher = student, teacher
        self.spo2_head, self.morph_head = spo2_head, morph_head
        self.lambda_morph, self.lambda_reg = lambda_morph, lambda_reg
        self.quality_threshold = quality_threshold
        if quality_mode not in {"gated", "ungated"}:
            raise ValueError("quality_mode must be 'gated' or 'ungated'")
        self.quality_mode = quality_mode
        self.teacher.eval(); self.spo2_head.eval(); self.morph_head.eval()
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
        if self.quality_mode == "gated":
            active = q >= self.quality_threshold
        else:
            active = torch.ones_like(q, dtype=torch.bool)
        if not bool(active.any()):
            return {"updated": 0, "loss": 0.0, "cons": 0.0, "morph": 0.0, "reg": 0.0}
        x = x[active]
        q = q[active]
        if self.quality_mode == "ungated":
            # Remove both the hard quality gate and quality weighting for a
            # clean consistency-only ablation.
            q = torch.ones_like(q)
        target_morph = target_morph[active]
        morph_valid = morph_valid[active].float()
        q = q / q.sum().clamp_min(1e-8)

        with torch.no_grad():
            y_teacher = self.spo2_head(self.teacher(x)).flatten()
        # A small additive perturbation on already z-scored PPG windows.
        x_aug = x + 0.005 * torch.randn_like(x)
        z_student = self.student(x_aug)
        y_student = self.spo2_head(z_student).flatten()
        cons_each = F.smooth_l1_loss(y_student, y_teacher, reduction="none")
        loss_cons = (q * cons_each).sum()

        morph_pred = self.morph_head(self.student(x))
        morph_each = F.smooth_l1_loss(morph_pred, target_morph, reduction="none")
        morph_weight = q[:, None] * morph_valid
        loss_morph = (morph_weight * morph_each).sum() / morph_weight.sum().clamp_min(1e-8)
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
                "reg": float(loss_reg.detach())}
