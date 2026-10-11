from __future__ import annotations

import numpy as np
import torch


def augment_ppg(x: torch.Tensor, sample_rate: float = 125.0) -> torch.Tensor:
    """Small shape-preserving PPG view: gain, Gaussian noise and low drift."""
    if x.ndim != 2:
        raise ValueError("PPG view must have shape [batch, time]")
    batch, length = x.shape
    scale = 0.95 + 0.10 * torch.rand((batch, 1), device=x.device, dtype=x.dtype)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = torch.randn_like(x) * (0.005 * std)
    t = torch.arange(length, device=x.device, dtype=x.dtype) / float(sample_rate)
    phase = 2 * torch.pi * torch.rand((batch, 1), device=x.device, dtype=x.dtype)
    drift_amp = (torch.rand((batch, 1), device=x.device, dtype=x.dtype) * 2 - 1) * 0.01 * std
    drift = drift_amp * torch.sin(2 * torch.pi * 0.1 * t.unsqueeze(0) + phase)
    return x * scale + noise + drift


def apply_patient_shift(x: np.ndarray, kind: str, severity: float,
                        case_id: int, starts_s: np.ndarray) -> np.ndarray:
    """Deterministic model-input stress shift after the existing per-window z-score."""
    x = np.asarray(x, dtype=np.float32)
    if kind == "clean":
        return x.copy()
    if kind not in {"gain", "baseline_drift", "gaussian_noise"}:
        raise ValueError(f"unknown shift {kind}")
    out = np.empty_like(x)
    fs = 125.0
    time = np.arange(x.shape[1], dtype=np.float32) / fs
    for i, (sample, start) in enumerate(zip(x, starts_s)):
        rng = np.random.default_rng(int(case_id) * 1000003 + int(start) * 9176 + int(severity * 10000))
        if kind == "gain":
            out[i] = sample * severity
        elif kind == "baseline_drift":
            out[i] = sample + severity * max(float(sample.std()), 1e-8) * np.sin(2 * np.pi * 0.1 * time + rng.uniform(0, 2*np.pi))
        else:
            out[i] = sample + rng.normal(0, severity * max(float(sample.std()), 1e-8), len(sample)).astype(np.float32)
    return out
