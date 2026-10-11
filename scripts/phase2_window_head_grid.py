#!/usr/bin/env python3
"""Case-grouped 3x3x3 frozen PaPaGei head/window/loss grid for VitalDB."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def case_mae(y: np.ndarray, p: np.ndarray, cases: np.ndarray) -> float:
    return float(pd.DataFrame({"case": cases, "ae": np.abs(y - p)}).groupby("case").ae.mean().mean())


class LinearHead(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.net = nn.Linear(d, 1)
    def forward(self, x): return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class ResidualMLPHead(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.norm = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 256); self.fc2 = nn.Linear(256, d); self.out = nn.Linear(d, 1); self.drop = nn.Dropout(.1)
    def forward(self, x):
        z = self.norm(x); z = self.drop(torch.nn.functional.gelu(self.fc1(z))); z = self.norm(x + self.fc2(z)); return self.out(z).squeeze(-1)


def make_loss(name):
    if name == "mse": return nn.MSELoss(reduction="none")
    if name == "mae": return nn.L1Loss(reduction="none")
    return nn.SmoothL1Loss(beta=1.0, reduction="none")


def fit_fold(x, index, head, loss_name, test_fold, epochs, patience, device):
    folds = sorted(index.fold.unique().astype(int).tolist())
    val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
    cases = index.case_id.to_numpy()
    y = index.spo2_median.to_numpy(dtype=np.float32)
    tr = (~index.fold.isin([test_fold, val_fold])).to_numpy()
    va = index.fold.eq(val_fold).to_numpy()
    te = index.fold.eq(test_fold).to_numpy()
    scaler = StandardScaler().fit(x[tr])
    xt = scaler.transform(x[tr]).astype("float32"); xv = scaler.transform(x[va]).astype("float32"); xe = scaler.transform(x[te]).astype("float32")
    # Equal total training weight per case, matching case-equal model selection.
    tr_cases = cases[tr]; counts = pd.Series(tr_cases).value_counts().to_dict()
    w = np.asarray([1.0 / counts[c] for c in tr_cases], dtype=np.float32)
    w /= w.mean()
    y_mean = float(y[tr].mean()); y_scale = float(y[tr].std()) or 1.0
    y_scaled = (y - y_mean) / y_scale
    tx = torch.from_numpy(xt).to(device); ty = torch.from_numpy(y_scaled[tr]).to(device); tw = torch.from_numpy(w).to(device)
    vx = torch.from_numpy(xv).to(device)
    seed = 20261011 + 100 * test_fold + (0 if head == "linear" else 1 if head == "mlp" else 2) * 10 + (0 if loss_name == "mse" else 1 if loss_name == "mae" else 2)
    random.seed(int(seed)); np.random.seed(seed); torch.manual_seed(seed)
    model = {"linear": LinearHead, "mlp": MLPHead, "residual_mlp": ResidualMLPHead}[head](x.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = make_loss(loss_name)
    best, best_epoch, state, stale = float("inf"), 0, None, 0
    # Full-batch updates keep the 135-fit comparison fast and deterministic.
    for ep in range(1, epochs + 1):
        model.train(); optim.zero_grad(set_to_none=True)
        pred = model(tx); objective = (criterion(pred, ty) * tw).mean(); objective.backward(); optim.step()
        model.eval()
        with torch.no_grad(): vp = model(vx).cpu().numpy() * y_scale + y_mean
        score = case_mae(y[va], vp, cases[va])
        if score < best - 1e-5:
            best, best_epoch, state, stale = score, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(xe).to(device)).cpu().numpy() * y_scale + y_mean
    return {"head": head, "loss": loss_name, "test_fold": test_fold, "validation_fold": val_fold,
            "best_epoch": best_epoch, "validation_case_mae": best, "n_train_windows": int(tr.sum()),
            "n_test_windows": int(te.sum()), "window_mae": float(mean_absolute_error(y[te], pred)),
            "window_rmse": float(mean_squared_error(y[te], pred) ** .5), "window_r2": float(r2_score(y[te], pred)),
            "case_mean_mae": case_mae(y[te], pred, cases[te]), "median_baseline_case_mae": case_mae(y[te], np.full(te.sum(), np.median(y[tr])), cases[te])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="phase2/window_matrix root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args(); root = Path(args.root); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for win in (10, 30, 60):
        sub = root / f"window_{win}s"; idx = pd.read_csv(sub / "window_index.csv"); pack = np.load(sub / "embeddings_papagei.npz"); x = pack["embeddings"]
        if len(idx) != len(x) or not np.array_equal(idx.case_id.to_numpy(), pack["case_id"]) or not np.array_equal(idx.start_s.to_numpy(), pack["start_s"]):
            raise RuntimeError(f"index/cache mismatch at {win}s")
        for head in ("linear", "mlp", "residual_mlp"):
            for loss in ("mse", "mae", "huber"):
                print(f"window={win}s head={head} loss={loss}", flush=True)
                rows = [dict(window_s=win, **fit_fold(x, idx, head, loss, f, args.epochs, args.patience, args.device)) for f in sorted(idx.fold.unique().astype(int))]
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(out / "head_grid_fold_metrics.csv", index=False)
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["window_s", "head", "loss"], as_index=False).agg(
        case_mean_mae=("case_mean_mae", "mean"), case_mae_sd=("case_mean_mae", "std"),
        window_mae=("window_mae", "mean"), window_rmse=("window_rmse", "mean"), window_r2=("window_r2", "mean"),
        median_baseline_case_mae=("median_baseline_case_mae", "mean"), best_epoch_mean=("best_epoch", "mean"))
    summary.to_csv(out / "head_grid_summary.csv", index=False)
    best = summary.sort_values("case_mean_mae").iloc[0].to_dict()
    result = {"dataset": "existing VitalDB only", "representation": "frozen PaPaGei; 512 dimensions; same fixed case folds across windows",
              "window_counts": {str(w): int(len(pd.read_csv(root / f"window_{w}s/window_index.csv"))) for w in (10,30,60)},
              "selection": "for each outer test fold, next fold selects checkpoint epoch by case-equal MAE; case-equal training weights; outer metric is case-equal MAE",
              "head_grid_cells": 27, "device": args.device, "epochs_max": args.epochs, "early_stop_patience": args.patience,
              "best_exploratory_cell": best, "all_cells_are_exploratory_after_model_selection": True}
    (out / "head_grid_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
