#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "prediction_std": float(np.std(pred))}


def main() -> None:
    p = argparse.ArgumentParser(description="Train a case-isolated PaPaGei linear-probe SpO2 baseline")
    p.add_argument("--index", default="results/vitaldb_pilot/window_index.csv")
    p.add_argument("--embeddings", default="results/vitaldb_pilot/embeddings.npz")
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--backbone", choices=["papagei", "sigma"], default="papagei")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/vitaldb_pilot/source")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    df = pd.read_csv(a.index)
    with np.load(a.embeddings) as f:
        z = f["embeddings"].astype(np.float32)
        if "case_id" in f and not np.array_equal(f["case_id"], df.case_id.to_numpy()):
            raise SystemExit("embedding case_id order does not match window index")
        if "start_s" in f and not np.array_equal(f["start_s"], df.start_s.to_numpy()):
            raise SystemExit("embedding start_s order does not match window index")
    if len(z) != len(df):
        raise SystemExit("embedding row count does not match window index")
    folds = sorted(df.fold.unique().astype(int).tolist())
    if len(folds) < 3 or a.test_fold not in folds:
        raise SystemExit("need at least three case folds; run audit_vitaldb.py first")
    val_fold = folds[(folds.index(a.test_fold) + 1) % len(folds)]
    train_mask = ~df.fold.isin([a.test_fold, val_fold]).to_numpy()
    val_mask = (df.fold.to_numpy() == val_fold)
    test_mask = (df.fold.to_numpy() == a.test_fold)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise SystemExit("train/validation/test split is empty")

    x_mean = z[train_mask].mean(0); x_std = np.maximum(z[train_mask].std(0), 1e-6)
    z = (z - x_mean) / x_std
    y = df.spo2_median.to_numpy(dtype=np.float32)
    y_mean = float(y[train_mask].mean()); y_std = max(float(y[train_mask].std()), 1e-6)
    yt = (y - y_mean) / y_std
    morph_cols = ["sVRI", "IPA", "SQI"]
    morph = df[morph_cols].to_numpy(dtype=np.float32)
    valid_m = np.ones_like(morph, dtype=np.float32)
    if "IPA_valid" in df:
        valid_m[:, 1] = df.IPA_valid.to_numpy(dtype=np.float32)
    m_mean = np.zeros(3, dtype=np.float32); m_std = np.ones(3, dtype=np.float32)
    for j in range(3):
        use = train_mask & (valid_m[:, j] > 0)
        if use.any():
            m_mean[j] = morph[use, j].mean()
            m_std[j] = max(float(morph[use, j].std()), 1e-6)
    mt = (morph - m_mean) / m_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.ModuleDict({"spo2": nn.Linear(z.shape[1], 1), "morph": nn.Linear(z.shape[1], 3)}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_set = TensorDataset(torch.from_numpy(z[train_mask]), torch.from_numpy(yt[train_mask]),
                              torch.from_numpy(mt[train_mask]), torch.from_numpy(valid_m[train_mask]))
    loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True)
    best_state, best_val, best_epoch = None, float("inf"), -1
    for epoch in range(a.epochs):
        model.train()
        for bx, by, bm, bv in loader:
            bx, by, bm, bv = bx.to(device), by.to(device), bm.to(device), bv.to(device)
            pred = model["spo2"](bx).squeeze(1)
            pred_m = model["morph"](bx)
            loss_y = nn.functional.huber_loss(pred, by)
            loss_m = (nn.functional.smooth_l1_loss(pred_m, bm, reduction="none") * bv).sum() / bv.sum().clamp_min(1)
            loss = loss_y + 0.1 * loss_m
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(z[val_mask]).to(device)
            val_pred = (model["spo2"](vx).squeeze(1).cpu().numpy() * y_std) + y_mean
        val_errors = np.abs(val_pred - y[val_mask])
        val_cases = df.loc[val_mask, "case_id"].to_numpy()
        val_mae = float(np.mean([val_errors[val_cases == case].mean()
                                 for case in np.unique(val_cases)]))
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(z[test_mask]).to(device)
        pred = model["spo2"](tx).squeeze(1).cpu().numpy() * y_std + y_mean
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": best_state, "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std, "morph_mean": m_mean,
            "morph_std": m_std, "test_fold": a.test_fold, "val_fold": val_fold,
            "best_epoch": best_epoch, "seed": a.seed,
            "embedding_source": f"{a.backbone} frozen pretrained latent"}
    torch.save(ckpt, out_dir / f"source_fold{a.test_fold}.pt")
    test = df.loc[test_mask, ["case_id", "start_s", "spo2_median"]].copy()
    test["prediction"] = pred
    test.to_csv(out_dir / f"predictions_fold{a.test_fold}.csv", index=False)
    baseline_mean = np.full(int(test_mask.sum()), y[train_mask].mean(), dtype=np.float32)
    baseline_median = np.full(int(test_mask.sum()), np.median(y[train_mask]), dtype=np.float32)
    report = {"test": metrics(y[test_mask], pred),
              "train_mean_baseline": metrics(y[test_mask], baseline_mean),
              "train_median_baseline": metrics(y[test_mask], baseline_median),
              "test_fold": a.test_fold, "validation_fold": val_fold,
              "train_case_ids": sorted(map(int, df.loc[train_mask, "case_id"].unique())),
              "validation_case_ids": sorted(map(int, df.loc[val_mask, "case_id"].unique())),
              "test_case_ids": sorted(map(int, df.loc[test_mask, "case_id"].unique())),
              "best_epoch": best_epoch, "selection_metric": "mean case MAE",
              "best_validation_mean_case_mae": best_val}
    (out_dir / f"metrics_fold{a.test_fold}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
