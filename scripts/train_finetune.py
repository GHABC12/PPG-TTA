#!/usr/bin/env python3
"""Source-supervised tail fine-tuning on case-disjoint VitalDB folds.

Only the final backbone block and supervised heads receive gradients. The
outer test fold is excluded from fitting and early stopping and is never
scored by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmtta.papagei import PaPaGeiEncoder
from qmtta.sigma import SigmaPPGEncoder


def load_signals(index: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Load cached windows in index order, reading only selected folds."""
    selected = index.loc[mask]
    if selected.empty:
        raise ValueError("selected fold has no windows")
    output = [None] * len(selected)
    positions = {row_index: i for i, row_index in enumerate(selected.index)}
    for cache_path, rows in selected.groupby("cache_path", sort=False).groups.items():
        ids = selected.loc[rows, "window_idx"].to_numpy(dtype=np.int64)
        with np.load(cache_path) as cache:
            signals = cache["ppg"][ids].astype(np.float32, copy=False)
        for j, row_index in enumerate(rows):
            output[positions[row_index]] = signals[j]
    return np.stack(output)


def batch_features(model: nn.Module, x: np.ndarray, batch_size: int,
                   device: torch.device) -> np.ndarray:
    result = []
    model.eval()  # keep pretrained batch-normalization/dropout behavior fixed
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).unsqueeze(1).to(device)
            result.append(model(xb).float().cpu().numpy())
    return np.concatenate(result, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune one source encoder tail block on VitalDB training cases")
    p.add_argument("--index", default="results/vitaldb_pilot/window_index.csv")
    p.add_argument("--backbone", choices=["papagei", "sigma"], required=True)
    p.add_argument("--test-fold", type=int, required=True,
                   help="outer held-out fold; its labels are never used here")
    p.add_argument("--val-fold", type=int, required=True,
                   help="explicit early-stopping fold, distinct from test-fold")
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--sigma-repo", default=os.environ.get("SIGMA_REPO", "third_party/SigmaPPG"))
    p.add_argument("--sigma-checkpoint", default=os.environ.get("SIGMA_CHECKPOINT", "weights/sigma.pth"))
    p.add_argument("--sigma-patch-size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/vitaldb_pilot/finetune")
    a = p.parse_args()
    if a.epochs < 1 or a.patience < 1 or a.batch_size < 1:
        raise SystemExit("epochs, patience, and batch-size must be positive")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    df = pd.read_csv(a.index)
    folds = sorted(df.fold.unique().astype(int).tolist())
    if len(folds) < 3 or a.test_fold not in folds or a.val_fold not in folds:
        raise SystemExit("need at least three case folds and valid --test-fold/--val-fold")
    if a.test_fold == a.val_fold:
        raise SystemExit("--test-fold and --val-fold must be different")
    train_mask = ~df.fold.isin([a.test_fold, a.val_fold]).to_numpy()
    val_mask = (df.fold.to_numpy(dtype=int) == a.val_fold)
    test_mask = (df.fold.to_numpy(dtype=int) == a.test_fold)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise SystemExit("train/validation/test split is empty")
    # Detect any malformed case-level split before touching signal data.
    case_sets = [set(map(int, df.loc[m, "case_id"].unique()))
                 for m in (train_mask, val_mask, test_mask)]
    if any(case_sets[i] & case_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit("case leakage detected across train/validation/test folds")

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    x_train = load_signals(df, train_mask)
    x_val = load_signals(df, val_mask)
    device = torch.device(a.device)
    if a.backbone == "papagei":
        encoder = PaPaGeiEncoder(a.repo, a.checkpoint, a.device)
        protocol = {"name": "papagei_10s_125hz", "sampling_rate": 125, "window_seconds": 10}
    else:
        if "fs" not in df or "start_s" not in df or "end_s" not in df:
            raise SystemExit("index must include fs/start_s/end_s for SIGMA")
        if df.protocol.nunique() != 1 or df.protocol.iloc[0] != "sigma":
            raise SystemExit("SIGMA requires an index built with --protocol sigma")
        input_size = int((df.end_s.iloc[0] - df.start_s.iloc[0]) * df.fs.iloc[0])
        encoder = SigmaPPGEncoder(a.sigma_repo, a.sigma_checkpoint, input_size=input_size,
                                  patch_size=a.sigma_patch_size, device=a.device)
        protocol = {"name": "sigma_20s_50hz", "sampling_rate": float(df.fs.iloc[0]),
                    "window_seconds": float(df.end_s.iloc[0] - df.start_s.iloc[0]),
                    "patch_size": a.sigma_patch_size}
    encoder.to(device).eval()

    # Freeze-fold feature normalization is estimated from the training cases
    # before fine-tuning, just like train_source.py; labels from val/test do not
    # enter these statistics.
    z_train = batch_features(encoder, x_train, a.batch_size, device)
    x_mean = z_train.mean(axis=0).astype(np.float32)
    x_std = np.maximum(z_train.std(axis=0), 1e-6).astype(np.float32)
    y_train = train_df.spo2_median.to_numpy(dtype=np.float32)
    y_val = val_df.spo2_median.to_numpy(dtype=np.float32)
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-6)
    morph_cols = ["sVRI", "IPA", "SQI"]
    morph_train = train_df[morph_cols].to_numpy(dtype=np.float32)
    morph_val = val_df[morph_cols].to_numpy(dtype=np.float32)
    valid_train = np.ones_like(morph_train, dtype=np.float32)
    valid_val = np.ones_like(morph_val, dtype=np.float32)
    if "IPA_valid" in df:
        valid_train[:, 1] = train_df.IPA_valid.to_numpy(dtype=np.float32)
        valid_val[:, 1] = val_df.IPA_valid.to_numpy(dtype=np.float32)
    morph_mean = np.zeros(3, dtype=np.float32)
    morph_std = np.ones(3, dtype=np.float32)
    for j in range(3):
        available = valid_train[:, j] > 0
        if available.any():
            morph_mean[j] = morph_train[available, j].mean()
            morph_std[j] = max(float(morph_train[available, j].std()), 1e-6)
    mt_train = ((morph_train - morph_mean) / morph_std).astype(np.float32)
    mt_val = ((morph_val - morph_mean) / morph_std).astype(np.float32)
    yt_train = ((y_train - y_mean) / y_std).astype(np.float32)
    yt_val = ((y_val - y_mean) / y_std).astype(np.float32)

    tail_parameters = encoder.train_last_blocks(1)
    trainable_state_keys = sorted(name for name, parameter in encoder.named_parameters()
                                  if parameter.requires_grad)
    heads = nn.ModuleDict({"spo2": nn.Linear(z_train.shape[1], 1),
                           "morph": nn.Linear(z_train.shape[1], 3)}).to(device)
    optimizer = torch.optim.AdamW([
        {"params": tail_parameters, "lr": a.encoder_lr},
        {"params": heads.parameters(), "lr": a.head_lr},
    ], weight_decay=a.weight_decay)
    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(yt_train),
                              torch.from_numpy(mt_train), torch.from_numpy(valid_train))
    loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True)
    best_val, best_epoch, best_encoder, best_heads, stale = float("inf"), -1, None, None, 0
    for epoch in range(a.epochs):
        encoder.eval()  # gradients flow through the tail; BN/dropout state remains fixed
        heads.train()
        for bx, by, bm, bv in loader:
            bx, by, bm, bv = bx.unsqueeze(1).to(device), by.to(device), bm.to(device), bv.to(device)
            z = (encoder(bx) - torch.as_tensor(x_mean, device=device)) / torch.as_tensor(x_std, device=device)
            pred = heads["spo2"](z).squeeze(1)
            pred_m = heads["morph"](z)
            loss_y = nn.functional.huber_loss(pred, by)
            loss_m = (nn.functional.smooth_l1_loss(pred_m, bm, reduction="none") * bv).sum() / bv.sum().clamp_min(1)
            loss = loss_y + 0.1 * loss_m
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tail_parameters, 1.0)
            optimizer.step()
        encoder.eval(); heads.eval()
        val_pred_norm = []
        with torch.no_grad():
            for start in range(0, len(x_val), a.batch_size):
                xb = torch.from_numpy(x_val[start:start + a.batch_size]).unsqueeze(1).to(device)
                z = (encoder(xb) - torch.as_tensor(x_mean, device=device)) / torch.as_tensor(x_std, device=device)
                val_pred_norm.extend(heads["spo2"](z).flatten().cpu().numpy())
        val_pred = np.asarray(val_pred_norm, dtype=np.float32) * y_std + y_mean
        val_error = np.abs(val_pred - y_val)
        val_case_ids = val_df.case_id.to_numpy()
        case_maes = {int(case): float(val_error[val_case_ids == case].mean())
                     for case in np.unique(val_case_ids)}
        val_mae = float(np.mean(list(case_maes.values())))
        print(json.dumps({"epoch": epoch + 1, "validation_mean_case_mae": val_mae,
                          "validation_case_mae": case_maes}))
        if val_mae < best_val:
            best_val, best_epoch, stale = val_mae, epoch + 1, 0
            full_state = encoder.state_dict()
            best_encoder = {k: full_state[k].detach().cpu().clone()
                            for k in trainable_state_keys}
            best_heads = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
        else:
            stale += 1
            if stale >= a.patience:
                break

    if best_encoder is None or best_heads is None:
        raise RuntimeError("no finite validation checkpoint was produced")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": best_heads,
        "encoder_state": best_encoder,
        "encoder_trainable_keys": trainable_state_keys,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "morph_mean": morph_mean, "morph_std": morph_std,
        "backbone": a.backbone,
        "test_fold": a.test_fold, "val_fold": a.val_fold,
        "train_folds": [f for f in folds if f not in (a.test_fold, a.val_fold)],
        "best_epoch": best_epoch, "best_validation_mae": best_val,
        "best_validation_mean_case_mae": best_val,
        "selection_metric": "mean case MAE",
        "seed": a.seed, "tail_blocks": 1, "protocol": protocol,
        "train_case_ids": sorted(case_sets[0]),
        "validation_case_ids": sorted(case_sets[1]),
        "test_case_ids": sorted(case_sets[2]),
        "labels_used_for_training": "train folds only",
        "labels_used_for_early_stopping": "validation fold only",
        "test_labels_used_for_training_or_early_stopping": False,
    }
    path = out_dir / f"finetune_{a.backbone}_fold{a.test_fold}.pt"
    torch.save(checkpoint, path)
    summary = {k: checkpoint[k] for k in (
        "backbone", "test_fold", "val_fold", "train_folds", "best_epoch",
        "best_validation_mae", "seed", "tail_blocks", "protocol",
        "train_case_ids", "validation_case_ids", "test_case_ids")}
    (out_dir / f"finetune_{a.backbone}_fold{a.test_fold}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(path), **summary}, indent=2))


if __name__ == "__main__":
    main()
