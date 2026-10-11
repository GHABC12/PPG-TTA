#!/usr/bin/env python3
"""Case-episodic, label-free adapter TTA on held-out VitalDB cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmtta.augment import apply_patient_shift, augment_ppg
from qmtta.papagei import PaPaGeiEncoder
from qmtta.patient_tta import PatientAdapterTTA


METHODS = ("consistency_only", "patient_only", "uncertainty_weighted", "full_patta")
SHIFTS = (("clean", 0.0), ("gain_0.8", 0.8), ("gain_1.2", 1.2),
          ("baseline_drift_0.1", 0.1), ("gaussian_noise_0.05", 0.05))


def make_head(checkpoint: dict, device: torch.device) -> nn.Module:
    d = len(checkpoint["x_mean"])
    spo2 = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            if checkpoint.get("strategy") == "ln_head" else nn.Linear(d, 1))
    heads = nn.ModuleDict({"spo2": spo2, "morph": nn.Linear(d, 3)})
    heads.load_state_dict(checkpoint["model"])
    heads.to(device).eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return heads["spo2"]


def load_encoder(checkpoint: dict, args, device: torch.device) -> PaPaGeiEncoder:
    encoder = PaPaGeiEncoder(args.repo, args.checkpoint, str(device)).to(device).eval()
    state = checkpoint.get("encoder_state", {})
    if state:
        unexpected = encoder.load_state_dict(state, strict=False).unexpected_keys
        if unexpected:
            raise RuntimeError(f"unexpected encoder keys: {unexpected[:4]}")
    encoder.freeze_all()
    return encoder


@torch.inference_mode()
def encode(encoder: nn.Module, x: np.ndarray, *, batch_size: int,
           device: torch.device, augment: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    zs, views = [], []
    for i in range(0, len(x), batch_size):
        bx = torch.from_numpy(x[i:i + batch_size]).to(device)
        z = encoder(bx.unsqueeze(1)).float().cpu().numpy()
        zs.append(z)
        if augment:
            xv = augment_ppg(bx)
            views.append(encoder(xv.unsqueeze(1)).float().cpu().numpy())
    return np.concatenate(zs), (np.concatenate(views) if augment else None)


def case_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    return {"n_windows_scored": int(len(y)), "case_mae": float(np.mean(np.abs(error))),
            "case_rmse": float(np.sqrt(np.mean(error ** 2))), "case_bias": float(np.mean(error)),
            "prediction_std": float(np.std(pred)), "target_std": float(np.std(y))}


def bootstrap_improvement(case_df: pd.DataFrame, method: str, shift: str,
                          reps: int, seed: int) -> dict:
    part = case_df[case_df["shift"].eq(shift)]
    wide = part.pivot(index="case_id", columns="method", values="case_mae").dropna(subset=["frozen", method])
    diff = (wide["frozen"] - wide[method]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.mean(diff[rng.integers(0, len(diff), (reps, len(diff)))], axis=1)
    return {"shift": shift, "method": method, "n_cases": len(diff),
            "source_minus_method_case_mae": float(diff.mean()),
            "paired_bootstrap_ci95_low": float(np.quantile(draws, .025)),
            "paired_bootstrap_ci95_high": float(np.quantile(draws, .975))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="results/vitaldb_pilot/window_index.csv")
    p.add_argument("--source-dir", default="results/vitaldb_pilot/source")
    p.add_argument("--ln-head-dir", default="results/vitaldb_pilot/baselines/ln_head")
    p.add_argument("--repo", default="/home/user/shiyunxiao/PPG/PapaGei")
    p.add_argument("--checkpoint", default="/home/user/shiyunxiao/PPG/PapaGei/weights/papagei_s.pt")
    p.add_argument("--out-dir", default="results/phase3_patient_aware_tta")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-windows", type=int, default=32)
    p.add_argument("--encoder-batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--uncertainty-scale", type=float, default=1.0)
    p.add_argument("--identity-weight", type=float, default=0.01)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    a = p.parse_args()
    if a.batch_size < 2 or a.warmup_windows < 2 or a.steps < 1:
        raise SystemExit("batch-size, warmup-windows, and steps must be >= 2, >= 2, and >= 1")
    device = torch.device(a.device)
    index = pd.read_csv(a.index)
    folds = sorted(index.fold.unique().astype(int))
    outer_folds = [int(v) for v in a.folds.split(",") if v.strip()]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    case_rows, fold_rows, update_rows = [], [], []
    source_dir, ln_dir = Path(a.source_dir), Path(a.ln_head_dir)

    for outer in outer_folds:
        src_path = source_dir / f"source_fold{outer}.pt"
        ln_path = ln_dir / f"finetune_papagei_ln_head_fold{outer}.pt"
        if not src_path.is_file() or not ln_path.is_file():
            raise SystemExit(f"missing fold checkpoints: {src_path} or {ln_path}")
        src_ck = torch.load(src_path, map_location="cpu", weights_only=False)
        ln_ck = torch.load(ln_path, map_location="cpu", weights_only=False)
        if int(src_ck["test_fold"]) != outer or int(ln_ck["test_fold"]) != outer:
            raise SystemExit(f"checkpoint fold metadata mismatch at outer fold {outer}")
        val_fold = int(src_ck["val_fold"])
        train_folds = [f for f in folds if f not in (outer, val_fold)]
        median_y = float(index.loc[index.fold.isin(train_folds), "spo2_median"].median())
        test = index[index.fold.eq(outer)]
        case_ids = sorted(test.case_id.unique().astype(int))
        encoder = load_encoder(src_ck, a, device)
        source_head, ln_head = make_head(src_ck, device), make_head(ln_ck, device)
        for shift_name, severity in SHIFTS:
            print(f"fold={outer} shift={shift_name} cases={len(case_ids)}", flush=True)
            for case_id in case_ids:
                frame = test[test.case_id.eq(case_id)].sort_values("start_s").reset_index(drop=True)
                cache_path = str(frame.cache_path.iloc[0])
                with np.load(cache_path) as cache:
                    x_clean = cache["ppg"][frame.window_idx.to_numpy(dtype=np.int64)].astype(np.float32, copy=False)
                starts = frame.start_s.to_numpy(dtype=np.int64)
                if shift_name.startswith("gain_"):
                    kind, level = "gain", severity
                elif shift_name.startswith("baseline_drift"):
                    kind, level = "baseline_drift", severity
                elif shift_name.startswith("gaussian_noise"):
                    kind, level = "gaussian_noise", severity
                else:
                    kind, level = "clean", 0.0
                x = apply_patient_shift(x_clean, kind, level, case_id, starts)
                # The synthetic view is created after the project's per-window z-score preprocessing.
                torch.manual_seed(20261012 + outer * 1009 + case_id + len(shift_name))
                z_raw, z_view = encode(encoder, x, batch_size=a.encoder_batch_size,
                                       device=device, augment=True)
                assert z_view is not None
                h_src = torch.from_numpy((z_raw - src_ck["x_mean"]) / np.maximum(src_ck["x_std"], 1e-6)).float().to(device)
                h_view = torch.from_numpy((z_view - src_ck["x_mean"]) / np.maximum(src_ck["x_std"], 1e-6)).float().to(device)
                h_ln = torch.from_numpy((z_raw - ln_ck["x_mean"]) / np.maximum(ln_ck["x_std"], 1e-6)).float().to(device)
                warm = min(a.warmup_windows, max(len(frame) - 2, 0))
                if warm < 2:
                    raise RuntimeError(f"case {case_id} has too few windows for warmup plus scoring")
                score_ids = np.arange(warm, len(frame))
                with torch.no_grad():
                    pred_source = source_head(h_src[score_ids]).flatten().cpu().numpy() * float(src_ck["y_std"]) + float(src_ck["y_mean"])
                    pred_ln = ln_head(h_ln[score_ids]).flatten().cpu().numpy() * float(ln_ck["y_std"]) + float(ln_ck["y_mean"])
                case_predictions = {"frozen": pred_source, "ln_head": pred_ln,
                                    "train_median": None}
                for mode in METHODS:
                    # A fresh adapter and optimizer state for each case/ablation.
                    tta = PatientAdapterTTA(source_head, bottleneck_dim=a.bottleneck,
                                            dropout=a.dropout, lr=a.lr, steps=a.steps,
                                            mode=mode, mc_samples=a.mc_samples,
                                            uncertainty_scale=a.uncertainty_scale,
                                            identity_weight=a.identity_weight)
                    warm_diag = tta.update(h_src[:warm], h_view[:warm])
                    pred_parts = []
                    for start in range(warm, len(frame), a.batch_size):
                        end = min(start + a.batch_size, len(frame))
                        pred_parts.append(tta.predict(h_src[start:end]).cpu().numpy())
                        # Prequential order: current batch is scored before its unlabeled signals update the adapter.
                        if end < len(frame):
                            diag = tta.update(h_src[start:end], h_view[start:end])
                            update_rows.append({"test_fold": outer, "case_id": case_id,
                                                "shift": shift_name, "method": mode,
                                                "batch_start_s": int(starts[start]), **diag})
                    pred = np.concatenate(pred_parts) * float(src_ck["y_std"]) + float(src_ck["y_mean"])
                    case_predictions[mode] = pred
                    update_rows.append({"test_fold": outer, "case_id": case_id,
                                        "shift": shift_name, "method": mode,
                                        "batch_start_s": int(starts[0]), "phase": "warmup", **warm_diag})
                # Read labels only after all unlabeled warmup/online updates for this case are complete.
                y = frame.spo2_median.to_numpy(dtype=np.float32)[score_ids]
                case_predictions["train_median"] = np.full(len(y), median_y, dtype=np.float32)
                for method, pred in case_predictions.items():
                    case_rows.append({"test_fold": outer, "case_id": case_id, "shift": shift_name,
                                      "method": method, "warmup_windows_excluded": warm,
                                      **case_metrics(y, pred)})
                encoder.to(device)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cases = pd.DataFrame([r for r in case_rows if r["test_fold"] == outer])
        fold_metric = cases.groupby(["test_fold", "shift", "method"], as_index=False).agg(
            n_cases=("case_id", "nunique"), case_mean_mae=("case_mae", "mean"),
            case_mae_sd=("case_mae", "std"), case_mean_rmse=("case_rmse", "mean"),
            case_mean_bias=("case_bias", "mean"), mean_prediction_std=("prediction_std", "mean"),
            scored_windows=("n_windows_scored", "sum"))
        fold_rows.extend(fold_metric.to_dict("records"))
        pd.DataFrame(case_rows).to_csv(out / "case_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
        pd.DataFrame(update_rows).to_csv(out / "update_diagnostics.csv", index=False)

    case_df = pd.DataFrame(case_rows)
    fold_df = pd.DataFrame(fold_rows)
    comparisons = []
    for shift_name, _ in SHIFTS:
        for method in ("ln_head", "train_median", *METHODS):
            comparisons.append(bootstrap_improvement(case_df, method, shift_name,
                                                      a.bootstrap_reps, 20261012))
    cmp_df = pd.DataFrame(comparisons)
    cmp_df.to_csv(out / "paired_bootstrap_summary.csv", index=False)
    summary = {
        "dataset": "pre-existing VitalDB only",
        "estimand_scope": "case-aware episodic adaptation; patient identifiers unavailable, so patient-level generalization is not claimed",
        "case_split": "existing five case-disjoint outer folds; source and LN+Head checkpoints fixed per fold",
        "adaptation_scope": "state and optimizer reset for each held-out case; first 32 unlabeled windows are warmup and excluded from scoring; later batches are predicted before their unlabeled update",
        "labels_used_in_update": False,
        "synthetic_shift_scope": "perturbations are applied after existing per-window z-score preprocessing and are model-input stress tests, not raw-sensor or clinical patient-shift evidence",
        "config": {"bottleneck": a.bottleneck, "lr": a.lr, "steps_per_batch": a.steps,
                   "batch_size": a.batch_size, "warmup_windows": a.warmup_windows,
                   "dropout": a.dropout, "mc_samples": a.mc_samples,
                   "uncertainty_scale_normalized_target_units": a.uncertainty_scale,
                   "lambda_patient": 0.1, "identity_guard_weight": a.identity_weight,
                   "shifts": [{"name": n, "severity": v} for n,v in SHIFTS],
                   "modes": {m: {"lambda_consistency": PatientAdapterTTA.MODES[m][0],
                                 "lambda_patient": PatientAdapterTTA.MODES[m][1],
                                 "uncertainty_weighted": PatientAdapterTTA.MODES[m][2]} for m in METHODS}},
        "metrics": "per-case MAE, macro mean, paired case bootstrap 95% CI for frozen-source minus method MAE",
        "fold_mean_case_mae": fold_df.groupby(["shift", "method"]).case_mean_mae.mean().unstack().round(6).to_dict(),
        "paired_bootstrap_file": "paired_bootstrap_summary.csv",
        "case_metrics_file": "case_metrics.csv",
        "updates_file": "update_diagnostics.csv",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
