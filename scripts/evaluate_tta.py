#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmtta.papagei import PaPaGeiEncoder, StandardizedEncoder
from qmtta.quality_tta import EpisodicQualityMorphTTA
from qmtta.sigma import SigmaPPGEncoder


def score(y, prediction):
    err = prediction - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "prediction_std": float(np.std(prediction))}


def mean_case_mae(frame: pd.DataFrame, prediction_column: str) -> float:
    errors = frame.assign(abs_error=np.abs(frame[prediction_column] - frame.spo2_true))
    return float(errors.groupby("case_id").abs_error.mean().mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate episodic, unlabeled QM-TTA on one held-out case fold")
    p.add_argument("--index", default="results/vitaldb_pilot/window_index.csv")
    p.add_argument("--source", default="results/vitaldb_pilot/source/source_fold0.pt")
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--eval-fold", type=int, help="fold to score; defaults to test-fold")
    p.add_argument("--role", choices=["validation", "test"], default="test",
                   help="validation scores may select settings; test scores are final only")
    p.add_argument("--backbone", choices=["papagei", "sigma"], default="papagei")
    p.add_argument("--sigma-patch-size", type=int, default=100)
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--sigma-repo", default=os.environ.get("SIGMA_REPO", "third_party/SigmaPPG"))
    p.add_argument("--sigma-checkpoint", default=os.environ.get("SIGMA_CHECKPOINT", "weights/sigma.pth"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--quality-threshold", type=float, default=0.5)
    p.add_argument("--quality-mode", choices=["gated", "ungated"], default="gated",
                   help="gated filters/weights by quality; ungated uses every sample uniformly")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-morph", type=float, default=0.1)
    p.add_argument("--lambda-reg", type=float, default=1e-4)
    p.add_argument("--out-dir", default="results/vitaldb_pilot/tta")
    p.add_argument("--run-name", default="", help="optional filename suffix for tuning candidates")
    a = p.parse_args()
    df = pd.read_csv(a.index)
    fold = torch.load(a.source, map_location="cpu", weights_only=False)
    if "backbone" in fold and fold["backbone"] != a.backbone:
        raise SystemExit("checkpoint backbone does not match --backbone")
    eval_fold = a.test_fold if a.eval_fold is None else a.eval_fold
    expected_val_fold = int(fold["val_fold"])
    if a.role == "validation" and eval_fold != expected_val_fold:
        raise SystemExit(f"validation role must evaluate the checkpoint validation fold {expected_val_fold}")
    if a.role == "test" and eval_fold != a.test_fold:
        raise SystemExit("test role must evaluate exactly --test-fold; use --role validation for tuning")
    if int(fold["test_fold"]) != a.test_fold:
        raise SystemExit("--test-fold does not match source checkpoint metadata")
    test = df[df.fold == eval_fold].copy()
    if test.empty:
        raise SystemExit("selected evaluation fold has no windows")
    device = torch.device(a.device)
    if a.backbone == "papagei":
        base = PaPaGeiEncoder(a.repo, a.checkpoint, a.device)
    else:
        input_size = int((test.end_s.iloc[0] - test.start_s.iloc[0]) * test.fs.iloc[0])
        base = SigmaPPGEncoder(a.sigma_repo, a.sigma_checkpoint, input_size=input_size,
                               patch_size=a.sigma_patch_size, device=a.device)
    if "encoder_state" in fold:
        saved_state = fold["encoder_state"]
        expected_keys = set(fold.get("encoder_trainable_keys", saved_state.keys()))
        if set(saved_state) != expected_keys:
            raise SystemExit("fine-tuned checkpoint parameter manifest does not match saved weights")
        incompatible = base.load_state_dict(saved_state, strict=False)
        if incompatible.unexpected_keys:
            raise SystemExit("fine-tuned encoder checkpoint does not match the selected backbone")
    teacher = StandardizedEncoder(base, fold["x_mean"], fold["x_std"]).to(device).eval()
    student = copy.deepcopy(teacher).to(device).eval()
    heads = nn.ModuleDict({"spo2": nn.Linear(len(fold["x_mean"]), 1),
                           "morph": nn.Linear(len(fold["x_mean"]), 3)})
    heads.load_state_dict(fold["model"])
    heads.to(device).eval()
    tta = EpisodicQualityMorphTTA(
        student, teacher, heads["spo2"], heads["morph"], lr=a.lr,
        lambda_morph=a.lambda_morph, lambda_reg=a.lambda_reg,
        quality_threshold=a.quality_threshold, quality_mode=a.quality_mode,
        tail_blocks=1)
    y_mu = float(fold["y_mean"]); y_sd = float(fold["y_std"])
    m_mu = np.asarray(fold["morph_mean"], dtype=np.float32)
    m_sd = np.maximum(np.asarray(fold["morph_std"], dtype=np.float32), 1e-6)
    predictions = []
    update_log = []
    for case_id, case_df in test.groupby("case_id", sort=True):
        case_df = case_df.sort_values("start_s")
        tta.reset()
        cache_path = str(case_df.cache_path.iloc[0])
        ids = case_df.window_idx.to_numpy(dtype=int)
        with np.load(cache_path) as cache:
            x_np = cache["ppg"][ids]
            q_np = cache["quality"][ids]
            m_np = (cache["morphology"][ids] - m_mu) / m_sd
            v_np = cache["morphology_valid"][ids]
        labels = case_df.spo2_median.to_numpy(dtype=np.float32)  # evaluation only; never passed to step()
        for start in range(0, len(ids), a.batch_size):
            sl = slice(start, start + a.batch_size)
            x = torch.from_numpy(x_np[sl]).unsqueeze(1).to(device)
            q = torch.from_numpy(q_np[sl]).to(device)
            m = torch.from_numpy(m_np[sl]).to(device)
            valid_m = torch.from_numpy(v_np[sl]).to(device)
            with torch.no_grad():
                pred_before = heads["spo2"](teacher(x)).flatten().cpu().numpy() * y_sd + y_mu
                # Prequential protocol: score this batch with the current model,
                # then adapt from its unlabeled PPG/morphology for later batches.
                pred_online = heads["spo2"](student(x)).flatten().cpu().numpy() * y_sd + y_mu
            for j in range(len(pred_online)):
                predictions.append({"case_id": int(case_id), "start_s": int(case_df.start_s.iloc[start+j]),
                    "spo2_true": float(labels[start+j]), "source_prediction": float(pred_before[j]),
                    "tta_prediction": float(pred_online[j]), "quality_score": float(q_np[start+j])})
            update = tta.step(x, q, m, valid_m)
            update_log.append({"case_id": int(case_id), "batch_start": int(start), **update})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(predictions)
    run_suffix = f"_{a.run_name}" if a.run_name else ""
    pred_df.to_csv(out / f"predictions_{a.role}_fold{eval_fold}{run_suffix}.csv", index=False)
    pd.DataFrame(update_log).to_csv(out / f"updates_{a.role}_fold{eval_fold}{run_suffix}.csv", index=False)
    fold_values = sorted(df.fold.unique().astype(int).tolist())
    val_fold = fold_values[(fold_values.index(a.test_fold) + 1) % len(fold_values)]
    train_median = float(df.loc[~df.fold.isin([a.test_fold, val_fold]), "spo2_median"].median())
    report = {"source": score(pred_df.spo2_true.to_numpy(), pred_df.source_prediction.to_numpy()),
              "episodic_qm_tta": score(pred_df.spo2_true.to_numpy(), pred_df.tta_prediction.to_numpy()),
              "source_mean_case_mae": mean_case_mae(pred_df, "source_prediction"),
              "tta_mean_case_mae": mean_case_mae(pred_df, "tta_prediction"),
              "delta_mae": float(np.mean(np.abs(pred_df.source_prediction-pred_df.spo2_true))-
                                 np.mean(np.abs(pred_df.tta_prediction-pred_df.spo2_true))),
              "delta_mean_case_mae": float(mean_case_mae(pred_df, "source_prediction")-
                                            mean_case_mae(pred_df, "tta_prediction")),
              "test_case_ids": sorted(map(int, pred_df.case_id.unique())),
              "train_median_constant_baseline": score(
                  pred_df.spo2_true.to_numpy(),
                  np.full(len(pred_df), train_median, dtype=np.float32)),
              "labels_used_for_update": False, "episode_reset": "per case",
              "adaptation_diagnostics": {
                  "batches": len(update_log),
                  "updated_batches": int(sum(row["updated"] > 0 for row in update_log)),
                  "updated_samples": int(sum(row["updated"] for row in update_log)),
                  "nonfinite_loss_batches": int(sum(not np.isfinite(row["loss"]) for row in update_log)),
              },
              "settings": {"quality_threshold": a.quality_threshold, "lr": a.lr,
                           "lambda_morph": a.lambda_morph, "lambda_reg": a.lambda_reg,
                           "quality_mode": a.quality_mode}}
    report.update({"outer_test_fold": a.test_fold, "evaluation_fold": eval_fold,
                   "role": a.role, "validation_fold": expected_val_fold,
                   "evaluation_case_ids": sorted(map(int, pred_df.case_id.unique())),
                   "prediction_protocol": "prequential: score current batch before adapting on its unlabeled signal/morphology",
                   "labels_used_for_update": False,
                   "labels_used_for_selection": a.role == "validation"})
    if a.role != "test":
        report.pop("test_case_ids", None)
    (out / f"metrics_{a.role}_fold{eval_fold}{run_suffix}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
