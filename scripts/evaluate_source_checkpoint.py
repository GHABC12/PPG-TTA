#!/usr/bin/env python3
"""Evaluate a saved source model on one held-out VitalDB fold."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmtta.papagei import PaPaGeiEncoder, StandardizedEncoder
from qmtta.sigma import SigmaPPGEncoder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", choices=["papagei", "sigma"], required=True)
    p.add_argument("--test-fold", type=int, required=True)
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--pretrained", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--sigma-repo", default=os.environ.get("SIGMA_REPO", "third_party/SigmaPPG"))
    p.add_argument("--sigma-checkpoint", default=os.environ.get("SIGMA_CHECKPOINT", "weights/sigma.pth"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    frame = pd.read_csv(a.index)
    fold = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    test = frame[frame.fold == a.test_fold].copy()
    if test.empty or int(fold["test_fold"]) != a.test_fold:
        raise SystemExit("held-out fold is empty or does not match checkpoint")
    device = torch.device(a.device)
    if a.backbone == "papagei":
        base = PaPaGeiEncoder(a.repo, a.pretrained, a.device)
    else:
        input_size = int((test.end_s.iloc[0] - test.start_s.iloc[0]) * test.fs.iloc[0])
        base = SigmaPPGEncoder(a.sigma_repo, a.sigma_checkpoint, input_size=input_size,
                               patch_size=int(fold["protocol"].get("patch_size", 100)),
                               device=a.device)
    saved = fold.get("encoder_state", {})
    if saved:
        incompatible = base.load_state_dict(saved, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected encoder weights: {incompatible.unexpected_keys}")
    encoder = StandardizedEncoder(base, fold["x_mean"], fold["x_std"]).to(device).eval()
    dim = len(fold["x_mean"])
    spo2 = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
            if fold.get("strategy") == "ln_head" else nn.Linear(dim, 1))
    spo2.load_state_dict({k.removeprefix("spo2."): v for k, v in fold["model"].items()
                          if k.startswith("spo2.")})
    spo2 = spo2.to(device).eval()
    output = []
    # Read only waveforms until every prediction has been produced.
    with torch.no_grad():
        for case_id, case in test.groupby("case_id", sort=True):
            case = case.sort_values("start_s")
            with np.load(str(case.cache_path.iloc[0])) as cache:
                signals = cache["ppg"][case.window_idx.to_numpy(dtype=int)]
            for start in range(0, len(signals), a.batch_size):
                x = torch.from_numpy(signals[start:start+a.batch_size]).unsqueeze(1).to(device)
                pred = spo2(encoder(x)).flatten().cpu().numpy() * float(fold["y_std"]) + float(fold["y_mean"])
                for j, value in enumerate(pred):
                    output.append({"case_id": int(case_id),
                                   "start_s": int(case.start_s.iloc[start+j]),
                                   "prediction": float(value)})
    predictions = pd.DataFrame(output).merge(
        test[["case_id", "start_s", "spo2_median"]], on=["case_id", "start_s"],
        how="left", validate="one_to_one")
    train = frame.loc[~frame.fold.isin([a.test_fold, int(fold["val_fold"])]), "spo2_median"]
    train_median, train_mean = float(train.median()), float(train.mean())
    predictions["train_median_constant"] = train_median
    predictions["train_mean_constant"] = train_mean
    error = predictions.prediction - predictions.spo2_median
    case_mae = np.abs(error).groupby(predictions.case_id).mean()
    median_case_mae = np.abs(predictions.train_median_constant - predictions.spo2_median).groupby(predictions.case_id).mean()
    report = {"backbone": a.backbone, "strategy": fold.get("strategy", "linear_probe"),
              "test_fold": a.test_fold, "validation_fold": int(fold["val_fold"]),
              "n_windows": int(len(predictions)), "n_cases": int(len(case_mae)),
              "window_mae": float(np.abs(error).mean()),
              "mean_case_mae": float(case_mae.mean()),
              "train_median_baseline_mean_case_mae": float(median_case_mae.mean()),
              "case_mae": {str(int(k)): float(v) for k, v in case_mae.items()},
              "prediction_std": float(predictions.prediction.std()),
              "labels_used_for_training_or_selection": False}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
