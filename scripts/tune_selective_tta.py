#!/usr/bin/env python3
"""Choose a compact selective-TTA configuration on one outer fold's validation cases."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def candidates() -> list[dict]:
    return [
        {"name": "no_update", "quality_mode": "gated", "quality_threshold": 1.01,
         "uncertainty_mode": "off", "lr": 1e-5},
        {"name": "quality_only", "quality_mode": "gated", "quality_threshold": 0.5,
         "uncertainty_mode": "off", "lr": 1e-6},
        {"name": "uncertainty_only", "quality_mode": "ungated", "quality_threshold": 0.5,
         "uncertainty_mode": "gated", "uncertainty_threshold": 0.01, "lr": 1e-6},
        {"name": "quality_plus_uncertainty", "quality_mode": "gated", "quality_threshold": 0.5,
         "uncertainty_mode": "gated", "uncertainty_threshold": 0.01, "lr": 1e-6},
        {"name": "quality_plus_uncertainty_weighted", "quality_mode": "gated",
         "quality_threshold": 0.5, "uncertainty_mode": "weighted",
         "uncertainty_scale": 0.01, "lr": 1e-6},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--source", required=True, help="source checkpoint for one outer fold")
    p.add_argument("--backbone", choices=["papagei", "sigma"], required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mc-samples", type=int, default=8)
    a = p.parse_args()
    source = Path(a.source)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    val_fold = int(ckpt["val_fold"])
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in candidates():
        run_dir = root / cfg["name"]
        cmd = [sys.executable, "scripts/evaluate_tta.py", "--index", a.index,
               "--source", str(source), "--test-fold", str(a.outer_fold),
               "--eval-fold", str(val_fold), "--role", "validation",
               "--backbone", a.backbone, "--repo", a.repo,
               "--checkpoint", a.checkpoint, "--device", a.device,
               "--batch-size", str(a.batch_size), "--mc-samples", str(a.mc_samples),
               "--quality-mode", cfg["quality_mode"], "--quality-threshold", str(cfg["quality_threshold"]),
               "--uncertainty-mode", cfg["uncertainty_mode"], "--lr", str(cfg["lr"]),
               "--lambda-morph", "0", "--out-dir", str(run_dir), "--run-name", cfg["name"]]
        for key in ("uncertainty_threshold", "uncertainty_scale"):
            if key in cfg:
                cmd.extend([f"--{key.replace('_', '-')}", str(cfg[key])])
        if a.backbone == "sigma":
            cmd.extend(["--sigma-patch-size", "100"])
        proc = subprocess.run(cmd, text=True, capture_output=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "runner.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr,
                                            encoding="utf-8")
        if proc.returncode:
            raise SystemExit(f"TTA candidate {cfg['name']} failed: {proc.stderr[-2000:]}")
        metrics_path = run_dir / f"metrics_validation_fold{val_fold}_{cfg['name']}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append({**cfg, "validation_mean_case_mae": metrics["tta_mean_case_mae"],
                        "validation_source_mean_case_mae": metrics["source_mean_case_mae"],
                        "updated_samples": metrics["adaptation_diagnostics"]["updated_samples"],
                        "seen_samples": metrics["adaptation_diagnostics"]["seen_samples"],
                        "mean_mc_predictive_variance": metrics["adaptation_diagnostics"]["mean_mc_predictive_variance"]})
    best = min(results, key=lambda r: (round(r["validation_mean_case_mae"], 6),
                                       r["name"] != "no_update"))
    selection = {"backbone": a.backbone, "outer_fold": a.outer_fold,
                 "validation_fold": val_fold, "selection_metric": "mean case MAE",
                 "labels_used_for_update": False, "candidates": results,
                 "selected": best,
                 "note": "validation-only exploratory selection; the same fold was also used for source early stopping"}
    (root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
