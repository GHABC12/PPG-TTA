#!/usr/bin/env python3
"""Select a small TTA configuration per outer fold using only its validation fold."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def candidates() -> list[dict]:
    # Prespecified 2 x 3 grid from the implementation plan, plus a no-update
    # candidate so selection can retain source-only behavior if TTA hurts.
    grid = [
        {"name": f"lr{lr:g}_m{morph:g}", "lr": lr, "lambda_morph": morph,
         "lambda_reg": 1e-4, "quality_threshold": 0.5}
        for lr, morph in itertools.product((1e-6, 1e-5), (0.0, 0.1, 0.5))
    ]
    grid.append({"name": "no_update", "lr": 1e-5, "lambda_morph": 0.0,
                 "lambda_reg": 1e-4, "quality_threshold": 1.01})
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["papagei", "sigma"], required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--source-dir", required=True,
                   help="directory containing source_foldK.pt or finetune_BACKBONE_foldK.pt")
    p.add_argument("--model-kind", choices=["lp", "finetune"], default="lp")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--run-test", action="store_true",
                   help="after selecting on validation, evaluate that fixed config once on the outer test fold")
    a = p.parse_args()
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for outer_fold in [int(v) for v in a.folds.split(",") if v.strip()]:
        if a.model_kind == "lp":
            checkpoint = Path(a.source_dir) / f"source_fold{outer_fold}.pt"
        else:
            checkpoint = Path(a.source_dir) / f"finetune_{a.backbone}_fold{outer_fold}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"missing source checkpoint: {checkpoint}")
        fold_results = []
        for cfg in candidates():
            run_dir = root / f"{a.backbone}_{a.model_kind}_outer{outer_fold}_{cfg['name']}"
            command = [sys.executable, "scripts/evaluate_tta.py",
                       "--index", a.index, "--source", str(checkpoint),
                       "--test-fold", str(outer_fold), "--eval-fold",
                       str(_validation_fold(checkpoint)), "--role", "validation",
                       "--backbone", a.backbone, "--repo", a.repo,
                       "--checkpoint", a.checkpoint, "--device", a.device,
                       "--batch-size", str(a.batch_size), "--quality-threshold",
                       str(cfg["quality_threshold"]), "--lr", str(cfg["lr"]),
                       "--lambda-morph", str(cfg["lambda_morph"]),
                       "--lambda-reg", str(cfg["lambda_reg"]),
                       "--out-dir", str(run_dir), "--run-name", cfg["name"]]
            if a.backbone == "sigma":
                command.extend(["--sigma-patch-size", "100"])
            completed = subprocess.run(command, text=True, capture_output=True)
            (run_dir / "runner.log").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "runner.log").write_text(
                completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise SystemExit(f"candidate failed ({cfg['name']}): {completed.stderr[-2000:]}")
            metrics_path = run_dir / f"metrics_validation_fold{_validation_fold(checkpoint)}_{cfg['name']}.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            fold_results.append({**cfg, "validation_mean_case_mae": metrics["tta_mean_case_mae"],
                                 "validation_window_mae": metrics["episodic_qm_tta"]["mae"],
                                 "validation_prediction_std": metrics["episodic_qm_tta"]["prediction_std"],
                                 "run_dir": str(run_dir)})
        # Prefer simpler, smaller updates when validation scores are numerically tied.
        chosen = min(fold_results, key=lambda r: (round(r["validation_mean_case_mae"], 6),
                                                   r["name"] != "no_update",
                                                   r["lr"], r["lambda_morph"]))
        selection = {"backbone": a.backbone, "model_kind": a.model_kind,
                     "outer_test_fold": outer_fold,
                     "validation_fold": _validation_fold(checkpoint),
                     "selection_metric": "mean case MAE",
                     "validation_reused_for_source_early_stopping": True,
                     "selected": chosen, "candidates": fold_results}
        if a.run_test:
            test_dir = root / f"{a.backbone}_{a.model_kind}_outer{outer_fold}_test"
            command = [sys.executable, "scripts/evaluate_tta.py",
                       "--index", a.index, "--source", str(checkpoint),
                       "--test-fold", str(outer_fold), "--role", "test",
                       "--backbone", a.backbone, "--repo", a.repo,
                       "--checkpoint", a.checkpoint, "--device", a.device,
                       "--batch-size", str(a.batch_size), "--quality-threshold",
                       str(chosen["quality_threshold"]), "--lr", str(chosen["lr"]),
                       "--lambda-morph", str(chosen["lambda_morph"]),
                       "--lambda-reg", str(chosen["lambda_reg"]),
                       "--out-dir", str(test_dir), "--run-name", chosen["name"]]
            if a.backbone == "sigma":
                command.extend(["--sigma-patch-size", "100"])
            completed = subprocess.run(command, text=True, capture_output=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "runner.log").write_text(
                completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise SystemExit(f"outer test evaluation failed: {completed.stderr[-2000:]}")
            test_metrics = test_dir / f"metrics_test_fold{outer_fold}_{chosen['name']}.json"
            selection["test_result_file"] = str(test_metrics)
        destination = root / f"selected_{a.backbone}_{a.model_kind}_fold{outer_fold}.json"
        destination.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        results.append(selection)
        print(json.dumps({"outer_fold": outer_fold, "selected": chosen}, indent=2))
    (root / f"selection_{a.backbone}_{a.model_kind}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


def _validation_fold(checkpoint_path: Path) -> int:
    import torch
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(state["val_fold"])


if __name__ == "__main__":
    main()
