#!/usr/bin/env python3
"""Run a prespecified, case-folded TTA component ablation on VitalDB."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "consistency_only": {"quality_mode": "ungated", "lambda_morph": 0.0},
    "quality_gated_consistency": {"quality_mode": "gated", "lambda_morph": 0.0},
    "morphology_without_quality": {"quality_mode": "ungated", "lambda_morph": 0.1},
    "full_qm": {"quality_mode": "gated", "lambda_morph": 0.1},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--backbones", default="papagei,sigma")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-reg", type=float, default=1e-4)
    p.add_argument("--quality-threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    a = p.parse_args()
    root = Path(a.results_root)
    for backbone in [v.strip() for v in a.backbones.split(",") if v.strip()]:
        if backbone not in {"papagei", "sigma"}:
            raise SystemExit(f"unsupported backbone: {backbone}")
        data_name = "vitaldb_pilot" if backbone == "papagei" else "vitaldb_sigma"
        index = root / data_name / "window_index.csv"
        source_dir = root / data_name / "source"
        for fold in [int(v) for v in a.folds.split(",") if v.strip()]:
            source = source_dir / f"source_fold{fold}.pt"
            if not source.is_file():
                raise SystemExit(f"missing source checkpoint: {source}")
            for name, variant in VARIANTS.items():
                out_dir = root / "tta_ablation" / backbone / name / f"fold{fold}"
                command = [sys.executable, "scripts/evaluate_tta.py",
                           "--index", str(index), "--source", str(source),
                           "--test-fold", str(fold), "--role", "test",
                           "--backbone", backbone, "--repo", a.repo,
                           "--checkpoint", a.checkpoint, "--device", a.device,
                           "--batch-size", str(a.batch_size), "--lr", str(a.lr),
                           "--lambda-morph", str(variant["lambda_morph"]),
                           "--lambda-reg", str(a.lambda_reg),
                           "--quality-threshold", str(a.quality_threshold),
                           "--quality-mode", variant["quality_mode"],
                           "--out-dir", str(out_dir), "--run-name", name]
                if backbone == "sigma":
                    command.extend(["--sigma-patch-size", "100"])
                completed = subprocess.run(command, text=True, capture_output=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "runner.log").write_text(
                    completed.stdout + "\n--- stderr ---\n" + completed.stderr,
                    encoding="utf-8")
                if completed.returncode:
                    raise SystemExit(f"{backbone} fold {fold} {name} failed:\n{completed.stderr[-2500:]}")
                print(f"completed {backbone} fold {fold}: {name}", flush=True)


if __name__ == "__main__":
    main()
