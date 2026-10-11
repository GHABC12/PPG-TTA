#!/usr/bin/env python3
"""Create paired case-bootstrap summaries from completed PATTA CSV outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHIFTS = ("clean", "gain_0.8", "gain_1.2", "baseline_drift_0.1", "gaussian_noise_0.05")
METHODS = ("frozen", "ln_head", "train_median", "consistency_only", "patient_only",
           "uncertainty_weighted", "full_patta")


def bootstrap(case_df: pd.DataFrame, method: str, shift: str, reps: int, seed: int) -> dict:
    part = case_df[case_df["shift"].eq(shift)]
    wide = part.pivot(index="case_id", columns="method", values="case_mae").dropna(subset=["frozen", method])
    difference = (wide["frozen"] - wide[method]).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.mean(difference[rng.integers(0, len(difference), (reps, len(difference)))], axis=1)
    return {"shift": shift, "method": method, "n_cases": int(len(difference)),
            "source_minus_method_case_mae": float(difference.mean()),
            "paired_bootstrap_ci95_low": float(np.quantile(samples, .025)),
            "paired_bootstrap_ci95_high": float(np.quantile(samples, .975))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", default="results/phase3_patient_aware_tta")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261012)
    a = p.parse_args(); root = Path(a.result_dir)
    case = pd.read_csv(root / "case_metrics.csv")
    fold = pd.read_csv(root / "fold_metrics.csv")
    expected_cases = 23 * len(SHIFTS) * len(METHODS)
    if len(case) != expected_cases:
        raise SystemExit(f"incomplete case metrics: expected {expected_cases} rows, found {len(case)}")
    comparisons = [bootstrap(case, method, shift, a.bootstrap_reps, a.seed)
                   for shift in SHIFTS for method in METHODS if method != "frozen"]
    pd.DataFrame(comparisons).to_csv(root / "paired_bootstrap_summary.csv", index=False)
    table = fold.groupby(["shift", "method"], as_index=False).agg(
        case_mean_mae=("case_mean_mae", "mean"), case_mae_sd=("case_mean_mae", "std"),
        case_mean_rmse=("case_mean_rmse", "mean"), case_mean_bias=("case_mean_bias", "mean"),
        mean_prediction_std=("mean_prediction_std", "mean"), n_scored_windows=("scored_windows", "sum"))
    table.to_csv(root / "summary_by_shift_method.csv", index=False)
    metrics = {}
    for shift in SHIFTS:
        metrics[shift] = {}
        for row in table[table["shift"].eq(shift)].itertuples(index=False):
            metrics[shift][row.method] = {"case_mean_mae": float(row.case_mean_mae),
                                          "case_mae_sd": float(row.case_mae_sd),
                                          "case_mean_rmse": float(row.case_mean_rmse),
                                          "case_mean_bias": float(row.case_mean_bias),
                                          "mean_prediction_std": float(row.mean_prediction_std),
                                          "scored_windows": int(row.n_scored_windows)}
    summary = {
        "dataset": "pre-existing VitalDB only",
        "estimand_scope": "case-aware episodic adaptation; patient identifiers unavailable, so patient-level generalization is not claimed",
        "case_split": "existing five case-disjoint outer folds; source and LN+Head checkpoints fixed per fold",
        "adaptation_scope": "state and optimizer reset for each held-out case; first 32 unlabeled windows are warmup and excluded from scoring; later batches are predicted before their unlabeled update",
        "labels_used_in_update": False,
        "synthetic_shift_scope": "perturbations are applied after existing per-window z-score preprocessing and are model-input stress tests, not raw-sensor or clinical patient-shift evidence",
        "config": {"bottleneck": 32, "lr": 1e-4, "steps_per_batch": 5, "batch_size": 32,
                   "warmup_windows": 32, "dropout": 0.1, "mc_samples": 8,
                   "uncertainty_scale_normalized_target_units": 1.0, "lambda_patient": 0.1,
                   "identity_guard_weight": 0.01,
                   "shifts": [{"name": "clean", "severity": 0.0}, {"name": "gain_0.8", "severity": 0.8},
                              {"name": "gain_1.2", "severity": 1.2}, {"name": "baseline_drift_0.1", "severity": 0.1},
                              {"name": "gaussian_noise_0.05", "severity": 0.05}]},
        "metrics": "per-case MAE, macro mean, paired case bootstrap 95% CI for frozen-source minus method MAE",
        "by_shift_method": metrics,
        "paired_bootstrap_file": "paired_bootstrap_summary.csv",
        "case_metrics_file": "case_metrics.csv", "fold_metrics_file": "fold_metrics.csv",
        "updates_file": "update_diagnostics.csv",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
