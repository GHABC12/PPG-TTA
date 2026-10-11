#!/usr/bin/env python3
"""Summarize validation selection and case-disjoint baseline strategy OOF results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/vitaldb_pilot")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    root, out_dir = Path(a.root), Path(a.out_dir)
    strategies = ["linear_probe", "ln_head", "last_layers", "full"]
    metric_rows, all_cases = [], []
    validation = {}
    for strategy in strategies:
        for fold in range(a.folds):
            if strategy == "linear_probe":
                ckpt_path = root / "source" / f"source_fold{fold}.pt"
                pred_path = root / "baseline_eval" / f"{strategy}_fold{fold}.csv"
                metric_path = root / "baseline_eval" / f"{strategy}_fold{fold}.json"
            else:
                ckpt_path = root / "baselines" / strategy / f"finetune_papagei_{strategy}_fold{fold}.pt"
                pred_path = root / "baseline_eval" / f"{strategy}_fold{fold}.csv"
                metric_path = root / "baseline_eval" / f"{strategy}_fold{fold}.json"
            if not (ckpt_path.exists() and pred_path.exists() and metric_path.exists()):
                raise SystemExit(f"missing completed evaluation: {strategy} fold {fold}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if strategy == "linear_probe":
                val_report = json.loads((root / "source" / f"metrics_fold{fold}.json").read_text(encoding="utf-8"))
                val_mae = float(val_report["best_validation_mean_case_mae"])
            else:
                val_mae = float(ckpt.get("best_validation_mean_case_mae", ckpt.get("best_validation_mae", np.nan)))
            validation[(strategy, fold)] = val_mae
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            metric_rows.append({"strategy": strategy, "fold": fold,
                                "validation_mean_case_mae": val_mae,
                                "test_mean_case_mae": metric["mean_case_mae"],
                                "train_median_baseline_mean_case_mae": metric["train_median_baseline_mean_case_mae"],
                                "test_window_mae": metric["window_mae"],
                                "test_prediction_std": metric["prediction_std"]})
            cases = pd.read_csv(pred_path)
            cases["fold"] = fold
            cases["strategy"] = strategy
            all_cases.append(cases)
    metrics = pd.DataFrame(metric_rows)
    cases = pd.concat(all_cases, ignore_index=True)
    picked = []
    for fold in range(a.folds):
        winner = min(strategies, key=lambda s: validation[(s, fold)])
        picked.append({"fold": fold, "selected_strategy": winner,
                       "validation_mean_case_mae": validation[(winner, fold)]})
    picks = pd.DataFrame(picked)
    selected = cases.merge(picks[["fold", "selected_strategy"]], on="fold")
    selected = selected[selected.strategy == selected.selected_strategy].copy()
    selected_error = np.abs(selected.prediction - selected.spo2_median)
    selected_case = selected.assign(abs_error=selected_error).groupby("case_id").abs_error.mean()
    strategy_case = cases.assign(abs_error=np.abs(cases.prediction - cases.spo2_median)).groupby(
        ["strategy", "case_id"], as_index=False).agg(case_mae=("abs_error", "mean"),
                                                      n_windows=("abs_error", "size"))
    median_case = cases.assign(abs_error=np.abs(cases.train_median_constant - cases.spo2_median)).groupby(
        "case_id").abs_error.mean()
    selected_delta = selected_case.reindex(median_case.index).to_numpy() - median_case.to_numpy()
    rng = np.random.default_rng(42)
    boot = np.asarray([rng.choice(selected_delta, size=len(selected_delta), replace=True).mean()
                       for _ in range(5000)])
    summary = {
        "selection_rule": "minimum validation mean-case MAE within each outer training split",
        "strategies": metrics.groupby("strategy").agg(
            folds=("fold", "count"), mean_validation_case_mae=("validation_mean_case_mae", "mean"),
            mean_outer_test_case_mae=("test_mean_case_mae", "mean"),
            mean_train_median_baseline_case_mae=("train_median_baseline_mean_case_mae", "mean"),
            mean_outer_test_window_mae=("test_window_mae", "mean")).reset_index().to_dict("records"),
        "oof_case_level_mae_by_strategy": strategy_case.groupby("strategy").case_mae.mean().to_dict(),
        "selected_by_validation": {
            "fold_selection": picked,
            "outer_oof_cases": int(len(selected_case)),
            "outer_oof_mean_case_mae": float(selected_case.mean()),
            "train_median_baseline_mean_case_mae": float(median_case.mean()),
            "selected_minus_median_case_mae": float(selected_delta.mean()),
            "selected_minus_median_case_bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                                                             float(np.percentile(boot, 97.5))],
        },
        "caveat": "single seed and 23 VitalDB cases; validation strategy selection is exploratory, not an independent clinical validation",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "baseline_strategy_metrics.csv", index=False)
    strategy_case.to_csv(out_dir / "baseline_strategy_case_metrics.csv", index=False)
    (out_dir / "baseline_strategy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
