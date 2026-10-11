#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "prediction_std": float(np.std(pred))}


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate five-fold case-level VitalDB pilot outputs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    root = Path(a.results_root)
    fold_rows, case_rows, oof_rows, fold_details = [], [], [], {}
    for backbone, folder in [("papagei", "vitaldb_pilot"), ("sigma", "vitaldb_sigma")]:
        base = root / folder
        index = pd.read_csv(base / "window_index.csv")
        if set(index.fold.unique().astype(int)) != set(range(a.folds)):
            raise SystemExit(f"{backbone}: expected fold IDs 0..{a.folds-1}")
        predictions = []
        for fold in range(a.folds):
            source = json.loads((base / "source" / f"metrics_fold{fold}.json").read_text())
            tta_metrics = json.loads((base / "tta" / f"metrics_fold{fold}.json").read_text())
            pred = pd.read_csv(base / "tta" / f"predictions_fold{fold}.csv")
            pred["fold"] = fold
            pred["backbone"] = backbone
            folds = sorted(index.fold.unique().astype(int).tolist())
            val_fold = folds[(folds.index(fold) + 1) % len(folds)]
            train = index.loc[~index.fold.isin([fold, val_fold]), "spo2_median"].to_numpy()
            med = float(np.median(train)); mean = float(np.mean(train))
            pred["train_median_prediction"] = med
            pred["train_mean_prediction"] = mean
            predictions.append(pred)
            fold_details[f"{backbone}_fold{fold}"] = {
                "test_case_ids": tta_metrics["test_case_ids"],
                "source": source["test"], "train_median_constant": source["train_median_baseline"],
                "episodic_qm_tta": tta_metrics["episodic_qm_tta"],
                "delta_mae_source_minus_tta": tta_metrics["delta_mae"]}
            for method, metrics in [("source_only_lp", source["test"]),
                                    ("train_median_constant", source["train_median_baseline"]),
                                    ("train_mean_constant", source["train_mean_baseline"]),
                                    ("episodic_qm_tta", tta_metrics["episodic_qm_tta"])]:
                fold_rows.append({"backbone": backbone, "fold": fold, "method": method, **metrics})

        all_pred = pd.concat(predictions, ignore_index=True)
        oof_rows.append(all_pred)
        for method, col in [("source_only_lp", "source_prediction"),
                            ("episodic_qm_tta", "tta_prediction"),
                            ("train_median_constant", "train_median_prediction"),
                            ("train_mean_constant", "train_mean_prediction")]:
            result = score(all_pred.spo2_true.to_numpy(), all_pred[col].to_numpy())
            fold_rows.append({"backbone": backbone, "fold": "OOF", "method": method, **result})

        for case_id, case in all_pred.groupby("case_id"):
            y = case.spo2_true.to_numpy()
            for method, col in [("source_only_lp", "source_prediction"),
                                ("episodic_qm_tta", "tta_prediction"),
                                ("train_median_constant", "train_median_prediction"),
                                ("train_mean_constant", "train_mean_prediction")]:
                case_rows.append({"backbone": backbone, "case_id": int(case_id),
                                  "fold": int(case.fold.iloc[0]), "method": method,
                                  **score(y, case[col].to_numpy())})

    fold_df = pd.DataFrame(fold_rows)
    case_df = pd.DataFrame(case_rows)
    fold_df.to_csv(root / "baseline_results.csv", index=False)
    case_df.to_csv(root / "case_metrics.csv", index=False)
    summaries = {}
    for (backbone, method), group in case_df.groupby(["backbone", "method"]):
        mae = group.mae.to_numpy()
        summaries[f"{backbone}_{method}"] = {
            "cases": int(len(group)), "mean_case_mae": float(mae.mean()),
            "std_case_mae": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
            "median_case_mae": float(np.median(mae))}
    summary = {"folds": a.folds, "case_grouped": True,
               "fold_metrics": fold_details, "case_level_mae": summaries,
               "note": "Out-of-fold results from one seed; no patient bootstrap CI; feasibility only."}
    (root / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

