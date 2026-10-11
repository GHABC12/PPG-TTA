#!/usr/bin/env python3
"""Aggregate prespecified TTA component ablations from outer-test predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ["consistency_only", "quality_gated_consistency",
            "morphology_without_quality", "full_qm"]


def metrics(y: np.ndarray, pred: np.ndarray, frame: pd.DataFrame) -> dict:
    err = pred - y
    case = frame.assign(abs_error=np.abs(err)).groupby("case_id").abs_error.mean()
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "prediction_std": float(np.std(pred)), "mean_case_mae": float(case.mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-prefix", default="tta_ablation")
    a = p.parse_args()
    root = Path(a.results_root)
    metric_rows, case_rows = [], []
    summary = {"selection_used_outer_test_labels": False,
               "protocol": "fixed lr=1e-5, lambda_morph=0.1 where enabled, lambda_reg=1e-4; prequential; per-case reset",
               "models": {}}
    for backbone in ["papagei", "sigma"]:
        data_name = "vitaldb_pilot" if backbone == "papagei" else "vitaldb_sigma"
        index = pd.read_csv(root / data_name / "window_index.csv")
        y_parts = {"source_only": [], "train_median_constant": []}
        pred_parts = {"source_only": [], "train_median_constant": []}
        for variant in VARIANTS:
            y_parts[variant], pred_parts[variant] = [], []
        variant_diagnostics = {}
        for fold in range(5):
            for variant in VARIANTS:
                out = root / "tta_ablation" / backbone / variant / f"fold{fold}"
                reports = list(out.glob(f"metrics_test_fold{fold}_{variant}.json"))
                if len(reports) != 1:
                    raise SystemExit(f"expected one test report for {backbone}/{variant}/fold{fold}, got {len(reports)}")
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                if report.get("role") != "test" or int(report.get("outer_test_fold", -1)) != fold:
                    raise SystemExit(f"test-fold metadata mismatch: {reports[0]}")
                pred_path = out / f"predictions_test_fold{fold}_{variant}.csv"
                frame = pd.read_csv(pred_path)
                if sorted(frame.case_id.unique().astype(int)) != sorted(map(int, report["evaluation_case_ids"])):
                    raise SystemExit(f"case IDs mismatch: {pred_path}")
                y = frame.spo2_true.to_numpy()
                pred = frame.tta_prediction.to_numpy()
                y_parts[variant].append(y); pred_parts[variant].append(pred)
                fold_metrics = metrics(y, pred, frame)
                metric_rows.append({"backbone": backbone, "fold": fold, "method": variant,
                                    **fold_metrics,
                                    "updated_samples": report["adaptation_diagnostics"]["updated_samples"],
                                    "nonfinite_loss_batches": report["adaptation_diagnostics"]["nonfinite_loss_batches"]})
                err = np.abs(pred - y)
                for case_id, positions in frame.groupby("case_id").groups.items():
                    case_rows.append({"backbone": backbone, "fold": fold, "method": variant,
                                      "case_id": int(case_id), "n": len(positions),
                                      "mae": float(err[np.asarray(list(positions), dtype=int)].mean())})
                variant_diagnostics[f"fold{fold}_{variant}"] = report["adaptation_diagnostics"]

                # Source prediction and fold-specific training median are carried in every run.
                if variant == VARIANTS[0]:
                    y_parts["source_only"].append(y)
                    source_pred = frame.source_prediction.to_numpy()
                    pred_parts["source_only"].append(source_pred)
                    base = report["train_median_constant_baseline"]
                    y_parts["train_median_constant"].append(y)
                    constant = np.full(len(y), base["bias"] + y.mean())
                    pred_parts["train_median_constant"].append(constant)
                    for method, pred in (("source_only", source_pred),
                                         ("train_median_constant", constant)):
                        errors = np.abs(pred - y)
                        for case_id, positions in frame.groupby("case_id").groups.items():
                            case_rows.append({"backbone": backbone, "fold": fold, "method": method,
                                              "case_id": int(case_id), "n": len(positions),
                                              "mae": float(errors[np.asarray(list(positions), dtype=int)].mean())})
        model_summary = {}
        for method in ["source_only", "train_median_constant", *VARIANTS]:
            y = np.concatenate(y_parts[method])
            pred = np.concatenate(pred_parts[method])
            # Case IDs are unique across these selected cases, and each case occurs once in OOF.
            full = pd.concat([pd.read_csv(root / "tta_ablation" / backbone / VARIANTS[0] /
                                         f"fold{fold}" / f"predictions_test_fold{fold}_{VARIANTS[0]}.csv")
                              for fold in range(5)], ignore_index=True)
            full["_pred"] = pred
            agg = metrics(y, pred, full.rename(columns={"_pred": "tta_prediction"}).assign(
                tta_prediction=pred))
            model_summary[method] = agg
            metric_rows.append({"backbone": backbone, "fold": "OOF", "method": method, **agg})
        summary["models"][backbone] = model_summary
        case_frame = pd.DataFrame([row for row in case_rows if row["backbone"] == backbone])
        pivot = case_frame.pivot(index="case_id", columns="method", values="mae")
        rng = np.random.default_rng(42)
        paired = {}
        for variant in VARIANTS:
            difference = (pivot["source_only"] - pivot[variant]).to_numpy()
            draws = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
            paired[variant] = {"mean_case_mae_improvement_vs_source": float(difference.mean()),
                               "case_bootstrap_95ci": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
                               "cases": int(len(difference))}
        summary["models"][backbone]["paired_case_bootstrap_vs_source"] = paired
        summary["models"][backbone]["adaptation_diagnostics"] = variant_diagnostics
    pd.DataFrame(metric_rows).to_csv(root / f"{a.out_prefix}_metrics.csv", index=False)
    pd.DataFrame(case_rows).to_csv(root / f"{a.out_prefix}_case_metrics.csv", index=False)
    (root / f"{a.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
