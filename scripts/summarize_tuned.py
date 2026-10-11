#!/usr/bin/env python3
"""Aggregate validation-selected, outer-fold test-only TTA results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    err = prediction - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "prediction_std": float(np.std(prediction))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--backbones", default="papagei,sigma")
    p.add_argument("--model-kinds", default="lp,finetune")
    p.add_argument("--out-prefix", default="tuned_oof")
    a = p.parse_args()
    root = Path(a.results_root)
    fold_rows, case_rows, summaries = [], [], {}
    for backbone in a.backbones.split(","):
        dataset = "vitaldb_pilot" if backbone == "papagei" else "vitaldb_sigma"
        for kind in a.model_kinds.split(","):
            base = root / dataset
            tuning_root = base / "tta_tuning" / kind
            predictions = []
            selections = []
            for fold in range(a.folds):
                selected_path = tuning_root / f"selected_{backbone}_{kind}_fold{fold}.json"
                if not selected_path.is_file():
                    raise SystemExit(f"missing selected validation config: {selected_path}")
                selection = json.loads(selected_path.read_text(encoding="utf-8"))
                if "test_result_file" not in selection:
                    raise SystemExit(f"missing final test evaluation for {selected_path}")
                report_path = Path(selection["test_result_file"])
                if not report_path.is_absolute() and not report_path.is_file():
                    report_path = root / report_path
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("role") != "test" or report.get("evaluation_fold") != fold:
                    raise SystemExit(f"test-role/fold mismatch in {report_path}")
                prediction_path = report_path.parent / f"predictions_test_fold{fold}_{selection['selected']['name']}.csv"
                pred = pd.read_csv(prediction_path)
                pred["fold"] = fold
                pred["backbone"] = backbone
                pred["model_kind"] = kind
                # The baseline prediction is the constant train median. Recover it from
                # the report's signed bias (mean(pred - y)) and observed fold mean.
                pred["train_median_prediction"] = (
                    pred.spo2_true.mean() + report["train_median_constant_baseline"]["bias"])
                pred["selection_name"] = selection["selected"]["name"]
                predictions.append(pred)
                selections.append({"fold": fold, "selected": selection["selected"],
                                   "validation_fold": selection["validation_fold"],
                                   "validation_reused_for_source_early_stopping":
                                       selection["validation_reused_for_source_early_stopping"]})
                for method, column in [("source_only", "source_prediction"),
                                       ("validation_selected_tta", "tta_prediction"),
                                       ("train_median_constant", "train_median_prediction")]:
                    fold_rows.append({"backbone": backbone, "model_kind": kind, "fold": fold,
                                      "method": method, **score(pred.spo2_true.to_numpy(), pred[column].to_numpy()),
                                      "mean_case_mae": float(pred.assign(ae=np.abs(pred[column]-pred.spo2_true))
                                                              .groupby("case_id").ae.mean().mean())})

            oof = pd.concat(predictions, ignore_index=True)
            for method, column in [("source_only", "source_prediction"),
                                   ("validation_selected_tta", "tta_prediction"),
                                   ("train_median_constant", "train_median_prediction")]:
                fold_rows.append({"backbone": backbone, "model_kind": kind, "fold": "OOF",
                                  "method": method, **score(oof.spo2_true.to_numpy(), oof[column].to_numpy()),
                                  "mean_case_mae": float(oof.assign(ae=np.abs(oof[column]-oof.spo2_true))
                                                          .groupby("case_id").ae.mean().mean())})
                for case_id, case in oof.groupby("case_id"):
                    case_rows.append({"backbone": backbone, "model_kind": kind,
                                      "case_id": int(case_id), "fold": int(case.fold.iloc[0]),
                                      "method": method, **score(case.spo2_true.to_numpy(),
                                                                 case[column].to_numpy())})
            summaries[f"{backbone}_{kind}"] = {
                "selections": selections,
                "oof_predictions": len(oof),
                "oof_mean_case_mae": {
                    method: float(oof.assign(ae=np.abs(oof[column]-oof.spo2_true))
                                  .groupby("case_id").ae.mean().mean())
                    for method, column in [("source_only", "source_prediction"),
                                           ("validation_selected_tta", "tta_prediction"),
                                           ("train_median_constant", "train_median_prediction")]}}

    pd.DataFrame(fold_rows).to_csv(root / f"{a.out_prefix}_metrics.csv", index=False)
    pd.DataFrame(case_rows).to_csv(root / f"{a.out_prefix}_case_metrics.csv", index=False)
    result = {"folds": a.folds, "case_grouped": True,
              "test_labels_used_for_selection": False,
              "prediction_protocol": "prequential; score before current-batch update",
              "validation_selection_note": "each outer fold selects only on its own validation fold; that fold also early-stops the source model",
              "models": summaries}
    (root / f"{a.out_prefix}_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
