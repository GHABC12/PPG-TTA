#!/usr/bin/env python3
"""Evaluate frozen PaPaGei embeddings on candidate VitalDB HR and respiration labels."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if importlib.util.find_spec("wfdb") is None:
    sys.modules["wfdb"] = types.ModuleType("wfdb")
import vitaldb

TARGETS = {
    "hr": ("Solar8000/HR", 30.0, 220.0),
    "resp_co2": ("Solar8000/RR_CO2", 4.0, 60.0),
}


def mean_case_mae(y: np.ndarray, pred: np.ndarray, cases: np.ndarray) -> float:
    d = pd.DataFrame({"case": cases, "ae": np.abs(y - pred)})
    return float(d.groupby("case", sort=False).ae.mean().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vital-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-s", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--coverage-threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    pack = np.load(args.embeddings)
    x = pack["embeddings"]
    if len(index) != len(x) or not np.array_equal(index.case_id.to_numpy(), pack["case_id"]):
        raise SystemExit("window index and embedding cache do not align")
    if not np.array_equal(index.start_s.to_numpy(), pack["start_s"]):
        raise SystemExit("window start times and embedding cache do not align")

    target_rows = []
    files = Path(args.vital_root) / "vital_files"
    for case_id in sorted(index.case_id.unique().astype(int)):
        path = files / f"{case_id:04d}.vital"
        vf = vitaldb.VitalFile(str(path), track_names=[v[0] for v in TARGETS.values()], interval=args.interval)
        tracks = [v[0] for v in TARGETS.values()]
        data = np.asarray(vf.to_numpy(tracks, args.interval), dtype=float)
        if data.ndim != 2 or data.shape[1] != len(tracks):
            raise RuntimeError(f"unexpected target matrix shape {data.shape} for case {case_id}")
        rows = index.case_id.to_numpy() == case_id
        for target_i, (name, (_, lo, hi)) in enumerate(TARGETS.items()):
            values = data[:, target_i]
            for row_i in np.flatnonzero(rows):
                start = int(round(index.iloc[row_i].start_s / args.interval))
                end = int(round(index.iloc[row_i].end_s / args.interval))
                block = values[start:end]
                valid = np.isfinite(block) & (block >= lo) & (block <= hi)
                coverage = float(valid.mean()) if len(block) else 0.0
                label = float(np.median(block[valid])) if coverage >= args.coverage_threshold and valid.any() else np.nan
                target_rows.append({"row_i": int(row_i), "case_id": case_id, "target": name,
                                    "label": label, "valid_fraction": coverage})
    target_frame = pd.DataFrame(target_rows)
    target_frame.to_csv(out / "transfer_window_labels.csv", index=False)

    all_metrics = []
    folds = sorted(index.fold.unique().astype(int).tolist())
    for target in TARGETS:
        targ = target_frame[target_frame.target.eq(target)].set_index("row_i").reindex(np.arange(len(index)))
        y = targ.label.to_numpy(dtype=float)
        cases = index.case_id.to_numpy()
        rows = []
        for test_fold in folds:
            val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
            train = (~index.fold.isin([test_fold, val_fold]).to_numpy()) & np.isfinite(y)
            val = index.fold.eq(val_fold).to_numpy() & np.isfinite(y)
            test = index.fold.eq(test_fold).to_numpy() & np.isfinite(y)
            scaler = StandardScaler().fit(x[train])
            xt, xv, xe = scaler.transform(x[train]), scaler.transform(x[val]), scaler.transform(x[test])
            candidates = []
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
                model = Ridge(alpha=alpha).fit(xt, y[train])
                candidates.append((mean_case_mae(y[val], model.predict(xv), cases[val]), alpha))
            _, alpha = min(candidates)
            model = Ridge(alpha=alpha).fit(xt, y[train])
            pred = model.predict(xe)
            baseline = np.full(test.sum(), float(np.median(y[train])))
            row = {"target": target, "test_fold": test_fold, "validation_fold": val_fold,
                   "train_windows": int(train.sum()), "test_windows": int(test.sum()),
                   "train_cases": int(index.loc[train, "case_id"].nunique()),
                   "test_cases": int(index.loc[test, "case_id"].nunique()), "ridge_alpha": alpha,
                   "ridge_window_mae": float(mean_absolute_error(y[test], pred)),
                   "ridge_window_rmse": float(mean_squared_error(y[test], pred) ** 0.5),
                   "ridge_window_r2": float(r2_score(y[test], pred)),
                   "ridge_case_mean_mae": mean_case_mae(y[test], pred, cases[test]),
                   "median_window_mae": float(mean_absolute_error(y[test], baseline)),
                   "median_case_mean_mae": mean_case_mae(y[test], baseline, cases[test])}
            rows.append(row)
            all_metrics.append(row)
        pd.DataFrame(rows).to_csv(out / f"{target}_fold_metrics.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(out / "transfer_probe_fold_metrics.csv", index=False)

    summary = {"data_source": "pre-existing VitalDB only", "representation": "cached frozen PaPaGei embeddings",
               "window_s": args.window_s, "coverage_threshold": args.coverage_threshold,
               "folds": "5 grouped case folds; next fold used for Ridge alpha selection",
               "label_rules": {"hr": "Solar8000/HR median within [30,220] bpm", "resp_co2": "Solar8000/RR_CO2 median within [4,60] breaths/min"},
               "targets": {}}
    for target in TARGETS:
        target_rows = target_frame[target_frame.target.eq(target)]
        labeled = target_rows[np.isfinite(target_rows.label)]
        g = pd.DataFrame([r for r in all_metrics if r["target"] == target])
        summary["targets"][target] = {
            "labeled_windows": int(len(labeled)), "all_windows": int(len(target_rows)),
            "label_coverage": float(len(labeled) / max(len(target_rows), 1)),
            "labeled_cases": int(labeled.case_id.nunique()),
            "ridge_case_mean_mae": float(g.ridge_case_mean_mae.mean()),
            "median_baseline_case_mean_mae": float(g.median_case_mean_mae.mean()),
            "ridge_window_mae": float(g.ridge_window_mae.mean()),
            "median_baseline_window_mae": float(g.median_window_mae.mean()),
            "ridge_window_r2": float(g.ridge_window_r2.mean()),
            "caveat": "Single VitalDB source; monitor-derived labels are not independent clinical references.",
        }
    (out / "transfer_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
