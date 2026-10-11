#!/usr/bin/env python3
"""Compare VitalDB SpO2 label coverage across 10/30/60-second windows."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

if importlib.util.find_spec("wfdb") is None:
    sys.modules["wfdb"] = types.ModuleType("wfdb")
import vitaldb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vital-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--windows", type=int, nargs="+", default=(10, 30, 60))
    ap.add_argument("--coverage-threshold", type=float, default=0.5)
    args = ap.parse_args()
    files = Path(args.vital_root) / "vital_files"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    cases = index[["case_id", "fold"]].drop_duplicates("case_id").sort_values("case_id")
    rows = []
    for cr in cases.itertuples(index=False):
        case_id, fold = int(cr.case_id), int(cr.fold)
        path = files / f"{case_id:04d}.vital"
        if not path.exists():
            rows.append({"case_id": case_id, "fold": fold, "file_exists": False})
            continue
        tracks = ["SNUADC/PLETH", "Solar8000/PLETH_SPO2"]
        vf = vitaldb.VitalFile(str(path), track_names=tracks, interval=args.interval)
        data = np.asarray(vf.to_numpy(tracks, args.interval, return_timestamp=True), dtype=float)
        if data.ndim != 2 or data.shape[1] != 3:
            raise RuntimeError(f"unexpected timestamped matrix shape {data.shape} for case {case_id}")
        ppg_ok = np.isfinite(data[:, 1])
        spo2 = data[:, 2]
        valid = np.isfinite(spo2) & (spo2 >= 70) & (spo2 <= 100)
        for win_s in args.windows:
            n = int(round(win_s / args.interval))
            n_windows = len(valid) // n
            if n_windows == 0:
                continue
            valid_w = valid[:n_windows * n].reshape(n_windows, n)
            ppg_w = ppg_ok[:n_windows * n].reshape(n_windows, n)
            counts = valid_w.sum(axis=1)
            coverage = counts / n
            fixed_coverage_ok = coverage >= args.coverage_threshold
            legacy_ok = counts >= 5
            labels = np.full(n_windows, np.nan, dtype=float)
            for j in np.flatnonzero(fixed_coverage_ok):
                labels[j] = np.median(spo2[j*n:(j+1)*n][valid_w[j]])
            valid_labels = labels[np.isfinite(labels)]
            rows.append({
                "case_id": case_id, "fold": fold, "window_s": win_s,
                "file_exists": True, "n_total_windows": int(n_windows),
                "n_valid_legacy_min5": int(legacy_ok.sum()),
                "n_valid_fixed_coverage": int(fixed_coverage_ok.sum()),
                "fixed_coverage_threshold": args.coverage_threshold,
                "median_valid_spo2_points": float(np.median(counts)),
                "median_valid_spo2_fraction": float(np.median(coverage)),
                "ppg_finite_fraction": float(ppg_w.mean()),
                "valid_label_median": float(np.median(valid_labels)) if len(valid_labels) else None,
                "valid_label_iqr": float(np.subtract(*np.percentile(valid_labels, [75, 25]))) if len(valid_labels) else None,
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "window_coverage_by_case.csv", index=False)
    valid_detail = detail[detail.file_exists.eq(True)]
    summary = {
        "data_source": "pre-existing VitalDB only",
        "cases": int(cases.case_id.nunique()), "interval_s": args.interval,
        "label": "median Solar8000/PLETH_SPO2 among finite values in [70,100]",
        "legacy_rule": "at least 5 valid SpO2 samples per window",
        "comparable_rule": f"at least {args.coverage_threshold:.0%} valid SpO2 samples per window",
        "window_summaries": {},
        "alignment_note": "PPG and SpO2 are sampled from the same VitalFile common start/end grid; timestamps were audited separately in timestamp_alignment_sample.csv.",
    }
    for win_s, group in valid_detail.groupby("window_s", sort=True):
        den = int(group.n_total_windows.sum())
        summary["window_summaries"][str(int(win_s))] = {
            "total_windows": den,
            "legacy_valid_windows": int(group.n_valid_legacy_min5.sum()),
            "legacy_valid_fraction": float(group.n_valid_legacy_min5.sum() / den),
            "fixed_coverage_valid_windows": int(group.n_valid_fixed_coverage.sum()),
            "fixed_coverage_valid_fraction": float(group.n_valid_fixed_coverage.sum() / den),
            "median_case_valid_fraction_fixed_coverage": float((group.n_valid_fixed_coverage / group.n_total_windows).median()),
        }
    (out / "window_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
