#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from qmtta.vitaldb import audit_cases, save_audit


def main() -> None:
    p = argparse.ArgumentParser(description="Audit paired PPG/SpO2 tracks in existing VitalDB data")
    p.add_argument("--data-dir", default=os.environ.get("VITALDB_DIR", "data/raw/vitaldb"))
    p.add_argument("--out-dir", default="results/vitaldb_pilot")
    p.add_argument("--max-cases", type=int, default=24, help="0 scans all case files")
    p.add_argument("--window-s", type=int, default=10)
    p.add_argument("--stride-s", type=int, default=10)
    p.add_argument("--fs", type=int, default=125)
    p.add_argument("--protocol", choices=["papagei", "sigma"], default="papagei")
    p.add_argument("--min-spo2-points", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    result = audit_cases(a.data_dir, max_cases=a.max_cases, window_s=a.window_s,
                         stride_s=a.stride_s, fs=a.fs,
                         min_spo2_points=a.min_spo2_points, protocol=a.protocol)
    save_audit(result, a.out_dir)
    windows = pd.DataFrame(result.rows)
    cases = pd.DataFrame(result.case_rows)
    if not windows.empty:
        groups = windows["case_id"].to_numpy()
        n_splits = min(5, len(np.unique(groups)))
        folds = np.full(len(windows), -1, dtype=int)
        if n_splits >= 2:
            for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(windows, groups=groups)):
                folds[test_idx] = fold
            windows["fold"] = folds
            windows.to_csv(Path(a.out_dir) / "window_index.csv", index=False)
            fold_cases = {str(int(k)): sorted(map(int, set(windows.loc[windows.fold == k, "case_id"])))
                          for k in sorted(set(windows.fold))}
            (Path(a.out_dir) / "folds.json").write_text(json.dumps(fold_cases, indent=2), encoding="utf-8")
    summary = {
        "data_dir": str(Path(a.data_dir).resolve()),
        "max_cases": a.max_cases,
        "protocol": a.protocol,
        "cases_scanned": int(len(cases)),
        "cases_with_both_tracks": int((cases.status == "ok").sum()) if not cases.empty else 0,
        "cases_with_usable_windows": int((cases.get("window_count", pd.Series(dtype=int)).fillna(0) > 0).sum()) if not cases.empty else 0,
        "windows": int(len(windows)),
        "spo2_median_distribution": windows.spo2_median.describe().to_dict() if not windows.empty else {},
        "note": "VitalDB-only feasibility pilot; case-level folds; labels are monitor SpO2, not SaO2.",
    }
    Path(a.out_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    statuses = cases.status.value_counts().to_dict() if not cases.empty else {}
    window_count_by_case = int(summary["cases_with_usable_windows"])
    lines = [
        "# VitalDB feasibility data audit", "",
        f"- Data root: `{Path(a.data_dir).resolve()}`",
        f"- Deterministic evenly spaced sample: {a.max_cases} cases (`0` means all records)",
        f"- Cases read: {summary['cases_scanned']}; both tracks: {summary['cases_with_both_tracks']}; cases with usable windows: {window_count_by_case}",
        f"- Paired {a.window_s} s windows: {summary['windows']}",
        f"- Window protocol: {a.protocol}, {a.fs} Hz PPG, {a.window_s} s windows, {a.stride_s} s stride, ≥{a.min_spo2_points} valid 1 Hz SpO₂ points, valid range [70,100]",
        f"- Status counts: `{json.dumps(statuses, ensure_ascii=False)}`",
        "- Fold assignment is grouped by `case_id`; no window from one case crosses folds.",
        "- SpO₂ labels are monitor values and do not represent arterial blood-gas SaO₂.",
        "", "## SpO₂ window-label distribution", "",
        "```json", json.dumps(summary["spo2_median_distribution"], indent=2), "```", "",
        "The sample is highly concentrated near 100. This audit demonstrates usable track pairing, not model validity or clinical performance.",
    ]
    Path(a.out_dir, "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
