#!/usr/bin/env python3
"""Aggregate deterministic input-shift tests for one outer VitalDB fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clean-metrics", help="clean-input metrics for the same fold/model/TTA settings")
    a = p.parse_args()
    clean = json.loads(Path(a.clean_metrics).read_text(encoding="utf-8")) if a.clean_metrics else None
    rows = []
    for path in sorted(Path(a.root).glob("*/metrics_test_fold*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        setting = report["settings"]
        rows.append({"run": path.parent.name, "shift": setting["shift"],
                     "severity": setting["severity"],
                     "source_mean_case_mae": report["source_mean_case_mae"],
                     "tta_mean_case_mae": report["tta_mean_case_mae"],
                     "source_delta_vs_clean": (report["source_mean_case_mae"] - clean["source_mean_case_mae"]
                                               if clean else None),
                     "tta_delta_vs_clean": (report["tta_mean_case_mae"] - clean["tta_mean_case_mae"]
                                            if clean else None),
                     "source_window_mae": report["source"]["mae"],
                     "tta_window_mae": report["episodic_qm_tta"]["mae"],
                     "delta_mean_case_mae_source_minus_tta": report["delta_mean_case_mae"],
                     "updated_samples": report["adaptation_diagnostics"]["updated_samples"],
                     "seen_samples": report["adaptation_diagnostics"]["seen_samples"]})
    if not rows:
        raise SystemExit("no shift test metrics found")
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
