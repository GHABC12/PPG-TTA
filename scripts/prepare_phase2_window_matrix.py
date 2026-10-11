#!/usr/bin/env python3
"""Build fixed-fold VitalDB PaPaGei windows for a comparable 10/30/60s matrix."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import cheby1, sosfiltfilt

if importlib.util.find_spec("wfdb") is None:
    sys.modules["wfdb"] = types.ModuleType("wfdb")
import vitaldb

from qmtta.vitaldb import PPG_TRACK, SPO2_TRACK, _morphology, quality_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="VitalDB 1.0.0 directory containing vital_files/")
    ap.add_argument("--reference-index", required=True, help="Existing 10-second index; defines the fixed case-fold map")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-s", type=int, nargs="+", default=(10, 30, 60))
    ap.add_argument("--coverage-threshold", type=float, default=0.5)
    ap.add_argument("--fs", type=int, default=125)
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.out_dir)
    files, cache_root = root / "vital_files", root.parent / "qmtta_cache"
    out.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(args.reference_index)
    if ref.case_id.duplicated().all():
        raise SystemExit("reference index does not contain case rows")
    fold_counts = ref.groupby("case_id").fold.nunique()
    if (fold_counts != 1).any():
        raise SystemExit("reference index assigns a case to multiple folds")
    fold_map = {int(case): int(group.fold.iloc[0]) for case, group in ref.groupby("case_id", sort=False)}
    case_ids = sorted(fold_map)
    pd.DataFrame([{"case_id": c, "fold": fold_map[c]} for c in case_ids]).to_csv(out / "fixed_case_folds.csv", index=False)

    old_keys = set(zip(ref.case_id.astype(int), ref.start_s.astype(int)))
    sos = cheby1(4, 0.5, [0.5, 12.0], btype="bandpass", fs=args.fs, output="sos")
    windows = {int(w): [] for w in args.window_s}
    label_eligible = {int(w): set() for w in args.window_s}
    rejected = []
    sampling_rows = []
    for pos, case_id in enumerate(case_ids, 1):
        path = files / f"{case_id:04d}.vital"
        if not path.exists():
            raise FileNotFoundError(path)
        vf = vitaldb.VitalFile(str(path), track_names=[PPG_TRACK, SPO2_TRACK])
        ppg = np.asarray(vf.get_track_samples(PPG_TRACK, 1.0 / args.fs), dtype=np.float32).reshape(-1)
        spo2 = np.asarray(vf.get_track_samples(SPO2_TRACK, 1.0), dtype=np.float32).reshape(-1)
        sampling_rows.append({"case_id": case_id, "ppg_n": len(ppg), "spo2_n": len(spo2),
                              "ppg_duration_s": len(ppg) / args.fs, "spo2_duration_s": len(spo2),
                              "duration_difference_s": len(ppg) / args.fs - len(spo2)})
        available_end = min(len(spo2), len(ppg) // args.fs)
        for window_s in map(int, args.window_s):
            starts = range(0, available_end, window_s)
            min_valid = int(math.ceil(args.coverage_threshold * window_s))
            cache_x, cache_m, cache_v, cache_q, records = [], [], [], [], []
            for start in starts:
                end = start + window_s
                if end > len(spo2) or end * args.fs > len(ppg):
                    continue
                label = spo2[start:end]
                valid = np.isfinite(label) & (label >= 70) & (label <= 100)
                if int(valid.sum()) < min_valid:
                    rejected.append({"case_id": case_id, "start_s": start, "window_s": window_s,
                                     "old_indexed": (case_id, start) in old_keys, "reason": "spo2_coverage",
                                     "spo2_valid_count": int(valid.sum()), "min_valid": min_valid})
                    continue
                label_eligible[window_s].add((case_id, start))
                raw = ppg[start * args.fs:end * args.fs].astype(np.float64)
                finite_fraction = float(np.isfinite(raw).mean())
                if np.mean(~np.isfinite(raw)) > 0.01:
                    rejected.append({"case_id": case_id, "start_s": start, "window_s": window_s,
                                     "old_indexed": (case_id, start) in old_keys, "reason": "ppg_missing_over_1pct",
                                     "spo2_valid_count": int(valid.sum()), "min_valid": min_valid,
                                     "ppg_finite_fraction": finite_fraction})
                    continue
                raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
                raw_std = float(raw.std())
                if raw_std < 1e-8:
                    rejected.append({"case_id": case_id, "start_s": start, "window_s": window_s,
                                     "old_indexed": (case_id, start) in old_keys, "reason": "ppg_flat_before_filter",
                                     "spo2_valid_count": int(valid.sum()), "min_valid": min_valid,
                                     "ppg_finite_fraction": finite_fraction, "ppg_raw_std": raw_std})
                    continue
                try:
                    filtered = sosfiltfilt(sos, raw)
                except ValueError as exc:
                    rejected.append({"case_id": case_id, "start_s": start, "window_s": window_s,
                                     "old_indexed": (case_id, start) in old_keys, "reason": "sosfiltfilt_error",
                                     "spo2_valid_count": int(valid.sum()), "min_valid": min_valid,
                                     "ppg_finite_fraction": finite_fraction, "ppg_raw_std": raw_std,
                                     "detail": str(exc)})
                    continue
                filtered_std = float(filtered.std())
                if filtered_std < 1e-8:
                    rejected.append({"case_id": case_id, "start_s": start, "window_s": window_s,
                                     "old_indexed": (case_id, start) in old_keys, "reason": "ppg_flat_after_filter",
                                     "spo2_valid_count": int(valid.sum()), "min_valid": min_valid,
                                     "ppg_finite_fraction": finite_fraction, "ppg_raw_std": raw_std,
                                     "ppg_filtered_std": filtered_std})
                    continue
                x = ((filtered - filtered.mean()) / max(filtered_std, 1e-8)).astype(np.float32)
                morph, morph_valid = _morphology(x, args.fs)
                quality = quality_score(x, args.fs)
                window_idx = len(cache_x)
                cache_x.append(x); cache_m.append(morph); cache_v.append(morph_valid); cache_q.append(quality)
                records.append({"case_id": case_id, "start_s": start, "end_s": end, "protocol": "papagei",
                                "fs": args.fs, "window_idx": window_idx,
                                "spo2_median": float(np.median(label[valid])), "spo2_mean": float(np.mean(label[valid])),
                                "spo2_std": float(np.std(label[valid])), "spo2_valid_count": int(valid.sum()),
                                "spo2_valid_fraction": float(valid.mean()), "quality_score": quality,
                                "sVRI": float(morph[0]), "IPA": float(morph[1]), "SQI": float(morph[2]),
                                "IPA_valid": int(morph_valid[1]), "fold": fold_map[case_id]})
            cache_path = cache_root / f"phase2_case_{case_id:04d}_papagei_{window_s}s_{args.fs}hz_v1.npz"
            if cache_x:
                np.savez_compressed(cache_path, ppg=np.stack(cache_x), morphology=np.stack(cache_m),
                                    morphology_valid=np.stack(cache_v), quality=np.asarray(cache_q, dtype=np.float32),
                                    case_id=np.asarray(case_id, dtype=np.int32), window_s=np.asarray(window_s, dtype=np.int32),
                                    fs=np.asarray(args.fs, dtype=np.int32), protocol=np.asarray("papagei"))
                for record in records:
                    record["cache_path"] = str(cache_path)
                    windows[window_s].append(record)
        print(f"[{pos}/{len(case_ids)}] case {case_id}: loaded {len(ppg)} PPG and {len(spo2)} SpO2 samples", flush=True)

    for window_s, records in windows.items():
        frame = pd.DataFrame(records).sort_values(["fold", "case_id", "start_s"]).reset_index(drop=True)
        target = out / f"window_{window_s}s"
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / "window_index.csv", index=False)
        counts = frame.groupby("case_id").size().to_dict()
        (target / "summary.json").write_text(json.dumps({
            "window_s": window_s, "stride_s": window_s, "coverage_threshold": args.coverage_threshold,
            "min_valid_spo2_points": int(math.ceil(args.coverage_threshold * window_s)),
            "n_windows": int(len(frame)), "n_cases": int(frame.case_id.nunique()),
            "per_case_window_counts": {str(k): int(v) for k, v in counts.items()},
            "fixed_case_folds": str(out / "fixed_case_folds.csv"),
            "input_channel": PPG_TRACK, "preprocessing": "Chebyshev 0.5-12Hz bandpass, per-window z-score",
        }, indent=2), encoding="utf-8")

    old_10 = {(int(r.case_id), int(r.start_s)) for r in ref.itertuples()}
    new_10 = {(int(r["case_id"]), int(r["start_s"])) for r in windows.get(10, [])}
    # PPG-rejected windows with an otherwise valid SpO2 label are captured in rejected;
    # they explain coverage-only counts and are separately counted by reason.
    rejection_counts = pd.DataFrame(rejected).groupby(["window_s", "reason"]).size().to_dict() if rejected else {}
    reconciliation = {
        "old_index_windows": len(old_10), "new_10s_quality_filtered_windows": len(new_10),
        "10s_label_eligible_windows_before_ppg_qc": len(label_eligible.get(10, set())),
        "10s_label_eligible_but_rejected_by_ppg_qc": sorted(
            [{"case_id": c, "start_s": s} for c, s in label_eligible.get(10, set()) - new_10],
            key=lambda x: (x["case_id"], x["start_s"])),
        "old_not_recreated": sorted([{"case_id": c, "start_s": s} for c, s in old_10 - new_10], key=lambda x: (x["case_id"], x["start_s"])),
        "new_not_in_old": sorted([{"case_id": c, "start_s": s} for c, s in new_10 - old_10], key=lambda x: (x["case_id"], x["start_s"])),
        "rejections_by_window_s_and_reason": {f"{k[0]}s::{k[1]}": int(v) for k, v in rejection_counts.items()},
        "track_sampling_lengths": sampling_rows,
        "notes": "New 10s windows use the exact project PPG QC and 50% SpO2 rule. The old 10s rule min 5 is equivalent to 50%. Compare key differences before treating them as a cache-only discrepancy.",
    }
    (out / "window_reconciliation.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
    pd.DataFrame(rejected).to_csv(out / "window_rejected_reasons.csv", index=False)
    print(json.dumps({k: v for k, v in reconciliation.items() if k not in ("track_sampling_lengths",)}, indent=2))


if __name__ == "__main__":
    main()
