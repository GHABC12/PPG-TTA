#!/usr/bin/env python3
"""Audit candidate VitalDB transfer labels and verify common timestamp alignment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

# VitalDB's offline .vital reader does not use wfdb, but its package imports it
# unconditionally for unrelated ECG utilities. Permit offline parsing when only
# wfdb is absent; if wfdb exists, use the real module.
if importlib.util.find_spec("wfdb") is None:
    sys.modules["wfdb"] = types.ModuleType("wfdb")
import vitaldb

TRACKS = {
    "ppg": "SNUADC/PLETH",
    "spo2": "Solar8000/PLETH_SPO2",
    "hr": "Solar8000/HR",
    "pleth_hr": "Solar8000/PLETH_HR",
    "resp_vent": "Solar8000/VENT_RR",
    "resp_co2": "Solar8000/RR_CO2",
    "resp_primus": "Primus/RR_CO2",
    "bis_sqi": "BIS/SQI",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vital-root", required=True, help="VitalDB directory containing vital_files/")
    ap.add_argument("--index", required=True, help="Existing 10-second window index")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    root = Path(args.vital_root)
    files = root / "vital_files"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    cases = index[["case_id", "fold"]].drop_duplicates("case_id").sort_values("case_id")

    availability = []
    for row in cases.itertuples(index=False):
        path = files / f"{int(row.case_id):04d}.vital"
        if not path.exists():
            availability.append({"case_id": int(row.case_id), "fold": int(row.fold), "file_exists": False})
            continue
        vf = vitaldb.VitalFile(str(path), header_only=True)
        present = set(vf.get_track_names())
        rec = {"case_id": int(row.case_id), "fold": int(row.fold), "file_exists": True}
        rec.update({name: track in present for name, track in TRACKS.items()})
        availability.append(rec)
    av = pd.DataFrame(availability)
    av.to_csv(out / "track_availability_by_case.csv", index=False)

    # One representative case per existing grouped fold, selected by lowest ID.
    sample_cases = cases.groupby("fold", sort=True).case_id.min().astype(int).tolist()
    time_rows = []
    timed_tracks = [TRACKS[k] for k in ("ppg", "spo2", "hr", "resp_vent", "resp_co2", "resp_primus", "bis_sqi")]
    for case_id in sample_cases:
        path = files / f"{case_id:04d}.vital"
        vf = vitaldb.VitalFile(str(path), track_names=timed_tracks, interval=args.interval)
        values = np.asarray(vf.to_numpy(timed_tracks, args.interval, return_timestamp=True), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(timed_tracks) + 1:
            raise RuntimeError(f"unexpected timestamped matrix shape {values.shape} for case {case_id}")
        ts = values[:, 0]
        dt = np.diff(ts)
        rec = {"case_id": case_id, "interval_s": args.interval, "samples": len(ts),
               "duration_h": float((ts[-1] - ts[0]) / 3600) if len(ts) else 0.0,
               "timestamp_monotonic": bool(np.all(dt > 0)),
               "timestamp_step_max_abs_error_s": float(np.max(np.abs(dt - args.interval))) if len(dt) else 0.0}
        finite = np.isfinite(values[:, 1:])
        for i, track in enumerate(timed_tracks):
            rec[f"finite_fraction::{track}"] = float(finite[:, i].mean())
        ppg = finite[:, timed_tracks.index(TRACKS["ppg"])]
        spo2 = finite[:, timed_tracks.index(TRACKS["spo2"])]
        rec["ppg_spo2_simultaneous_finite_fraction"] = float((ppg & spo2).mean())
        time_rows.append(rec)
    pd.DataFrame(time_rows).to_csv(out / "timestamp_alignment_sample.csv", index=False)

    candidate_cols = ["ppg", "spo2", "hr", "pleth_hr", "resp_vent", "resp_co2", "resp_primus", "bis_sqi"]
    summary = {
        "data_source": "pre-existing VitalDB only",
        "indexed_cases": int(len(cases)), "existing_window_count": int(len(index)),
        "track_availability_counts": {name: int(av[name].fillna(False).sum()) for name in candidate_cols if name in av},
        "file_missing_cases": av.loc[~av.file_exists, "case_id"].astype(int).tolist() if "file_exists" in av else [],
        "timestamp_sample_cases_one_per_fold": sample_cases,
        "timestamp_interval_s": args.interval,
        "sqi_caveat": "BIS/SQI is a BIS/EEG quality signal, not a PPG signal-quality ground truth.",
        "respiratory_caveat": "Respiratory tracks are candidates; coverage and device/measurement semantics must be checked before treating them as validated targets.",
    }
    (out / "track_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
