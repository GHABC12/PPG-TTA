from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import cheby1, sosfiltfilt

PPG_TRACK = "SNUADC/PLETH"
SPO2_TRACK = "Solar8000/PLETH_SPO2"


@dataclass
class AuditResult:
    rows: list[dict]
    case_rows: list[dict]


def _morphology(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [sVRI, IPA, skew-SQI] and validity mask; exploratory only."""
    valid = np.isfinite(x)
    if valid.mean() < 0.99 or np.nanstd(x) < 1e-8:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    scale = np.ptp(x)
    if scale <= 1e-8:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    z = (x - x.min()) / scale
    peak = int(np.argmax(z))
    left = float(np.mean(z[:max(peak, 1)]))
    right = float(np.mean(z[peak:]))
    svri_ok = left > 1e-6
    svri = right / left if svri_ok else 0.0

    # First two local minima give a conservative single-beat interval.
    from scipy.signal import argrelmin
    minima = argrelmin(z, order=max(1, fs // 5))[0]
    ipa, ipa_ok = 0.0, False
    if len(minima) >= 2:
        beat = z[minima[0]:minima[1]]
        inner = argrelmin(beat, order=max(1, fs // 10))[0]
        if len(inner) and inner[0] > 0 and inner[0] < len(beat) - 1:
            notch = int(inner[0])
            # np.trapz is available in the server's NumPy 1.26 environment.
            dia = float(np.trapz(beat[notch:]))
            sys = float(np.trapz(beat[:notch]))
            if abs(dia) > 1e-6:
                ipa, ipa_ok = sys / dia, True

    # A skewness-derived SQI is a signal descriptor, not clinical quality truth.
    from scipy.stats import skew
    sqi = float(skew(x, bias=True))
    vals = np.asarray([svri, ipa, sqi], dtype=np.float32)
    mask = np.asarray([svri_ok, ipa_ok, np.isfinite(sqi)], dtype=np.float32)
    vals[~np.isfinite(vals)] = 0.0
    return vals, mask


def quality_score(x: np.ndarray, fs: int = 125) -> float:
    """Unsupervised PPG quality heuristic in [0, 1]; fixed for this pilot."""
    if x.size == 0 or np.mean(~np.isfinite(x)) > 0.01:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) <= 1e-8 or np.std(x) <= 1e-8:
        return 0.0
    usable = x[np.isfinite(x)]
    patches = usable[: (len(usable) // fs) * fs].reshape(-1, fs)
    if not len(patches):
        return 0.0
    stds = np.std(patches, axis=1)
    med = np.median(stds)
    mad = np.median(np.abs(stds - med)) + 1e-8
    amp = np.exp(-np.maximum(0.0, np.abs(stds - med) - 3 * mad) / (med + mad))
    from scipy.stats import skew
    sk = np.abs(skew(patches, axis=1, bias=True, nan_policy="omit"))
    skew_score = np.tanh(np.nan_to_num(sk, nan=0.0, posinf=0.0) / 2.0)
    return float(np.clip(np.percentile(0.5 * amp + 0.5 * skew_score, 25), 0.0, 1.0))


def _load_track(vf, name: str, interval: float) -> np.ndarray | None:
    try:
        values = vf.get_track_samples(name, interval)
    except (KeyError, ValueError, TypeError):
        return None
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr if arr.size else None


def audit_cases(
    data_dir: str | Path,
    *,
    max_cases: int = 24,
    window_s: int = 10,
    stride_s: int = 10,
    fs: int = 125,
    min_spo2_points: int = 5,
    protocol: str = "papagei",
    case_ids: Iterable[int] | None = None,
) -> AuditResult:
    """Read paired VitalDB cases and produce cached, non-overlapping windows.

    The PPG and numeric SpO2 tracks are sampled independently from the record's
    common zero time. Window labels use valid 1 Hz SpO2 values in [70, 100].
    """
    try:
        import vitaldb
    except ImportError as exc:
        raise RuntimeError("Install vitaldb in the selected Python environment") from exc

    root = Path(data_dir)
    files = sorted((root / "vital_files").glob("*.vital"), key=lambda p: int(p.stem))
    if case_ids is not None:
        wanted = {str(int(c)) for c in case_ids}
        files = [p for p in files if p.stem in wanted]
    if max_cases > 0 and len(files) > max_cases:
        # Evenly spread the deterministic pilot over case IDs instead of using
        # only the earliest records in the archive.
        selected = np.linspace(0, len(files) - 1, max_cases, dtype=int)
        files = [files[i] for i in selected]

    rows: list[dict] = []
    case_rows: list[dict] = []
    if protocol == "papagei":
        sos = cheby1(4, 0.5, [0.5, 12.0], btype="bandpass", fs=fs, output="sos")
    elif protocol == "sigma":
        from scipy.signal import butter
        sos = butter(2, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    else:
        raise ValueError(f"unsupported preprocessing protocol: {protocol}")
    for pos, path in enumerate(files, 1):
        rows_before_case = len(rows)
        record = {"case_id": int(path.stem), "path": str(path), "status": "error"}
        try:
            vf = vitaldb.VitalFile(str(path), track_names=[PPG_TRACK, SPO2_TRACK])
            ppg = _load_track(vf, PPG_TRACK, 1.0 / fs)
            spo2 = _load_track(vf, SPO2_TRACK, 1.0)
            if ppg is None or spo2 is None:
                record["status"] = "missing_track"
                case_rows.append(record)
                continue
            record.update({"status": "ok", "ppg_samples": int(len(ppg)),
                           "spo2_samples": int(len(spo2)),
                           "ppg_duration_s": float(len(ppg) / fs),
                           "spo2_valid_count": int(np.isfinite(spo2).sum())})
            cache_x: list[np.ndarray] = []
            cache_m: list[np.ndarray] = []
            cache_v: list[np.ndarray] = []
            cache_q: list[float] = []
            starts = range(0, min(len(spo2), len(ppg) // fs), stride_s)
            for start in starts:
                end = start + window_s
                if end > len(spo2) or end * fs > len(ppg):
                    continue
                labels = spo2[start:end]
                valid = np.isfinite(labels) & (labels >= 70) & (labels <= 100)
                if int(valid.sum()) < min_spo2_points:
                    continue
                x = ppg[start * fs:end * fs].astype(np.float64)
                if np.mean(~np.isfinite(x)) > 0.01:
                    continue
                x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
                if np.std(x) < 1e-8:
                    continue
                try:
                    x = sosfiltfilt(sos, x)
                except ValueError:
                    continue
                if protocol == "papagei":
                    x = (x - x.mean()) / max(float(x.std()), 1e-8)
                else:
                    x_min, x_max = float(x.min()), float(x.max())
                    x = 2.0 * (x - x_min) / max(x_max - x_min, 1e-8) - 1.0
                morph, morph_valid = _morphology(x, fs)
                q = quality_score(x, fs)
                idx = len(cache_x)
                cache_x.append(x.astype(np.float32))
                cache_m.append(morph)
                cache_v.append(morph_valid)
                cache_q.append(q)
                rows.append({
                    "case_id": int(path.stem), "start_s": int(start), "end_s": int(end),
                    "protocol": protocol,
                    "fs": fs, "window_idx": idx,
                    "spo2_median": float(np.median(labels[valid])),
                    "spo2_mean": float(np.mean(labels[valid])),
                    "spo2_std": float(np.std(labels[valid])),
                    "spo2_valid_count": int(valid.sum()), "quality_score": q,
                    "sVRI": float(morph[0]), "IPA": float(morph[1]),
                    "SQI": float(morph[2]), "IPA_valid": int(morph_valid[1]),
                    "cache_path": "",
                })
            record["window_count"] = len(cache_x)
            if cache_x:
                cache_path = root.parent / "qmtta_cache" / f"case_{path.stem}_{protocol}_{window_s}s_{fs}hz.npz"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path,
                    ppg=np.stack(cache_x), morphology=np.stack(cache_m),
                    morphology_valid=np.stack(cache_v), quality=np.asarray(cache_q, dtype=np.float32),
                    case_id=np.asarray(int(path.stem), dtype=np.int32),
                    window_s=np.asarray(window_s, dtype=np.int32), fs=np.asarray(fs, dtype=np.int32),
                    protocol=np.asarray(protocol))
                for row in rows[-len(cache_x):]:
                    row["cache_path"] = str(cache_path)
            case_rows.append(record)
            print(f"[{pos}/{len(files)}] case {path.stem}: {record['status']}, windows={record.get('window_count', 0)}")
        except Exception as exc:
            # A failed case must not leave partial windows in the index.
            del rows[rows_before_case:]
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            case_rows.append(record)
            print(f"[{pos}/{len(files)}] case {path.stem}: ERROR {type(exc).__name__}: {exc}")

    return AuditResult(rows, case_rows)


def save_audit(result: AuditResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.rows).to_csv(out / "window_index.csv", index=False)
    pd.DataFrame(result.case_rows).to_csv(out / "case_audit.csv", index=False)
