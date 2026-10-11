#!/usr/bin/env python3
"""Diagnose frozen VitalDB encoder embeddings within each training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(x: np.ndarray) -> dict:
    std = x.std(axis=0, ddof=0)
    scale = max(float(np.median(std)), 1e-12)
    threshold = max(scale * 1e-6, 1e-8)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(x), 1)
    eig = np.linalg.eigvalsh(cov).clip(min=0)
    mass = eig / max(float(eig.sum()), 1e-12)
    effective_rank = float(np.exp(-(mass[mass > 0] * np.log(mass[mass > 0])).sum()))
    return {
        "n_windows": int(len(x)), "n_features": int(x.shape[1]),
        "feature_std_mean": float(std.mean()), "feature_std_median": float(np.median(std)),
        "feature_std_p05": float(np.percentile(std, 5)),
        "feature_std_min": float(std.min()), "near_zero_std_threshold": threshold,
        "near_zero_feature_fraction": float(np.mean(std <= threshold)),
        "covariance_effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / x.shape[1],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    index = pd.read_csv(a.index)
    pack = np.load(a.embeddings)
    x = pack["embeddings"]
    if len(index) != len(x):
        raise SystemExit(f"index/embedding length mismatch: {len(index)} != {len(x)}")
    if "case_id" in pack and not np.array_equal(index.case_id.to_numpy(), pack["case_id"]):
        raise SystemExit("embedding case_id order does not match the window index")
    if "start_s" in pack and not np.array_equal(index.start_s.to_numpy(), pack["start_s"]):
        raise SystemExit("embedding start_s order does not match the window index")
    folds = sorted(index.fold.unique().astype(int).tolist())
    rows = []
    for test_fold in folds:
        val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
        mask = ~index.fold.isin([test_fold, val_fold]).to_numpy()
        row = {"test_fold": test_fold, "validation_fold": val_fold,
               "train_case_count": int(index.loc[mask, "case_id"].nunique()),
               **summarize(x[mask])}
        rows.append(row)
    result = {"diagnostic": "frozen PaPaGei VitalDB embeddings, training folds only",
              "folds": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
