#!/usr/bin/env python3
"""Leakage-aware representation probes for the cached VitalDB PPG embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def case_mae(y: np.ndarray, pred: np.ndarray, cases: np.ndarray) -> float:
    d = pd.DataFrame({"case": cases, "abs_error": np.abs(y - pred)})
    return float(d.groupby("case", sort=False).abs_error.mean().mean())


def metrics(y: np.ndarray, pred: np.ndarray, cases: np.ndarray) -> dict:
    corr = pearsonr(y, pred).statistic if np.std(pred) else float("nan")
    rank = spearmanr(y, pred).statistic if np.std(pred) else float("nan")
    return {
        "n_windows": int(len(y)), "n_cases": int(len(np.unique(cases))),
        "window_mae": float(mean_absolute_error(y, pred)),
        "window_rmse": float(mean_squared_error(y, pred) ** 0.5),
        "window_r2": float(r2_score(y, pred)), "window_pearson_r": float(corr),
        "window_spearman_r": float(rank), "case_mean_mae": case_mae(y, pred, cases),
    }


def evaluate(x: np.ndarray, index: pd.DataFrame, model_name: str, max_per_case: int) -> list[dict]:
    y = index.spo2_median.to_numpy(dtype=float)
    cases = index.case_id.to_numpy()
    folds = sorted(index.fold.unique().astype(int).tolist())
    rows = []
    for test_fold in folds:
        val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
        train = ~index.fold.isin([test_fold, val_fold]).to_numpy()
        val = index.fold.eq(val_fold).to_numpy()
        test = index.fold.eq(test_fold).to_numpy()
        fit = np.flatnonzero(train)
        if model_name != "ridge" and max_per_case > 0:
            rng = np.random.default_rng(2026 + test_fold)
            fit = np.concatenate([rng.choice(g, max_per_case, replace=False) if len(g) > max_per_case else g
                                  for _, g in pd.Series(fit, index=cases[fit]).groupby(level=0)])
        scaler = StandardScaler().fit(x[fit])
        xt, xv, xe = scaler.transform(x[fit]), scaler.transform(x[val]), scaler.transform(x[test])
        if model_name == "ridge":
            candidates = []
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
                model = Ridge(alpha=alpha).fit(xt, y[fit])
                candidates.append((case_mae(y[val], model.predict(xv), cases[val]), alpha))
            _, best = min(candidates)
            model = Ridge(alpha=best).fit(xt, y[fit])
            params = {"alpha": best}
        elif model_name == "rf":
            model = RandomForestRegressor(n_estimators=80, max_features=0.3, min_samples_leaf=10,
                                          n_jobs=1, random_state=2026).fit(x[fit], y[fit])
            params = {"n_estimators": 80, "max_features": 0.3, "min_samples_leaf": 10, "n_jobs": 1}
        else:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.6, reg_lambda=10,
                                 objective="reg:squarederror", n_jobs=1, random_state=2026,
                                 verbosity=0).fit(x[fit], y[fit])
            params = {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.03,
                      "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 10, "n_jobs": 1}
        pred_x = xe if model_name == "ridge" else x[test]
        row = {"model": model_name, "test_fold": test_fold, "validation_fold": val_fold,
               "train_cases": int(np.unique(cases[fit]).size), "train_windows": int(len(fit)),
               "max_train_windows_per_case": max_per_case if model_name != "ridge" else None, "params": params,
               **metrics(y[test], model.predict(pred_x), cases[test])}
        rows.append(row)
    return rows


def plots(x: np.ndarray, index: pd.DataFrame, out: Path) -> dict:
    # Use one centroid per case so visualization does not treat adjacent windows as independent.
    cases = index.case_id.to_numpy()
    labels = index.spo2_median.to_numpy(dtype=float)
    frame = pd.DataFrame(x)
    frame["case_id"] = cases
    means = frame.groupby("case_id", sort=True).mean().to_numpy()
    case_ids = np.sort(np.unique(cases))
    case_y = np.array([np.median(labels[cases == c]) for c in case_ids])
    pca = PCA(n_components=2, random_state=2026)
    p = pca.fit_transform(means)
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = ax.scatter(p[:, 0], p[:, 1], c=case_y, cmap="viridis", s=48, edgecolor="black", linewidth=0.3)
    for i, c in enumerate(case_ids):
        ax.annotate(str(c), (p[i, 0], p[i, 1]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set(title="PaPaGei case-centroid PCA (color = case median SpO2)", xlabel="PC1", ylabel="PC2")
    fig.colorbar(pts, ax=ax, label="SpO2 (%)")
    fig.tight_layout()
    fig.savefig(out / "case_centroid_pca.png", dpi=180)
    plt.close(fig)
    tsne = TSNE(n_components=2, perplexity=min(7, max(2, (len(case_ids) - 1) // 3)),
                init="pca", learning_rate="auto", random_state=2026)
    t = tsne.fit_transform(means)
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = ax.scatter(t[:, 0], t[:, 1], c=case_y, cmap="viridis", s=48, edgecolor="black", linewidth=0.3)
    for i, c in enumerate(case_ids):
        ax.annotate(str(c), (t[i, 0], t[i, 1]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set(title="PaPaGei case-centroid t-SNE (color = case median SpO2)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    fig.colorbar(pts, ax=ax, label="SpO2 (%)")
    fig.tight_layout()
    fig.savefig(out / "case_centroid_tsne.png", dpi=180)
    plt.close(fig)
    rho = spearmanr(p[:, 0], case_y).statistic
    return {"n_case_centroids": int(len(case_ids)), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pc1_spo2_spearman_r": float(rho)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", choices=("ridge", "rf", "xgb"), default=("ridge", "xgb"))
    ap.add_argument("--max-train-windows-per-case", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    pack = np.load(args.embeddings)
    x = pack["embeddings"]
    if len(index) != len(x) or not np.array_equal(index.case_id.to_numpy(), pack["case_id"]):
        raise SystemExit("window index and embedding cache do not align")
    if "start_s" in pack and not np.array_equal(index.start_s.to_numpy(), pack["start_s"]):
        raise SystemExit("window start times and embedding cache do not align")
    rows = []
    for model in args.models:
        print(f"Fitting {model} probes", flush=True)
        rows.extend(evaluate(x, index, model, args.max_train_windows_per_case))
        pd.DataFrame(rows).to_csv(out / "probe_fold_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "probe_fold_metrics.csv", index=False)
    summary = {
        "dataset": "existing VitalDB only", "representation": "cached frozen PaPaGei embeddings",
        "split": "five case-grouped folds; one fold held out, next fold used for Ridge alpha selection",
        "n_cases": int(index.case_id.nunique()), "n_windows": int(len(index)), "embedding_dim": int(x.shape[1]),
        "label": "10-second window median Solar SpO2, already filtered to valid 70-100 values",
        "case_mean_mae_definition": "mean of per-case mean absolute window errors (matches existing baseline case-MAE aggregation)",
        "models": {}, "visualization": plots(x, index, out),
        "nonlinear_train_sampling": f"up to {args.max_train_windows_per_case} windows per training case; full held-out folds used for evaluation",
    }
    table = pd.DataFrame(rows)
    for name, g in table.groupby("model"):
        summary["models"][name] = {c: float(g[c].mean()) for c in
                                   ("window_mae", "window_rmse", "window_r2", "window_pearson_r", "window_spearman_r", "case_mean_mae")}
    (out / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
