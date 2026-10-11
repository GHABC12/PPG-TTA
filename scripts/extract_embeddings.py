#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qmtta.papagei import PaPaGeiEncoder
from qmtta.sigma import SigmaPPGEncoder


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frozen PaPaGei-S embeddings from audited windows")
    p.add_argument("--index", default="results/vitaldb_pilot/window_index.csv")
    p.add_argument("--out", default="results/vitaldb_pilot/embeddings.npz")
    p.add_argument("--repo", default=os.environ.get("PAPAGEI_REPO", "third_party/papagei-foundation-model"))
    p.add_argument("--checkpoint", default=os.environ.get("PAPAGEI_CHECKPOINT", "weights/papagei_s.pt"))
    p.add_argument("--sigma-repo", default=os.environ.get("SIGMA_REPO", "third_party/SigmaPPG"))
    p.add_argument("--sigma-checkpoint", default=os.environ.get("SIGMA_CHECKPOINT", "weights/sigma.pth"))
    p.add_argument("--backbone", choices=["papagei", "sigma"], default="papagei")
    p.add_argument("--sigma-patch-size", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    index = pd.read_csv(a.index)
    if index.empty:
        raise SystemExit("window index is empty; run audit_vitaldb.py first")
    if a.backbone == "papagei":
        model = PaPaGeiEncoder(a.repo, a.checkpoint, a.device)
    else:
        input_size = int((index.end_s.iloc[0] - index.start_s.iloc[0]) * index.fs.iloc[0])
        model = SigmaPPGEncoder(a.sigma_repo, a.sigma_checkpoint, input_size=input_size,
                                patch_size=a.sigma_patch_size, device=a.device)
        if index.protocol.nunique() != 1 or index.protocol.iloc[0] != "sigma":
            raise SystemExit("Sigma extraction requires an index built with --protocol sigma")
    model.eval()
    output = np.empty((len(index), model.embedding_dim), dtype=np.float32)
    # Each cache stores one case. This keeps disk reads and .vital parsing out of
    # the model stage while preserving case-level folds.
    for cache_path, row_indices in index.groupby("cache_path", sort=False).groups.items():
        window_ids = index.loc[row_indices, "window_idx"].to_numpy(dtype=int)
        with np.load(cache_path) as cache:
            signals = cache["ppg"][window_ids]
        for start in range(0, len(signals), a.batch_size):
            batch = torch.from_numpy(signals[start:start + a.batch_size]).unsqueeze(1).to(a.device)
            with torch.inference_mode():
                z = model(batch).float().cpu().numpy()
            output[np.asarray(row_indices)[start:start + len(z)]] = z
        print(f"embedded {len(window_ids)} windows: {Path(cache_path).name}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=output, case_id=index.case_id.to_numpy(dtype=np.int32),
                        start_s=index.start_s.to_numpy(dtype=np.int32))
    print(f"saved {output.shape} embeddings to {out}")


if __name__ == "__main__":
    main()
