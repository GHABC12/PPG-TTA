#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-papagei}"
INDEX="${INDEX:-results/vitaldb_pilot/window_index.csv}"
PRETRAINED="${PAPAGEI_CHECKPOINT:-weights/papagei_s.pt}"
REPO="${PAPAGEI_REPO:-third_party/papagei-foundation-model}"
OUT="${OUT:-results/vitaldb_pilot/baseline_eval}"
mkdir -p "${OUT}"

for strategy in linear_probe ln_head last_layers full; do
  for fold in 0 1 2 3 4; do
    if [[ "${strategy}" == linear_probe ]]; then
      checkpoint="results/vitaldb_pilot/source/source_fold${fold}.pt"
    else
      checkpoint="results/vitaldb_pilot/baselines/${strategy}/finetune_${BACKBONE}_${strategy}_fold${fold}.pt"
    fi
    python scripts/evaluate_source_checkpoint.py \
      --index "${INDEX}" --checkpoint "${checkpoint}" --backbone "${BACKBONE}" \
      --test-fold "${fold}" --repo "${REPO}" --pretrained "${PRETRAINED}" \
      --out "${OUT}/${strategy}_fold${fold}.json"
  done
done
python scripts/summarize_baseline_grid.py --root results/vitaldb_pilot --out-dir results
