#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-papagei}"
INDEX="${INDEX:-results/vitaldb_pilot/window_index.csv}"
STRATEGIES="${STRATEGIES:-ln_head last_layers full}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-8}"

for strategy in ${STRATEGIES}; do
  for test_fold in 0 1 2 3 4; do
    val_fold=$(((test_fold + 1) % 5))
    out_dir="results/vitaldb_pilot/baselines/${strategy}"
    checkpoint="${out_dir}/finetune_${BACKBONE}_${strategy}_fold${test_fold}.pt"
    if [[ -f "${checkpoint}" ]]; then
      echo "skip existing ${checkpoint}"
      continue
    fi
    extra=()
    if [[ "${strategy}" == "last_layers" ]]; then
      extra+=(--tail-blocks "${TAIL_BLOCKS:-4}")
    fi
    if [[ "${strategy}" == "full" ]]; then
      extra+=(--encoder-lr "${FULL_ENCODER_LR:-3e-6}")
    fi
    python scripts/train_finetune.py \
      --backbone "${BACKBONE}" --strategy "${strategy}" \
      --index "${INDEX}" --test-fold "${test_fold}" --val-fold "${val_fold}" \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" "${extra[@]}" \
      --out-dir "${out_dir}"
  done
done
