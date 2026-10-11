#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-papagei}"
INDEX="${INDEX:-results/vitaldb_pilot/window_index.csv}"
TEST_FOLD="${TEST_FOLD:-0}"
SOURCE="${SOURCE:?set SOURCE to the selected source checkpoint for TEST_FOLD}"
OUT="${OUT:-results/vitaldb_pilot/shift_stress/fold${TEST_FOLD}}"
BATCH_SIZE="${BATCH_SIZE:-16}"

for spec in gain:0.75 gain:1.25 gain:1.5 \
            baseline_drift:0.1 baseline_drift:0.25 baseline_drift:0.5 \
            gaussian_noise:0.01 gaussian_noise:0.05 gaussian_noise:0.1 \
            motion:0.5 motion:1 motion:2; do
  kind="${spec%%:*}"
  severity="${spec#*:}"
  tag="${kind}_${severity//./p}"
  python scripts/evaluate_tta.py \
    --index "${INDEX}" --source "${SOURCE}" --test-fold "${TEST_FOLD}" \
    --role test --backbone "${BACKBONE}" --batch-size "${BATCH_SIZE}" \
    --quality-mode gated --quality-threshold 0.5 --uncertainty-mode off \
    --lambda-morph 0 --lr 1e-6 --shift "${kind}" --severity "${severity}" \
    --out-dir "${OUT}/${tag}" --run-name "${tag}"
done
