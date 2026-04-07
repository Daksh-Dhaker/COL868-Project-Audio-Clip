#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash reproduce/run_esc50.sh <DATASET_ROOT> <CHECKPOINT_PATH> [FOLD] [EPOCHS]"
  exit 1
fi

DATASET_ROOT="$1"
CHECKPOINT_PATH="$2"
FOLD="${3:-1}"
EPOCHS="${4:-50}"

python reproduce/run_fold.py \
  --dataset esc50 \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --fold "${FOLD}" \
  --epochs "${EPOCHS}"
