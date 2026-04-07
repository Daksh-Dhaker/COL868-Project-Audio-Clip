#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash reproduce/run_claims_full_and_partial.sh <ESC50_ROOT> <US8K_ROOT> [FULL_CHECKPOINT] [PARTIAL_CHECKPOINT]"
  exit 1
fi

ESC50_ROOT="$1"
US8K_ROOT="$2"
FULL_CHECKPOINT="${3:-assets/AudioCLIP-Full-Training.pt}"
PARTIAL_CHECKPOINT="${4:-assets/AudioCLIP-Partial-Training.pt}"

python reproduce/run_paper_claims.py \
  --esc50-root "${ESC50_ROOT}" \
  --us8k-root "${US8K_ROOT}" \
  --full-checkpoint "${FULL_CHECKPOINT}" \
  --partial-checkpoint "${PARTIAL_CHECKPOINT}"
