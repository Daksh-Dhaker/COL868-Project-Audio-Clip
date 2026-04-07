#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-reproduce/models}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
BASE_DIR="${OUTPUT_DIR}/base"

mkdir -p "${CHECKPOINT_DIR}" "${BASE_DIR}"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "Skipping existing file: ${out}"
  else
    echo "Downloading $(basename "${out}")"
    curl -L --fail "${url}" -o "${out}"
  fi
}

# Main AudioCLIP checkpoints (pass one of these to --checkpoint)
download_if_missing "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt" "${CHECKPOINT_DIR}/AudioCLIP-Full-Training.pt"
download_if_missing "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt" "${CHECKPOINT_DIR}/AudioCLIP-Partial-Training.pt"

# Base model weights used by fallback paths in AudioCLIP internals
download_if_missing "https://raw.githubusercontent.com/AndreyGuzhov/AudioCLIP/master/assets/CLIP.pt" "${BASE_DIR}/CLIP.pt"
download_if_missing "https://raw.githubusercontent.com/AndreyGuzhov/AudioCLIP/master/assets/ESRNXFBSP.pt" "${BASE_DIR}/ESRNXFBSP.pt"

# Tokenizer vocab used by CLIP text encoder
download_if_missing "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz" "${BASE_DIR}/bpe_simple_vocab_16e6.txt.gz"

echo ""
echo "Downloaded models to: ${OUTPUT_DIR}"
echo "Checkpoint to use: ${CHECKPOINT_DIR}/AudioCLIP-Full-Training.pt"
