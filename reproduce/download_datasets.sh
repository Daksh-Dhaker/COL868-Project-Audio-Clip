#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash reproduce/download_datasets.sh [OUTPUT_DIR]
# Example:
#   bash reproduce/download_datasets.sh /kaggle/working/datasets

OUTPUT_DIR="${1:-/kaggle/working/datasets}"
ESC50_DIR="${OUTPUT_DIR}/ESC-50-master"
US8K_DIR="${OUTPUT_DIR}/UrbanSound8K"

mkdir -p "${OUTPUT_DIR}"

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

# ESC-50 from official GitHub repository archive.
ESC50_ZIP="${OUTPUT_DIR}/ESC-50-master.zip"
download_if_missing "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" "${ESC50_ZIP}"
if [[ -d "${ESC50_DIR}" ]]; then
  echo "ESC-50 already extracted: ${ESC50_DIR}"
else
  echo "Extracting ESC-50..."
  unzip -q "${ESC50_ZIP}" -d "${OUTPUT_DIR}"
fi

# UrbanSound8K from Zenodo mirror.
US8K_TAR="${OUTPUT_DIR}/UrbanSound8K.tar.gz"
download_if_missing "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz" "${US8K_TAR}"
if [[ -d "${US8K_DIR}" ]]; then
  echo "UrbanSound8K already extracted: ${US8K_DIR}"
else
  echo "Extracting UrbanSound8K..."
  tar -xzf "${US8K_TAR}" -C "${OUTPUT_DIR}"
fi

echo ""
echo "Done. Dataset roots:"
echo "ESC-50 root:      ${ESC50_DIR}"
echo "UrbanSound8K root:${US8K_DIR}"
echo ""
echo "Use with reproduce scripts:"
echo "  --dataset-root ${ESC50_DIR}"
echo "  --dataset-root ${US8K_DIR}"
