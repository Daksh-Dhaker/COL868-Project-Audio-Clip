#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash reproduce/download_datasets.sh [esc|urban|all] [OUTPUT_DIR]
# Example:
#   bash reproduce/download_datasets.sh esc /kaggle/working/datasets
#   bash reproduce/download_datasets.sh urban /kaggle/working/datasets
#   bash reproduce/download_datasets.sh all /kaggle/working/datasets

DATASET_CHOICE="${1:-all}"
OUTPUT_DIR="${2:-/kaggle/working/datasets}"
ESC50_DIR="${OUTPUT_DIR}/ESC-50-master"
US8K_DIR="${OUTPUT_DIR}/UrbanSound8K"

case "${DATASET_CHOICE}" in
  esc|esc50)
    DATASET_CHOICE="esc"
    ;;
  urban|us8k)
    DATASET_CHOICE="urban"
    ;;
  all)
    ;;
  *)
    echo "Invalid dataset choice: ${DATASET_CHOICE}"
    echo "Usage: bash reproduce/download_datasets.sh [esc|urban|all] [OUTPUT_DIR]"
    exit 1
    ;;
esac

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

if [[ "${DATASET_CHOICE}" == "esc" || "${DATASET_CHOICE}" == "all" ]]; then
  # ESC-50 from official GitHub repository archive.
  ESC50_ZIP="${OUTPUT_DIR}/ESC-50-master.zip"
  download_if_missing "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" "${ESC50_ZIP}"
  if [[ -d "${ESC50_DIR}" ]]; then
    echo "ESC-50 already extracted: ${ESC50_DIR}"
  else
    echo "Extracting ESC-50..."
    unzip -q "${ESC50_ZIP}" -d "${OUTPUT_DIR}"
  fi
fi

if [[ "${DATASET_CHOICE}" == "urban" || "${DATASET_CHOICE}" == "all" ]]; then
  # UrbanSound8K from Zenodo mirror.
  US8K_TAR="${OUTPUT_DIR}/UrbanSound8K.tar.gz"
  download_if_missing "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz" "${US8K_TAR}"
  if [[ -d "${US8K_DIR}" ]]; then
    echo "UrbanSound8K already extracted: ${US8K_DIR}"
  else
    echo "Extracting UrbanSound8K..."
    tar -xzf "${US8K_TAR}" -C "${OUTPUT_DIR}"
  fi
fi

echo ""
echo "Done. Dataset roots:"
if [[ "${DATASET_CHOICE}" == "esc" || "${DATASET_CHOICE}" == "all" ]]; then
  echo "ESC-50 root:      ${ESC50_DIR}"
fi
if [[ "${DATASET_CHOICE}" == "urban" || "${DATASET_CHOICE}" == "all" ]]; then
  echo "UrbanSound8K root:${US8K_DIR}"
fi
echo ""
echo "Use with reproduce scripts:"
if [[ "${DATASET_CHOICE}" == "esc" || "${DATASET_CHOICE}" == "all" ]]; then
  echo "  --dataset-root ${ESC50_DIR}"
fi
if [[ "${DATASET_CHOICE}" == "urban" || "${DATASET_CHOICE}" == "all" ]]; then
  echo "  --dataset-root ${US8K_DIR}"
fi
