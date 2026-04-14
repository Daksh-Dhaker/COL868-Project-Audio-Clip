# AudioCLIP Reproduction

This folder contains helper scripts to reproduce multiple AudioCLIP results from the paper.

## Table 4 Retrieval Reproduction (table4.py)

Use `reproduce/table4.py` to compute Table 4 metrics (`P@1`, `R@1`, `mAP`) for one setting at a time.

Required inputs:
1. `--dataset` (`imagenet`, `audioset`, `esc50`, `us8k`)
2. `--query-type` (`text`, `audio`, `image`)
3. `--result-type` (`text`, `audio`, `image`)
4. `--model-path` (full or partial AudioCLIP checkpoint)

Example (ESC-50, text -> audio, full model):

```text
python reproduce/table4.py --dataset esc50 --query-type text --result-type audio --model-path assets/AudioCLIP-Full-Training.pt --dataset-root /path/to/ESC-50-master
```

Example (UrbanSound8K, text -> audio, partial model):

```text
python reproduce/table4.py --dataset us8k --query-type text --result-type audio --model-path assets/AudioCLIP-Partial-Training.pt --dataset-root /path/to/UrbanSound8K
```

For `imagenet` and `audioset`, pass `--dataset-root` pointing to a directory containing modality manifests:
1. `audio.jsonl` (optional unless audio modality is used)
2. `image.jsonl` (optional unless image modality is used)
3. `text.jsonl` (optional; auto-generated from labels if omitted)

Manifest line format:

```text
{"path": "relative/or/absolute/path.ext", "labels": ["label_a", "label_b"]}
```

Text manifest line format:

```text
{"text": "cat", "labels": ["cat"]}
```

## What Is Included

1. Single-fold reproduction for ESC-50 and UrbanSound8K.
2. Full cross-validation (ESC-50: 5 folds, UrbanSound8K: 10 folds).
3. Combined runners for multiple paper-claim cases (full and partial checkpoints).

## Prerequisites

1. Python environment with dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r reproduce/requirements-kaggle.txt
```

2. Dataset roots available locally:
   ESC-50 root containing `audio/` and `meta/esc50.csv`
   UrbanSound8K root containing `audio/` and `metadata/UrbanSound8K.csv`

  You can download datasets selectively with Python:

```bash
# Download only ESC-50
python reproduce/download_datasets.py esc /path/to/datasets

# Download only UrbanSound8K
python reproduce/download_datasets.py urban /path/to/datasets

# Download both datasets
python reproduce/download_datasets.py all /path/to/datasets
```

  Shell version (equivalent):

```bash
# Download only ESC-50
bash reproduce/download_datasets.sh esc /path/to/datasets

# Download only UrbanSound8K
bash reproduce/download_datasets.sh urban /path/to/datasets

# Download both datasets
bash reproduce/download_datasets.sh all /path/to/datasets
```

3. Checkpoints and tokenizer files are available in `AudioCLIP/assets`.
   The main scripts default to `assets/AudioCLIP-Full-Training.pt`.

## Run Single Fold

ESC-50 fold 1:

```bash
python reproduce/run_fold.py \
  --dataset esc50 \
  --dataset-root /path/to/ESC-50-master \
  --fold 1 \
  --enable-checkpoint-saving \
  --saved-models-path reproduce/outputs/esc50_saved_models
```

UrbanSound8K fold 1:

```bash
python reproduce/run_fold.py \
  --dataset us8k \
  --dataset-root /path/to/UrbanSound8K \
  --fold 1 \
  --enable-checkpoint-saving \
  --saved-models-path reproduce/outputs/us8k_saved_models
```

## Run Full Cross-Validation

ESC-50 (5 folds):

```bash
python reproduce/run_cv.py \
  --dataset esc50 \
  --dataset-root /path/to/ESC-50-master \
  --enable-checkpoint-saving \
  --saved-models-path reproduce/outputs/esc50_cv_saved_models
```

UrbanSound8K (10 folds):

```bash
python reproduce/run_cv.py \
  --dataset us8k \
  --dataset-root /path/to/UrbanSound8K \
  --enable-checkpoint-saving \
  --saved-models-path reproduce/outputs/us8k_cv_saved_models
```

## Run Additional Paper-Claim Cases

The claims runner is now selection-based: you choose datasets and model variants.

Run full-checkpoint CV for both datasets:

```bash
python reproduce/run_paper_claims.py \
  --datasets esc50 us8k \
  --models full \
  --esc50-root /path/to/ESC-50-master \
  --us8k-root /path/to/UrbanSound8K \
  --full-checkpoint assets/AudioCLIP-Full-Training.pt
```

Run full + partial checkpoint CV for both datasets:

```bash
python reproduce/run_paper_claims.py \
  --datasets esc50 us8k \
  --models full partial \
  --esc50-root /path/to/ESC-50-master \
  --us8k-root /path/to/UrbanSound8K \
  --full-checkpoint assets/AudioCLIP-Full-Training.pt \
  --partial-checkpoint assets/AudioCLIP-Partial-Training.pt
```

Run a single dataset and single model variant (example: ESC-50 + full):

```bash
python reproduce/run_paper_claims.py \
  --datasets esc50 \
  --models full \
  --esc50-root /path/to/ESC-50-master \
  --full-checkpoint assets/AudioCLIP-Full-Training.pt
```

Run only UrbanSound8K with partial checkpoint:

```bash
python reproduce/run_paper_claims.py \
  --datasets us8k \
  --models partial \
  --us8k-root /path/to/UrbanSound8K \
  --partial-checkpoint assets/AudioCLIP-Partial-Training.pt
```

Shell wrappers:

```bash
bash reproduce/run_claims_full.sh /path/to/ESC-50-master /path/to/UrbanSound8K
bash reproduce/run_claims_full_and_partial.sh /path/to/ESC-50-master /path/to/UrbanSound8K
```

## Notes

1. Visdom is disabled by default in wrappers; enable it with `--enable-visdom`.
2. Scripts print fold-level values and aggregated `Mean Val. Acc. Eval.` and `Std Val. Acc. Eval.`.
3. Reference supervised values from paper:
   ESC-50: 97.15%
   UrbanSound8K: 90.07%
