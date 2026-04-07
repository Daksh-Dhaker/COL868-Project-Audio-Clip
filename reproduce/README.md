# AudioCLIP Reproduction

This folder contains helper scripts to reproduce multiple AudioCLIP results from the paper.

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
