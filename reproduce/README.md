# AudioCLIP Reproduction on Kaggle (Input Dataset First)

This workflow assumes you upload the entire `AudioCLIP/` folder as a Kaggle Dataset input.

Design goals of this setup:

1. Use model files directly from `AudioCLIP/assets` in input dataset.
2. Use `/kaggle/working` only for downloaded datasets (and optional outputs).
3. Disable Visdom and checkpoint saving by default.
4. Reproduce printed validation accuracies with minimal setup.

## Assumptions

- Code dataset path:
  - `/kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP`
- `AudioCLIP/assets` contains real files (not Git LFS pointers):
  - `AudioCLIP-Full-Training.pt`
  - `CLIP.pt`
  - `ESRNXFBSP.pt`
  - `bpe_simple_vocab_16e6.txt.gz`

## 1) Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/requirements-kaggle.txt
```

## 2) Download Datasets (ESC-50 and UrbanSound8K)

```bash
bash /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/download_datasets.sh /kaggle/working/datasets
```

This creates:

- ESC-50 root: `/kaggle/working/datasets/ESC-50-master`
- UrbanSound8K root: `/kaggle/working/datasets/UrbanSound8K`

## 3) Run Single Fold (uses assets checkpoint by default)

ESC-50 fold 1:

```bash
python /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/run_fold.py \
  --dataset esc50 \
  --dataset-root /kaggle/working/datasets/ESC-50-master \
  --fold 1
```

UrbanSound8K fold 1:

```bash
python /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/run_fold.py \
  --dataset us8k \
  --dataset-root /kaggle/working/datasets/UrbanSound8K \
  --fold 1
```

## 4) Run Full Cross-Validation

ESC-50 (5 folds):

```bash
python /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/run_cv.py \
  --dataset esc50 \
  --dataset-root /kaggle/working/datasets/ESC-50-master
```

UrbanSound8K (10 folds):

```bash
python /kaggle/input/datasets/dakshddhaker/code-5/AudioCLIP/reproduce/run_cv.py \
  --dataset us8k \
  --dataset-root /kaggle/working/datasets/UrbanSound8K
```

## 5) Optional Overrides

Use custom checkpoint path:

```bash
--checkpoint /some/path/AudioCLIP-Full-Training.pt
```

Enable checkpoint saving:

```bash
--enable-checkpoint-saving --saved-models-path /kaggle/working/outputs/saved_models
```

Enable visdom (off by default):

```bash
--enable-visdom
```

## 6) Notes

- Ignite is still required because AudioCLIP trainer is built on Ignite.
- Visdom is disabled by default.
- Checkpoint saving is disabled by default.
- Console logs still print `Val. Acc. Eval.` values needed for reproduction.
