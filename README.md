# Acknowledgment

This project is based on the original AudioCLIP implementation by Andrey Guzhov et al.:

AudioCLIP GitHub Repository: https://github.com/AndreyGuzhov/AudioCLIP

I used the original repository as a base and performed additional experiments/modifications for this work. All credit for the original implementation goes to the original authors.


# AudioCLIP Reproduction

This folder contains helper scripts to reproduce AudioCLIP results from the paper, including zero-shot evaluation, retrieval metrics, fine-tuned model evaluation, cross-dataset transfer, and qualitative analysis.

## Prerequisites


1a. Install dependencies (Kaggle environment):

```bash
!python -m pip install -r AudioCLIP/reproduce/requirements-kaggle.txt
```

1b. Install dependencies (Local)

Create Conda enviroment and activate it

```bash
conda create -n Aclip python=3.9

conda activate Aclip
```
install requirements

```bash
pip install -r requirements.txt
```

2. Datasets:
   - **ESC-50**: root containing `audio/` and `meta/esc50.csv`
   - **UrbanSound8K**: root containing `audio/` and `metadata/UrbanSound8K.csv`
   - **Flickr8k**: root containing images and caption annotations
  
   ## Download Datasets
   - **ESC-50**: `wget -q https://github.com/karoldvl/ESC-50/archive/master.zip -O ESC-50.zip`
   - **UrbanSound8K**: `kaggle datasets download -d chrisfilo/urbansound8k`
   - **Flickr8k**: `kaggle datasets download -d shadabhussain/flickr8k`

   ## Datasets can also be downloaded using below links
   - **UrbanSound8K**: https://www.kaggle.com/datasets/chrisfilo/urbansound8k
   - **Flickr8k**: https://www.kaggle.com/datasets/shadabhussain/flickr8k
   
   ## Unzipping ESC-50
   ```bash
   unzip ESC-50.zip
   ```

   ## Unzipping UrbanSound8k
   ```bash
   mkdir -p US8k
   unzip urbansound8k.zip -d US8k
   ```
   
   ## Unzipping Flickr8k
   ```bash
   mkdir -p Flickr8k
   unzip flickr8k.zip -d Flickr8k
   ```
2. **Model Weights (Google Drive)**
   All the model weights are available in this drive link - https://drive.google.com/drive/folders/17dQUZ_mE49zjZZrI8LRQaphbv2DJoiOo?usp=share_link

   The individual models can also be installed using gdown if needed:

```bash
pip install gdown
```



**AudioCLIP (Full Training)**

```bash
gdown https://drive.google.com/uc?id=1Lmq5MGeZG-H64PI-Gy9fOe6cYQNS9dhU
```

**AudioCLIP (Partially Trained)**

```bash
gdown https://drive.google.com/uc?id=119vdeAaHcO-SgiQ6iZujpuwJDfQxuMze
```


**ESC + USK + Flickr (Combined Training)**

```bash
gdown https://drive.google.com/uc?id=1v8omhbr7IKKdSpGv89-ArRnzOuRWxpkB
```



**ESC-50 (Fully Trained)**

```bash
gdown https://drive.google.com/uc?id=15nhkQKbmgmVzth6gvYm6hvKw-6On7HML
```

**ESC-50 (Partially Trained)**

```bash
gdown https://drive.google.com/uc?id=1aTaK8_DDJMtVrVqDOOFdMUpc7VG_EqwQ
```

**ESC-50 + USK (Joint Training)**

```bash
gdown https://drive.google.com/uc?id=1d-W5_qvu1l8yaeKlES_LpEVdbBeG1q1m
```


**Flickr (Fully Trained)**

```bash
gdown https://drive.google.com/uc?id=1wzP5MdxMV_8qLL9FdmWp2gYy6TzXJQlk
```

**Flickr (Partially Trained)**

```bash
gdown https://drive.google.com/uc?id=1vPLN2XvpqGc-kevHjEYT-MrQtBY1T-S3
```


**UrbanSound8K (Fully Trained)**

```bash
gdown https://drive.google.com/uc?id=1g0eroXavlxW3c2LKQ7bGu6XvUN2e74ZA
```

**UrbanSound8K (Partially Trained)**

```bash
gdown https://drive.google.com/uc?id=1zKn4jzKU4Bo1_0IHbOxdRTzFTAvw--ps
```



4. Model checkpoints:
   - `AudioCLIP-Full-Training.pt` (pre-trained, full)
   - `AudioCLIP-Partial-Training.pt` (pre-trained, partial)
   - Fine-tuned checkpoints: `Esc-50_Full.pt`, `Esc-50_Partial.pt`, `Usk8-Full.pt`, `Usk8-Partial.pt`, `Flickr_Full.pt`, `Flickr_Partial.pt`, `Esc-50+Usk8_Full.pt`, `Esc_Usk_Flickr.pt`

---

Change current directory

```bash
cd reproduce/
```

## 1. Zero-Shot Evaluation (`zero_shot_eval.py`)

Computes R@1, R@5, R@10, and mAP for a given query→result modality pair.

### ESC-50

#### Audio → Text

```bash
# Full pre-trained model
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol paper

# Partial pre-trained model
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol paper
```

#### Text → Audio

```bash
# Full pre-trained model
python zero_shot_eval.py \
    --dataset esc50 \
    --pair text-audio \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol paper

# Partial pre-trained model
python zero_shot_eval.py \
    --dataset esc50 \
    --pair text-audio \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol paper
```

### UrbanSound8K

#### Audio → Text

```bash
# Partial pre-trained model
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol paper

# Full pre-trained model
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol paper
```

#### Text → Audio

```bash
# Full pre-trained model
python zero_shot_eval.py \
    --dataset us8k \
    --pair text-audio \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol paper

# Partial pre-trained model
python zero_shot_eval.py \
    --dataset us8k \
    --pair text-audio \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol paper
```

### Flickr8k

#### Image → Text

```bash
# Full pre-trained model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol global

# Partial pre-trained model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol global
```

#### Text → Image

```bash
# Full pre-trained model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair text-image \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/AudioCLIP-Full-Training.pt \
    --protocol global

# Partial pre-trained model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair text-image \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/AudioCLIP-Partial-Training.pt \
    --protocol global
```

---

## 2. Fine-Tuned Model Evaluation

Evaluate models fine-tuned on specific datasets. Uses `--fold 1` for fold-based datasets.

### ESC-50 Fine-Tuned → ESC-50

```bash
# Fine-tuned Full model (audio → text)
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Esc-50_Full.pt \
    --protocol paper --fold 1

# Fine-tuned Partial model (audio → text)
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Esc-50_Partial.pt \
    --protocol paper --fold 1

# Fine-tuned Partial model (text → audio)
python zero_shot_eval.py \
    --dataset esc50 \
    --pair text-audio \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Esc-50_Partial.pt \
    --protocol paper --fold 1
```

### UrbanSound8K Fine-Tuned → UrbanSound8K

```bash
# Fine-tuned Full model
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/Usk8-Full.pt \
    --protocol paper --fold 1

# Fine-tuned Partial model
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/Usk8-Partial.pt \
    --protocol paper --fold 1
```

### Flickr8k Fine-Tuned → Flickr8k

```bash
# Fine-tuned Full model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/Flickr_Full.pt \
    --protocol global

# Fine-tuned Partial model
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/Flickr_Partial.pt \
    --protocol global
```

---

## 3. Cross-Dataset Transfer Evaluation

Evaluate fine-tuned models on datasets they were **not** fine-tuned on, to measure transfer and generalization.

### Flickr Fine-Tuned → ESC-50 / UrbanSound8K

```bash
# Flickr_Full on ESC-50 (audio → text)
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Flickr_Full.pt \
    --protocol paper --fold 1

# Flickr_Full on UrbanSound8K (audio → text)
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/Flickr_Full.pt \
    --protocol paper --fold 1
```

### ESC-50 + UrbanSound8K Joint Fine-Tuned → All Datasets

```bash
# Esc-50+Usk8_Full on ESC-50
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Esc-50+Usk8_Full.pt \
    --protocol paper --fold 1

# Esc-50+Usk8_Full on UrbanSound8K
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/Esc-50+Usk8_Full.pt \
    --protocol paper --fold 1

# Esc-50+Usk8_Full on Flickr8k
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/Esc-50+Usk8_Full.pt \
    --protocol global
```

### ESC-50 + UrbanSound8K + Flickr Joint Fine-Tuned → All Datasets

```bash
# Esc_Usk_Flickr on ESC-50
python zero_shot_eval.py \
    --dataset esc50 \
    --pair audio-text \
    --dataset-root /path/to/ESC-50-master \
    --model-path /path/to/Esc_Usk_Flickr.pt \
    --protocol paper --fold 1

# Esc_Usk_Flickr on UrbanSound8K
python zero_shot_eval.py \
    --dataset us8k \
    --pair audio-text \
    --dataset-root /path/to/UrbanSound8K \
    --model-path /path/to/Esc_Usk_Flickr.pt \
    --protocol paper --fold 1

# Esc_Usk_Flickr on Flickr8k
python zero_shot_eval.py \
    --dataset flickr8k \
    --pair image-text \
    --dataset-root /path/to/flickr8k \
    --model-path /path/to/Esc_Usk_Flickr.pt \
    --protocol global
```

---

## 4. Qualitative Analysis (`analysis.py`)

Compare two model checkpoints side-by-side. Generates per-class accuracy charts, confusion matrices, prediction flip tables, and a summary JSON.

```bash
python analysis.py \
    --dataset esc50 --pair audio-text --protocol paper --fold 1 \
    --dataset-root /path/to/ESC-50-master \
    --model-a /path/to/AudioCLIP-Full-Training.pt --label-a "Untrained" \
    --model-b /path/to/Esc-50_Full.pt --label-b "FT-ESC50" \
    --output-dir ./analysis_esc50
```

---

## Notes

1. All `zero_shot_eval.py` commands use `--protocol paper` (fold-wise averaging) for ESC-50 and UrbanSound8K, and `--protocol global` for Flickr8k.
2. Fine-tuned model evaluations use `--fold 1` to evaluate on the held-out fold matching the training split.
3. Reference supervised values from paper: ESC-50 97.15%, UrbanSound8K 90.07%.
