#!/usr/bin/env python3
"""
Qualitative & quantitative analysis comparing two AudioCLIP model checkpoints.

Generates:
  1. Per-class accuracy bar chart (side-by-side)
  2. Confusion matrices (side-by-side)
  3. Score-distribution histograms (correct vs incorrect, per model)
  4. Prediction-flip table (wrong→right, right→wrong examples)
  5. Top-K prediction examples for selected samples
  6. Per-class accuracy delta chart (improvement / degradation)
  7. Retrieval rank-shift scatter plot
  8. Summary JSON with all numeric results

Usage examples:
  # Compare untrained full model vs fine-tuned on ESC-50 (audio→text, fold 5)
  python analysis.py \\
      --dataset esc50 --pair text-audio --protocol paper --fold 5 \\
      --dataset-root /data/ESC-50 \\
      --model-a /models/AudioCLIP-Full-Training.pt --label-a "Untrained" \\
      --model-b /models/Esc-50_Full.pt            --label-b "FT-ESC50" \\
      --output-dir ./analysis_outputs

  # Compare on Flickr8k (image→text, 30% test split)
  python analysis.py \\
      --dataset flickr8k --pair image-text --protocol global \\
      --dataset-root /data/flickr8k \\
      --model-a /models/AudioCLIP-Full-Training.pt --label-a "Untrained" \\
      --model-b /models/Flickr_Full.pt             --label-b "FT-Flickr" \\
      --output-dir ./analysis_flickr
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---------------------------------------------------------------------------
# Reuse infrastructure from table4 / zero_shot_eval
# ---------------------------------------------------------------------------
from table4 import (
    AudioCLIP,
    Item,
    audio_transform,
    default_audio_length,
    encode_audio,
    encode_image,
    encode_text,
    folds_for_dataset,
    image_transform,
    pair_logits,
    resolve_checkpoint_path,
)
from zero_shot_eval import (
    DATASET_CHOICES,
    DATASET_MODALITIES,
    FLICKR_SPLIT_SEED,
    K_VALUES,
    PAIR_CHOICES,
    build_coco_combined,
    build_single_dataset,
    filter_items_by_labels,
    flickr_train_test_split,
    mean_average_precision,
    parse_pair,
    prepare_items_for_pair,
    retrieval_hit_at_k,
    retrieval_r_at_k,
    sample_classification_accuracy,
    supervised_classification_accuracy_any,
    validate_pair_for_dataset,
)


# ============================= CLI =========================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qualitative analysis comparing two AudioCLIP checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    # Dataset / pair
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--pair", required=True, choices=PAIR_CHOICES)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--coco-image-root", type=Path, default=None)
    p.add_argument("--coco-audio-root", type=Path, default=None)
    p.add_argument("--coco-text-source", choices=["coco2014", "spokencoco", "union"], default="coco2014")
    p.add_argument("--protocol", choices=["paper", "global"], default="paper")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--prompt-template", default="{}")

    # Two models
    p.add_argument("--model-a", required=True, type=Path, help="First model (baseline).")
    p.add_argument("--label-a", default="Model-A", help="Display label for model A.")
    p.add_argument("--model-b", required=True, type=Path, help="Second model (e.g. fine-tuned).")
    p.add_argument("--label-b", default="Model-B", help="Display label for model B.")

    # Output
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-length", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-examples", type=int, default=20, help="Max flip-examples to print/save.")
    p.add_argument("--top-k-examples", type=int, default=5, help="Number of Top-K example items to detail.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    return p.parse_args()


# ============================= Data loading =================================

def load_items(args: argparse.Namespace) -> tuple[dict[str, list[Item]], str]:
    """Load dataset items; apply Flickr 30% test split when applicable."""
    if args.dataset == "coco":
        if args.coco_image_root is None or args.coco_audio_root is None:
            raise ValueError("--dataset coco requires --coco-image-root and --coco-audio-root")
        image_root = args.coco_image_root.resolve()
        audio_root = args.coco_audio_root.resolve()
        if not image_root.exists():
            raise FileNotFoundError(f"COCO image root not found: {image_root}")
        if not audio_root.exists():
            raise FileNotFoundError(f"SpokenCOCO root not found: {audio_root}")
        text_items, audio_items, image_items = build_coco_combined(image_root, audio_root, args.coco_text_source)
        desc = f"coco (image={image_root}, audio={audio_root})"
    else:
        if args.dataset_root is None:
            raise ValueError(f"--dataset {args.dataset} requires --dataset-root")
        dataset_root = args.dataset_root.resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        text_items, audio_items, image_items = build_single_dataset(args.dataset, dataset_root)
        desc = f"{args.dataset} ({dataset_root})"

    # Flickr: use 30% test split only
    if args.dataset == "flickr8k":
        _tt, _ti, test_text, test_image = flickr_train_test_split(text_items, image_items)
        text_items = test_text
        image_items = test_image
        print(f"[Info] Flickr8k: using 30% test split ({len(test_image)} images, seed={FLICKR_SPLIT_SEED})")

    return {"text": text_items, "audio": audio_items, "image": image_items}, desc


def load_model(path: Path, device: torch.device) -> AudioCLIP:
    resolved = resolve_checkpoint_path(path.resolve())
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    model = AudioCLIP(pretrained=str(resolved)).to(device)
    model.eval()
    return model


# ============================= Feature encoding =============================

def encode_all(
    model: AudioCLIP,
    q_type: str,
    r_type: str,
    all_items: dict[str, list[Item]],
    batch_size: int,
    device: torch.device,
    sample_rate: int,
    audio_len: int,
    prompt_template: str,
    label: str,
    show_progress: bool,
) -> tuple[np.ndarray, list[Item], list[Item]]:
    """Encode features for both modalities and return (scores, q_items, r_items)."""
    q_items = all_items[q_type]
    r_items = all_items[r_type]

    img_tf = image_transform()
    aud_tf = audio_transform(audio_len)

    feats: dict[str, torch.Tensor] = {}
    for mod in {q_type, r_type}:
        if mod == "text":
            if show_progress:
                print(f"  [{label}] Encoding text features...")
            feats["text"] = encode_text(model, all_items["text"], prompt_template, batch_size)
        elif mod == "audio":
            if show_progress:
                print(f"  [{label}] Encoding audio features...")
            feats["audio"] = encode_audio(model, all_items["audio"], batch_size, device, sample_rate, aud_tf)
        elif mod == "image":
            if show_progress:
                print(f"  [{label}] Encoding image features...")
            feats["image"] = encode_image(model, all_items["image"], batch_size, device, img_tf)

    def select(mod: str, subset: list[Item]) -> torch.Tensor:
        key_to_idx = {it.key: i for i, it in enumerate(all_items[mod])}
        ids = [key_to_idx[it.key] for it in subset]
        return feats[mod][ids]

    qf = select(q_type, q_items)
    rf = select(r_type, r_items)
    scores = pair_logits(model, q_type, r_type, qf, rf).numpy()
    return scores, q_items, r_items


# ============================= Per-sample predictions =======================

def per_sample_predictions(
    scores: np.ndarray,
    q_items: list[Item],
    r_items: list[Item],
    q_type: str,
    r_type: str,
) -> tuple[list[Item], list[Item], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (sample_items, prototype_items, per_sample_scores, pred_indices, correct).
    Normalises direction so samples are always the non-text side when text is involved,
    otherwise queries are 'samples'.
    """
    if q_type == "text":
        sample_items = r_items
        prototype_items = q_items
        ps = scores.T
    elif r_type == "text":
        sample_items = q_items
        prototype_items = r_items
        ps = scores
    else:
        # No text: treat queries as samples, results as prototypes
        sample_items = q_items
        prototype_items = r_items
        ps = scores

    pred_idx = np.argmax(ps, axis=1)
    correct = np.array(
        [bool(sample_items[i].labels & prototype_items[int(pred_idx[i])].labels) for i in range(len(sample_items))],
        dtype=bool,
    )
    return sample_items, prototype_items, ps, pred_idx, correct


def item_display(item: Item) -> str:
    """Human-readable one-liner for an Item."""
    label_str = ", ".join(sorted(item.labels))
    if item.text:
        return f'"{item.text}" [{label_str}]'
    if item.path:
        return f"{item.path.name} [{label_str}]"
    return f"{item.key} [{label_str}]"


# ============================= Plotting helpers =============================

def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_accuracy(
    classes: list[str],
    acc_a: dict[str, float],
    acc_b: dict[str, float],
    label_a: str,
    label_b: str,
    out_path: Path,
    dpi: int,
) -> None:
    n = len(classes)
    x = np.arange(n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
    ax.bar(x - w / 2, [acc_a.get(c, 0) for c in classes], w, label=label_a, alpha=0.85)
    ax.bar(x + w / 2, [acc_b.get(c, 0) for c in classes], w, label=label_b, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy Comparison")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_per_class_delta(
    classes: list[str],
    acc_a: dict[str, float],
    acc_b: dict[str, float],
    label_a: str,
    label_b: str,
    out_path: Path,
    dpi: int,
) -> None:
    deltas = [acc_b.get(c, 0) - acc_a.get(c, 0) for c in classes]
    colors = ["green" if d >= 0 else "red" for d in deltas]
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
    ax.bar(range(n), deltas, color=colors, alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Δ Accuracy (%) [{label_b} − {label_a}]")
    ax.set_title("Per-Class Accuracy Improvement / Degradation")
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(6, n * 0.35)))
    norm = LogNorm(vmin=max(1, cm.min()), vmax=max(1, cm.max())) if cm.max() > 0 else None
    cax = ax.matshow(cm, cmap="Blues", norm=norm)
    fig.colorbar(cax, ax=ax, shrink=0.7)
    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=12)
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_score_distributions(
    scores_correct: np.ndarray,
    scores_incorrect: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 50
    if len(scores_correct) > 0:
        ax.hist(scores_correct, bins=bins, alpha=0.6, label="Correct", color="green", density=True)
    if len(scores_incorrect) > 0:
        ax.hist(scores_incorrect, bins=bins, alpha=0.6, label="Incorrect", color="red", density=True)
    ax.set_xlabel("Similarity Score (of predicted class)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_rank_shift(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    label_a: str,
    label_b: str,
    out_path: Path,
    dpi: int,
    max_points: int = 2000,
) -> None:
    """Scatter plot of correct-class rank in Model A vs Model B."""
    n = len(ranks_a)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        ra, rb = ranks_a[idx], ranks_b[idx]
    else:
        ra, rb = ranks_a, ranks_b

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ra, rb, s=6, alpha=0.35, edgecolors="none")
    lim = max(ra.max(), rb.max()) + 1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7, label="No change")
    ax.set_xlabel(f"Rank of true class ({label_a})")
    ax.set_ylabel(f"Rank of true class ({label_b})")
    ax.set_title("Retrieval Rank Shift (lower = better)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_topk_comparison(
    examples: list[dict],
    out_path: Path,
    dpi: int,
) -> None:
    """Bar chart per example showing score of top-K predictions for both models."""
    n_ex = len(examples)
    if n_ex == 0:
        return
    fig, axes = plt.subplots(n_ex, 2, figsize=(14, 3.5 * n_ex), squeeze=False)
    for row, ex in enumerate(examples):
        for col, key in enumerate(["model_a", "model_b"]):
            ax = axes[row][col]
            preds = ex[key]["top_preds"][:7]
            labels_bar = [p["label"][:20] for p in preds]
            scores_bar = [p["score"] for p in preds]
            colors = ["green" if p["correct"] else "salmon" for p in preds]
            ax.barh(range(len(labels_bar)), scores_bar, color=colors, alpha=0.8)
            ax.set_yticks(range(len(labels_bar)))
            ax.set_yticklabels(labels_bar, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Score")
            title = f'{ex[key]["model_label"]} | True: {ex["true_label"][:25]}'
            if col == 0:
                title = f'Sample: {ex["sample_name"][:30]}\n{title}'
            ax.set_title(title, fontsize=8)
    fig.tight_layout()
    _save(fig, out_path, dpi)


# ============================= Core analysis ================================

def compute_per_class_accuracy(
    sample_items: list[Item],
    pred_indices: np.ndarray,
    prototype_items: list[Item],
) -> dict[str, float]:
    """Returns {class_label: accuracy_percent}."""
    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    for i, item in enumerate(sample_items):
        label = next(iter(item.labels))
        class_total[label] += 1
        pred_label = next(iter(prototype_items[int(pred_indices[i])].labels))
        if pred_label in item.labels:
            class_correct[label] += 1
    return {c: 100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in class_total}


def build_confusion_matrix(
    sample_items: list[Item],
    pred_indices: np.ndarray,
    prototype_items: list[Item],
    classes: list[str],
) -> np.ndarray:
    cls2idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for i, item in enumerate(sample_items):
        true_label = next(iter(item.labels))
        pred_label = next(iter(prototype_items[int(pred_indices[i])].labels))
        ti = cls2idx.get(true_label)
        pi = cls2idx.get(pred_label)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def compute_correct_class_ranks(
    sample_items: list[Item],
    prototype_items: list[Item],
    per_sample_scores: np.ndarray,
) -> np.ndarray:
    """For each sample, rank (0-based) of the correct prototype."""
    ranks = np.zeros(len(sample_items), dtype=np.int64)
    sorted_idx = np.argsort(-per_sample_scores, axis=1)
    for i, item in enumerate(sample_items):
        for rank, pi in enumerate(sorted_idx[i]):
            if prototype_items[int(pi)].labels & item.labels:
                ranks[i] = rank
                break
        else:
            ranks[i] = per_sample_scores.shape[1]
    return ranks


def build_topk_examples(
    sample_items: list[Item],
    prototype_items: list[Item],
    ps_a: np.ndarray,
    ps_b: np.ndarray,
    pred_a: np.ndarray,
    correct_a: np.ndarray,
    pred_b: np.ndarray,
    correct_b: np.ndarray,
    label_a: str,
    label_b: str,
    n_examples: int,
    k: int = 7,
) -> list[dict]:
    """Pick diverse examples: some flipped wrong→right, some right→wrong, some always-wrong."""
    examples: list[dict] = []

    # wrong→right (improved)
    improved = np.where(~correct_a & correct_b)[0]
    rng = np.random.default_rng(42)
    if len(improved) > 0:
        pick = rng.choice(improved, min(n_examples // 2, len(improved)), replace=False)
        for idx in pick:
            examples.append(_build_one_example(idx, sample_items, prototype_items, ps_a, ps_b, pred_a, pred_b, label_a, label_b, k, tag="improved"))

    # right→wrong (regressed)
    regressed = np.where(correct_a & ~correct_b)[0]
    if len(regressed) > 0:
        pick = rng.choice(regressed, min(max(1, n_examples // 4), len(regressed)), replace=False)
        for idx in pick:
            examples.append(_build_one_example(idx, sample_items, prototype_items, ps_a, ps_b, pred_a, pred_b, label_a, label_b, k, tag="regressed"))

    # always-wrong in both
    both_wrong = np.where(~correct_a & ~correct_b)[0]
    remain = n_examples - len(examples)
    if len(both_wrong) > 0 and remain > 0:
        pick = rng.choice(both_wrong, min(remain, len(both_wrong)), replace=False)
        for idx in pick:
            examples.append(_build_one_example(idx, sample_items, prototype_items, ps_a, ps_b, pred_a, pred_b, label_a, label_b, k, tag="both_wrong"))

    return examples[:n_examples]


def _build_one_example(idx, sample_items, prototype_items, ps_a, ps_b, pred_a, pred_b, label_a, label_b, k, tag):
    item = sample_items[idx]
    true_label = next(iter(item.labels))

    def topk_info(ps, model_label):
        sorted_idx = np.argsort(-ps[idx])[:k]
        preds = []
        for pi in sorted_idx:
            plabel = next(iter(prototype_items[int(pi)].labels))
            preds.append({
                "label": plabel,
                "score": round(float(ps[idx, pi]), 4),
                "correct": plabel in item.labels,
            })
        return {"model_label": model_label, "top_preds": preds}

    return {
        "index": int(idx),
        "sample_name": item.path.name if item.path else item.key,
        "true_label": true_label,
        "tag": tag,
        "model_a": topk_info(ps_a, label_a),
        "model_b": topk_info(ps_b, label_b),
    }


# ============================= Main =========================================

def main() -> None:
    args = parse_args()

    if not HAS_MPL:
        print("WARNING: matplotlib not installed. Plots will be skipped; only text/JSON output produced.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q_type, r_type = parse_pair(args.pair)
    validate_pair_for_dataset(args.dataset, q_type, r_type)

    show = not args.no_progress
    device = torch.device(args.device)
    sample_rate = 44_100
    audio_len = default_audio_length(args.dataset) if args.audio_length is None else args.audio_length

    # ---- Load data --------------------------------------------------------
    all_items, dataset_desc = load_items(args)
    all_items = prepare_items_for_pair(args.dataset, q_type, r_type, all_items)

    # ---- Fold handling for esc50/us8k paper protocol ----------------------
    use_folds = args.protocol == "paper" and args.dataset in {"esc50", "us8k"}
    if use_folds:
        if args.fold is not None:
            folds = [args.fold]
        else:
            folds = folds_for_dataset(args.dataset)
    else:
        folds = [None]  # single pass

    # ---- Load models ------------------------------------------------------
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

    # ---- Per-fold analysis (accumulated) ----------------------------------
    all_sample_items: list[Item] = []
    all_proto_items: list[Item] = []  # same across all folds for classification datasets
    all_correct_a: list[bool] = []
    all_correct_b: list[bool] = []
    all_pred_a: list[int] = []
    all_pred_b: list[int] = []
    all_ps_a_rows: list[np.ndarray] = []
    all_ps_b_rows: list[np.ndarray] = []

    fold_metrics: list[dict] = []

    fold_iter = folds
    if show and tqdm is not None and len(folds) > 1:
        fold_iter = tqdm(folds, desc="Folds", unit="fold")

    for fold in fold_iter:
        if fold is not None:
            fold_audio = [it for it in all_items["audio"] if it.fold == fold]
            if not fold_audio:
                continue
            fold_labels = sorted({next(iter(it.labels)) for it in fold_audio})
            fold_text = [Item(key=f"class:{lb}", labels={lb}, text=lb) for lb in fold_labels]
            fold_items = {
                "text": fold_text if "text" in {q_type, r_type} else [],
                "audio": fold_audio if "audio" in {q_type, r_type} else [],
                "image": all_items["image"] if "image" in {q_type, r_type} else [],
            }
            fold_items = prepare_items_for_pair(args.dataset, q_type, r_type, fold_items)
            prefix = f"Fold {fold}"
        else:
            fold_items = all_items
            prefix = "Global"

        with torch.no_grad():
            scores_a, q_items_a, r_items_a = encode_all(
                model_a, q_type, r_type, fold_items,
                args.batch_size, device, sample_rate, audio_len,
                args.prompt_template, args.label_a, show,
            )
            scores_b, q_items_b, r_items_b = encode_all(
                model_b, q_type, r_type, fold_items,
                args.batch_size, device, sample_rate, audio_len,
                args.prompt_template, args.label_b, show,
            )

        sa_items, pa_items, ps_a, pred_a, corr_a = per_sample_predictions(scores_a, q_items_a, r_items_a, q_type, r_type)
        sb_items, pb_items, ps_b, pred_b, corr_b = per_sample_predictions(scores_b, q_items_b, r_items_b, q_type, r_type)

        acc_a = corr_a.mean() * 100
        acc_b = corr_b.mean() * 100
        improved = int((~corr_a & corr_b).sum())
        regressed = int((corr_a & ~corr_b).sum())

        fm = {
            "fold": fold,
            "n_samples": len(sa_items),
            "n_prototypes": len(pa_items),
            "improved": improved,
            "regressed": regressed,
        }
        fold_metrics.append(fm)

        # Accumulate
        all_sample_items.extend(sa_items)
        all_proto_items = pa_items  # prototypes same across folds for classification datasets
        all_correct_a.extend(corr_a.tolist())
        all_correct_b.extend(corr_b.tolist())
        all_pred_a.extend(pred_a.tolist())
        all_pred_b.extend(pred_b.tolist())
        all_ps_a_rows.append(ps_a)
        all_ps_b_rows.append(ps_b)

    # ---- Aggregate --------------------------------------------------------
    correct_a_all = np.array(all_correct_a, dtype=bool)
    correct_b_all = np.array(all_correct_b, dtype=bool)
    pred_a_all = np.array(all_pred_a)
    pred_b_all = np.array(all_pred_b)

    total_improved = int((~correct_a_all & correct_b_all).sum())
    total_regressed = int((correct_a_all & ~correct_b_all).sum())

    # ---- Confusion matrices -----------------------------------------------
    classes = sorted({next(iter(it.labels)) for it in all_proto_items})
    cm_a = build_confusion_matrix(all_sample_items, pred_a_all, all_proto_items, classes)
    cm_b = build_confusion_matrix(all_sample_items, pred_b_all, all_proto_items, classes)

    # ---- Score distributions for correct/incorrect predictions ------------
    # We can stack ps rows if prototypes are consistent across folds, otherwise use last fold
    last_ps_a = all_ps_a_rows[-1]
    last_ps_b = all_ps_b_rows[-1]
    last_n = last_ps_a.shape[0]
    last_corr_a = correct_a_all[-last_n:]
    last_corr_b = correct_b_all[-last_n:]
    last_pred_a = pred_a_all[-last_n:]
    last_pred_b = pred_b_all[-last_n:]

    score_correct_a = np.array([last_ps_a[i, last_pred_a[i]] for i in range(last_n) if last_corr_a[i]])
    score_incorrect_a = np.array([last_ps_a[i, last_pred_a[i]] for i in range(last_n) if not last_corr_a[i]])
    score_correct_b = np.array([last_ps_b[i, last_pred_b[i]] for i in range(last_n) if last_corr_b[i]])
    score_incorrect_b = np.array([last_ps_b[i, last_pred_b[i]] for i in range(last_n) if not last_corr_b[i]])

    # ---- Rank shifts (last fold) ------------------------------------------
    last_sample_items = all_sample_items[-last_n:]
    ranks_a = compute_correct_class_ranks(last_sample_items, all_proto_items, last_ps_a)
    ranks_b = compute_correct_class_ranks(last_sample_items, all_proto_items, last_ps_b)

    # ---- Top-K examples ---------------------------------------------------
    topk_examples = build_topk_examples(
        last_sample_items, all_proto_items,
        last_ps_a, last_ps_b,
        last_pred_a, last_corr_a,
        last_pred_b, last_corr_b,
        args.label_a, args.label_b,
        n_examples=args.top_k_examples,
    )

    # ---- Prediction flip table (text) -------------------------------------
    flip_lines: list[str] = []
    flip_data: list[dict] = []
    flips_wrong_to_right = np.where(~correct_a_all & correct_b_all)[0]
    flips_right_to_wrong = np.where(correct_a_all & ~correct_b_all)[0]

    rng = np.random.default_rng(42)
    for tag, indices in [("IMPROVED", flips_wrong_to_right), ("REGRESSED", flips_right_to_wrong)]:
        if len(indices) == 0:
            continue
        pick = rng.choice(indices, min(args.max_examples, len(indices)), replace=False)
        flip_lines.append(f"\n--- {tag} ({len(indices)} total, showing {len(pick)}) ---")
        for idx in pick:
            item = all_sample_items[idx]
            true_label = next(iter(item.labels))
            pred_label_a = next(iter(all_proto_items[int(pred_a_all[idx])].labels))
            pred_label_b = next(iter(all_proto_items[int(pred_b_all[idx])].labels))
            name = item.path.name if item.path else item.key
            line = f"  {name}  true={true_label}  {args.label_a}={pred_label_a}  {args.label_b}={pred_label_b}"
            flip_lines.append(line)
            flip_data.append({
                "index": int(idx), "sample": name, "true": true_label,
                f"pred_{args.label_a}": pred_label_a, f"pred_{args.label_b}": pred_label_b,
                "tag": tag.lower(),
            })

    flip_text = "\n".join(flip_lines)
    if flip_text:
        print("\nPrediction Flips:")
        print(flip_text)

    # ---- Save everything --------------------------------------------------
    # 1. Summary JSON
    summary = {
        "dataset": args.dataset,
        "pair": args.pair,
        "protocol": args.protocol,
        "fold": args.fold,
        "model_a": {"path": str(args.model_a), "label": args.label_a},
        "model_b": {"path": str(args.model_b), "label": args.label_b},
        "total_samples": len(all_sample_items),
        "total_prototypes": len(all_proto_items),
        "improved_count": total_improved,
        "regressed_count": total_regressed,
        "net_gain": total_improved - total_regressed,
        "fold_metrics": fold_metrics,
        "flip_examples": flip_data,
        "topk_examples": topk_examples,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")

    # 2. Flip table text
    flip_path = output_dir / "prediction_flips.txt"
    with open(flip_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis: {args.label_a} vs {args.label_b}\n")
        f.write(f"Dataset: {dataset_desc}\nPair: {q_type} -> {r_type}\n")
        f.write(f"Samples: {len(all_sample_items)}\n")
        f.write(f"Improved: {total_improved}  Regressed: {total_regressed}\n")
        f.write(flip_text + "\n")
    print(f"Saved flips: {flip_path}")

    # 3. Plots
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_confusion_matrix(cm_a, classes, f"Confusion Matrix — {args.label_a}",
                              output_dir / "confusion_matrix_a.png", args.dpi)
        plot_confusion_matrix(cm_b, classes, f"Confusion Matrix — {args.label_b}",
                              output_dir / "confusion_matrix_b.png", args.dpi)
        plot_score_distributions(score_correct_a, score_incorrect_a,
                                 f"Score Distribution — {args.label_a}",
                                 output_dir / "score_dist_a.png", args.dpi)
        plot_score_distributions(score_correct_b, score_incorrect_b,
                                 f"Score Distribution — {args.label_b}",
                                 output_dir / "score_dist_b.png", args.dpi)
        plot_rank_shift(ranks_a, ranks_b, args.label_a, args.label_b,
                        output_dir / "rank_shift.png", args.dpi)
        if topk_examples:
            plot_topk_comparison(topk_examples, output_dir / "topk_examples.png", args.dpi)
    else:
        print("\n[Skip] matplotlib not available — no plots generated.")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
