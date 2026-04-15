
#!/usr/bin/env python3
from __future__ import annotations

def mean_average_precision(scores: np.ndarray, rel: np.ndarray) -> float:
    """
    Compute mean Average Precision (mAP) for retrieval.
    Each query's AP is computed, then averaged.
    rel: binary relevance matrix (queries x results)
    scores: similarity matrix (queries x results)
    """
    num_queries = scores.shape[0]
    APs = []
    for i in range(num_queries):
        # Sort results for this query by descending score
        sorted_idx = np.argsort(-scores[i])
        relevant = rel[i][sorted_idx]
        num_relevant = int(np.sum(relevant))
        if num_relevant == 0:
            continue
        precisions = []
        num_hits = 0
        for rank, is_rel in enumerate(relevant, 1):
            if is_rel:
                num_hits += 1
                precisions.append(num_hits / rank)
        AP = np.mean(precisions) if precisions else 0.0
        APs.append(AP)
    return float(np.mean(APs)) if APs else float('nan')

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from table4 import AudioCLIP
from table4 import Item
from table4 import audio_transform
from table4 import default_audio_length
from table4 import encode_audio
from table4 import encode_image
from table4 import encode_text
from table4 import folds_for_dataset
from table4 import image_transform
from table4 import load_esc50
from table4 import load_us8k
from table4 import pair_logits
from table4 import relevance
from table4 import resolve_checkpoint_path


PAIR_CHOICES = [
    "audio-text",
    "text-audio",
    "image-text",
    "text-image",
    "audio-image",
    "image-audio",
]

DATASET_CHOICES = ["esc50", "us8k", "flickr8k", "coco2014", "spokencoco", "coco"]
K_VALUES = [1, 5, 10]

DATASET_MODALITIES = {
    "esc50": {"audio", "text"},
    "us8k": {"audio", "text"},
    "flickr8k": {"image", "text"},
    "coco2014": {"image", "text"},
    "spokencoco": {"audio", "text"},
    "coco": {"audio", "image", "text"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot AudioCLIP evaluation across ESC-50, US8K, Flickr8k, COCO2014, SpokenCOCO, and combined COCO.")
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--pair", required=True, choices=PAIR_CHOICES, help="Query-result direction, e.g. audio-text.")
    p.add_argument("--model-path", required=True, type=Path)

    p.add_argument("--dataset-root", type=Path, default=None, help="Root for esc50/us8k/flickr8k/coco2014/spokencoco.")
    p.add_argument("--coco-image-root", type=Path, default=None, help="COCO2014 root (required when --dataset coco).")
    p.add_argument("--coco-audio-root", type=Path, default=None, help="SpokenCOCO root (required when --dataset coco).")
    p.add_argument(
        "--coco-text-source",
        choices=["coco2014", "spokencoco", "union"],
        default="coco2014",
        help="Text source for --dataset coco.",
    )

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-length", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--protocol",
        choices=["paper", "global"],
        default="paper",
        help="paper: fold-wise averaging for esc50/us8k, global otherwise.",
    )
    p.add_argument("--fold", type=int, default=None, help="Optional fold for esc50/us8k paper protocol.")
    p.add_argument("--prompt-template", default="{}", help="Template for text, e.g. 'a sound of {}' or 'a photo of {}'.")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output (tqdm bars and per-fold logs).",
    )
    return p.parse_args()


def normalize_labels(labels: Iterable[str]) -> set[str]:
    return {str(lb).strip() for lb in labels if str(lb).strip()}


def parse_pair(pair: str) -> tuple[str, str]:
    q_type, r_type = pair.split("-", 1)
    if q_type == r_type:
        raise ValueError("Query and result modalities must be different.")
    return q_type, r_type


def top1_accuracy(scores: np.ndarray, rel: np.ndarray) -> float:
    return retrieval_hit_at_k(scores, rel, k=1)


def retrieval_r_at_1(scores: np.ndarray, rel: np.ndarray) -> float:
    return retrieval_r_at_k(scores, rel, k=1)


def retrieval_hit_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    if scores.ndim != 2 or rel.ndim != 2:
        raise ValueError("scores and rel must be 2D matrices")
    if scores.shape != rel.shape:
        raise ValueError("scores and rel must have identical shape")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return float("nan")

    k_eff = max(1, min(k, scores.shape[1]))
    topk_idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    row_idx = np.arange(scores.shape[0])[:, None]
    hit = np.any(rel[row_idx, topk_idx], axis=1)
    return float(np.mean(hit.astype(np.float32)))


def retrieval_r_at_k(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    return retrieval_hit_at_k(scores.T, rel.T, k)


def supervised_classification_accuracy_any(
    scores: np.ndarray,
    q_items: list[Item],
    r_items: list[Item],
) -> float | None:
    """
    Supervised-like class accuracy for any query/result modality pair.

    Result items are grouped by label into classes, then class score for each query
    is the mean score across class members. The predicted class is argmax over these
    class scores; prediction is correct if predicted label overlaps query labels.
    """
    if scores.size == 0 or not q_items or not r_items:
        return None

    label_to_result_ids: dict[str, list[int]] = defaultdict(list)
    for ri, item in enumerate(r_items):
        for lb in item.labels:
            label_to_result_ids[lb].append(ri)

    if not label_to_result_ids:
        return None

    class_labels = sorted(label_to_result_ids.keys())
    nq = scores.shape[0]
    best_scores = np.full((nq,), -np.inf, dtype=np.float32)
    best_label_ids = np.full((nq,), -1, dtype=np.int32)

    # Compute class-level scores without materializing a large [nq x n_classes] matrix.
    for ci, label in enumerate(class_labels):
        idxs = label_to_result_ids[label]
        cls_scores = scores[:, idxs].mean(axis=1)
        update = cls_scores > best_scores
        best_scores[update] = cls_scores[update]
        best_label_ids[update] = ci

    if np.any(best_label_ids < 0):
        return None

    correct = np.fromiter(
        (
            float(class_labels[int(best_label_ids[i])] in q_items[i].labels)
            for i in range(nq)
        ),
        dtype=np.float32,
        count=nq,
    )
    if correct.size == 0:
        return None
    return float(correct.mean())


def sample_classification_accuracy(
    scores: np.ndarray,
    q_items: list[Item],
    r_items: list[Item],
    q_type: str,
    r_type: str,
) -> float | None:
    """
    Per-sample classification accuracy.

    For any pair involving text (as class prototypes), classify each non-text
    sample by the text prototype with highest similarity and check correctness.
    Works for both audio-text and text-audio pair directions.

    This matches the 'Val. Acc. Eval.' metric from the fine-tuning pipeline:
    for each audio/image sample, predict its class label.
    """
    if "text" not in {q_type, r_type}:
        return None

    if q_type == "text":
        prototype_items = q_items
        sample_items = r_items
        per_sample_scores = scores.T
    else:
        prototype_items = r_items
        sample_items = q_items
        per_sample_scores = scores

    if per_sample_scores.size == 0 or not sample_items or not prototype_items:
        return None

    pred_indices = np.argmax(per_sample_scores, axis=1)
    correct = sum(
        1
        for i, item in enumerate(sample_items)
        if item.labels & prototype_items[int(pred_indices[i])].labels
    )
    return correct / len(sample_items) if sample_items else None


def relevance_with_progress(
    queries: list[Item],
    results: list[Item],
    show_pair_progress: bool,
    progress_prefix: str = "",
) -> np.ndarray:
    rel = np.zeros((len(queries), len(results)), dtype=np.bool_)

    total_pairs = len(queries) * len(results)
    pair_bar = None
    if show_pair_progress and tqdm is not None and total_pairs > 0:
        pair_bar = tqdm(total=total_pairs, desc=f"{progress_prefix}Pair relevance", unit="pair")

    for qi, q in enumerate(queries):
        q_labels = q.labels
        for ri, r in enumerate(results):
            rel[qi, ri] = bool(q_labels & r.labels)
        if pair_bar is not None:
            pair_bar.update(len(results))

    if pair_bar is not None:
        pair_bar.close()

    return rel


def extract_coco_image_id(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    m = re.search(r"(\d{6,12})", stem)
    if m:
        return str(int(m.group(1)))
    return stem


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_flickr_image_dir(root: Path) -> Path:
    candidates = [
        root / "Flicker8k_Dataset",
        root / "Flickr8k_Dataset",
        root / "Flickr_Data" / "Flickr_Data" / "Images",
        root / "Flickr_Data" / "Images",
        root / "Images",
        root / "images",
        root,
    ]
    found = first_existing(candidates)
    if found is None:
        recursive_dirs = sorted(
            [
                p
                for p in root.rglob("*")
                if p.is_dir() and p.name.lower() in {"images", "flickr8k_dataset", "flicker8k_dataset"}
            ]
        )
        if recursive_dirs:
            found = recursive_dirs[0]
    if found is None:
        raise FileNotFoundError(f"Could not find Flickr8k image directory under: {root}")
    return found


def resolve_flickr_caption_file(root: Path) -> Path:
    candidates = [
        root / "Flickr8k.token.txt",
        root / "captions.txt",
        root / "Flickr8k_text" / "Flickr8k.token.txt",
        root / "Flickr8k_text" / "captions.txt",
        root / "Flickr_TextData" / "Flickr8k.token.txt",
        root / "Flickr_TextData" / "captions.txt",
        root / "flickr8ktextfiles" / "Flickr8k.token.txt",
        root / "flickr8ktextfiles" / "captions.txt",
        root / "Flickr_Data" / "Flickr_Data" / "Flickr_TextData" / "Flickr8k.token.txt",
        root / "Flickr_Data" / "Flickr_Data" / "flickr8ktextfiles" / "Flickr8k.token.txt",
    ]
    found = first_existing(candidates)
    if found is None:
        names = ["Flickr8k.token.txt", "captions.txt"]
        recursive_candidates: list[Path] = []
        for name in names:
            recursive_candidates.extend(root.rglob(name))
        recursive_candidates = sorted(set(recursive_candidates))
        if recursive_candidates:
            found = recursive_candidates[0]
    if found is None:
        raise FileNotFoundError(
            f"Could not find Flickr8k caption file under: {root}. "
            "Expected one of: Flickr8k.token.txt or captions.txt (possibly in nested folders)."
        )
    return found


def build_flickr_image_index(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts and path.name not in index:
            index[path.name] = path
    return index


def load_flickr8k(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    image_dir = resolve_flickr_image_dir(root)
    caption_file = resolve_flickr_caption_file(root)
    image_index: dict[str, Path] | None = None

    image_to_captions: dict[str, list[str]] = defaultdict(list)

    if caption_file.name.lower() == "captions.txt":
        with caption_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() in {"image", "filename"}:
                    continue
                if len(row) < 2:
                    continue
                image_name = row[0].strip()
                caption = row[1].strip()
                if image_name and caption:
                    image_to_captions[image_name].append(caption)
    else:
        with caption_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if "\t" in line:
                    lhs, caption = line.split("\t", 1)
                elif "," in line:
                    lhs, caption = line.split(",", 1)
                else:
                    continue
                image_name = lhs.split("#", 1)[0].strip()
                caption = caption.strip()
                if image_name and caption:
                    image_to_captions[image_name].append(caption)

    image_items: list[Item] = []
    text_items: list[Item] = []

    for image_name, captions in image_to_captions.items():
        image_path = image_dir / image_name
        if not image_path.exists():
            if image_index is None:
                image_index = build_flickr_image_index(root)
            image_path = image_index.get(image_name)
            if image_path is None or not image_path.exists():
                continue

        image_id = Path(image_name).stem
        image_items.append(Item(key=f"img:{image_name}", labels={image_id}, path=image_path))

        for cap_idx, caption in enumerate(captions):
            text_items.append(Item(key=f"txt:{image_id}:{cap_idx}", labels={image_id}, text=caption))

    if not image_items or not text_items:
        raise RuntimeError("No valid Flickr8k image/text pairs were found.")

    return text_items, [], image_items


FLICKR_SPLIT_SEED = 42
FLICKR_TEST_RATIO = 0.30


def flickr_train_test_split(
    text_items: list[Item],
    image_items: list[Item],
    seed: int = FLICKR_SPLIT_SEED,
    test_ratio: float = FLICKR_TEST_RATIO,
) -> tuple[list[Item], list[Item], list[Item], list[Item]]:
    """Split Flickr8k items by image_id into train (70%) and test (30%).

    Returns (train_text, train_image, test_text, test_image).
    """
    image_ids = sorted({next(iter(it.labels)) for it in image_items})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n_test = max(1, int(len(image_ids) * test_ratio))
    test_ids = set(image_ids[:n_test])
    train_ids = set(image_ids[n_test:])

    train_text = [it for it in text_items if it.labels & train_ids]
    test_text = [it for it in text_items if it.labels & test_ids]
    train_image = [it for it in image_items if it.labels & train_ids]
    test_image = [it for it in image_items if it.labels & test_ids]
    return train_text, train_image, test_text, test_image


def coco_image_path_candidates(root: Path, file_name: str) -> list[Path]:
    return [
        root / file_name,
        root / "train2014" / file_name,
        root / "val2014" / file_name,
        root / "images" / "train2014" / file_name,
        root / "images" / "val2014" / file_name,
    ]


def coco_annotation_files(root: Path) -> list[Path]:
    candidates = [
        root / "annotations" / "captions_train2014.json",
        root / "annotations" / "captions_val2014.json",
        root / "captions" / "captions_train2014.json",
        root / "captions" / "captions_val2014.json",
        root / "captions_train2014.json",
        root / "captions_val2014.json",
    ]
    found = [p for p in candidates if p.exists()]
    if found:
        return found

    recursive: list[Path] = []
    for name in ["captions_train2014.json", "captions_val2014.json"]:
        recursive.extend(root.rglob(name))

    # Keep deterministic order and unique paths.
    return sorted(set(recursive))


def coco_roots_from_annotation(ann_path: Path) -> list[Path]:
    parent = ann_path.parent
    if parent.name in {"annotations", "captions"}:
        base = parent.parent
    else:
        base = parent

    roots = [base]
    nested = base / "COCO2014"
    if nested.exists():
        roots.append(nested)
    return roots


def coco_image_name_index(search_roots: list[Path]) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    index: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                key = p.name.lower()
                if key not in index:
                    index[key] = p
    return index


def load_coco2014(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    ann_files = coco_annotation_files(root)
    if not ann_files:
        raise FileNotFoundError(f"COCO2014 captions annotations not found under: {root}")

    image_id_to_path: dict[str, Path] = {}
    image_id_to_captions: dict[str, list[str]] = defaultdict(list)

    for ann_path in ann_files:
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        search_roots = [root]
        for candidate_root in coco_roots_from_annotation(ann_path):
            if candidate_root not in search_roots:
                search_roots.append(candidate_root)

        image_index: dict[str, Path] | None = None

        images = data.get("images", [])
        for row in images:
            image_id = str(row.get("id"))
            file_name = str(row.get("file_name", "")).strip()
            if not image_id or not file_name:
                continue

            all_candidates: list[Path] = []
            for search_root in search_roots:
                all_candidates.extend(coco_image_path_candidates(search_root, file_name))

            found_path = first_existing(all_candidates)
            if found_path is None:
                if image_index is None:
                    image_index = coco_image_name_index(search_roots)

                name_keys = [Path(file_name).name.lower(), file_name.lower()]
                for nk in name_keys:
                    found_path = image_index.get(nk)
                    if found_path is not None:
                        break

            if found_path is not None:
                image_id_to_path[image_id] = found_path

        annotations = data.get("annotations", [])
        for row in annotations:
            image_id = str(row.get("image_id"))
            caption = str(row.get("caption", "")).strip()
            if image_id and caption:
                image_id_to_captions[image_id].append(caption)

    image_items: list[Item] = []
    text_items: list[Item] = []

    for image_id, image_path in image_id_to_path.items():
        image_items.append(Item(key=f"img:{image_id}", labels={image_id}, path=image_path))

    for image_id, captions in image_id_to_captions.items():
        for cap_idx, caption in enumerate(captions):
            text_items.append(Item(key=f"txt:{image_id}:{cap_idx}", labels={image_id}, text=caption))

    if not image_items or not text_items:
        raise RuntimeError("No valid COCO2014 image/text items were found.")

    return text_items, [], image_items


def spokencoco_metadata_files(root: Path) -> list[Path]:
    candidates = [
        root / "SpokenCOCO_train.json",
        root / "SpokenCOCO_val.json",
        root / "SpokenCOCO" / "SpokenCOCO_train.json",
        root / "SpokenCOCO" / "SpokenCOCO_val.json",
        root / "metadata" / "SpokenCOCO_train.json",
        root / "metadata" / "SpokenCOCO_val.json",
    ]
    return [p for p in candidates if p.exists()]


def resolve_spokencoco_audio_path(root: Path, rel: str) -> Path | None:
    rel_path = Path(rel)
    candidates = [
        rel_path,
        root / rel_path,
        root / "SpokenCOCO" / rel_path,
        root / "wavs" / rel_path,
        root / "audio" / rel_path,
    ]
    return first_existing(candidates)


def load_spokencoco(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    meta_files = spokencoco_metadata_files(root)
    if not meta_files:
        raise FileNotFoundError(f"SpokenCOCO metadata json files not found under: {root}")

    audio_items: list[Item] = []
    text_items: list[Item] = []

    for meta_path in meta_files:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("data", [])
        for entry_idx, entry in enumerate(entries):
            image_rel = str(entry.get("image", "")).strip()
            image_id = extract_coco_image_id(image_rel) if image_rel else f"entry_{entry_idx}"

            captions = entry.get("captions", [])
            if isinstance(captions, dict):
                captions = [captions]

            for cap_idx, cap in enumerate(captions):
                wav_rel = str(cap.get("wav") or cap.get("audio") or cap.get("speech") or "").strip()
                text = str(cap.get("text") or cap.get("utt") or cap.get("caption") or "").strip()

                if wav_rel:
                    wav_path = resolve_spokencoco_audio_path(root, wav_rel)
                    if wav_path is not None:
                        audio_items.append(
                            Item(
                                key=f"aud:{image_id}:{entry_idx}:{cap_idx}",
                                labels={image_id},
                                path=wav_path,
                            )
                        )

                if text:
                    text_items.append(Item(key=f"txt:{image_id}:{entry_idx}:{cap_idx}", labels={image_id}, text=text))

    if not audio_items or not text_items:
        raise RuntimeError("No valid SpokenCOCO audio/text items were found.")

    return text_items, audio_items, []


def select_text_source(
    coco_text: list[Item],
    spoken_text: list[Item],
    source: str,
) -> list[Item]:
    if source == "coco2014":
        return coco_text
    if source == "spokencoco":
        return spoken_text

    merged = list(coco_text)
    seen = {it.key for it in merged}
    for it in spoken_text:
        if it.key not in seen:
            merged.append(it)
            seen.add(it.key)
    return merged


def filter_items_by_labels(items: list[Item], keep_labels: set[str]) -> list[Item]:
    return [it for it in items if bool(it.labels & keep_labels)]


def build_single_dataset(dataset: str, dataset_root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    if dataset == "esc50":
        return load_esc50(dataset_root)
    if dataset == "us8k":
        return load_us8k(dataset_root)
    if dataset == "flickr8k":
        return load_flickr8k(dataset_root)
    if dataset == "coco2014":
        return load_coco2014(dataset_root)
    if dataset == "spokencoco":
        return load_spokencoco(dataset_root)
    raise ValueError(f"Unsupported dataset for single-root loading: {dataset}")


def build_coco_combined(
    image_root: Path,
    audio_root: Path,
    text_source: str,
) -> tuple[list[Item], list[Item], list[Item]]:
    coco_text, _, coco_images = load_coco2014(image_root)
    spoken_text, spoken_audio, _ = load_spokencoco(audio_root)

    text_items = select_text_source(coco_text, spoken_text, text_source)
    return text_items, spoken_audio, coco_images


def validate_pair_for_dataset(dataset: str, q_type: str, r_type: str) -> None:
    allowed = DATASET_MODALITIES[dataset]
    if q_type not in allowed or r_type not in allowed:
        raise ValueError(
            f"Pair {q_type}-{r_type} is invalid for dataset {dataset}. "
            f"Allowed modalities: {sorted(allowed)}"
        )


def prepare_items_for_pair(
    dataset: str,
    q_type: str,
    r_type: str,
    all_items: dict[str, list[Item]],
) -> dict[str, list[Item]]:
    out = {"text": list(all_items.get("text", [])), "audio": list(all_items.get("audio", [])), "image": list(all_items.get("image", []))}

    labels_q = {lb for it in out[q_type] for lb in it.labels}
    labels_r = {lb for it in out[r_type] for lb in it.labels}
    shared = labels_q & labels_r

    if not shared:
        raise RuntimeError(
            f"No shared labels between {q_type} and {r_type}. "
            "Check metadata alignment and dataset roots."
        )

    out[q_type] = filter_items_by_labels(out[q_type], shared)
    out[r_type] = filter_items_by_labels(out[r_type], shared)

    # Keep support modalities restricted to same label universe for consistent relevance mapping.
    out["text"] = filter_items_by_labels(out["text"], shared) if out["text"] else []
    out["audio"] = filter_items_by_labels(out["audio"], shared) if out["audio"] else []
    out["image"] = filter_items_by_labels(out["image"], shared) if out["image"] else []

    return out


def evaluate_once_accuracy(
    model,
    q_type: str,
    r_type: str,
    all_items: dict[str, list[Item]],
    batch_size: int,
    device: torch.device,
    sample_rate: int,
    audio_len: int,
    prompt_template: str,
    show_progress: bool,
    progress_prefix: str = "",
    show_pair_progress: bool = False,
) -> tuple[dict[int, float], dict[int, float], float | None, int, int]:
    q_items = all_items[q_type]
    r_items = all_items[r_type]
    if not q_items or not r_items:
        raise ValueError(f"No items for setting {q_type}->{r_type}")

    img_tf = image_transform()
    aud_tf = audio_transform(audio_len)

    feats: dict[str, torch.Tensor] = {}
    if q_type == "text":
        if show_progress:
            print(f"{progress_prefix}Encoding query-text features...")
        feats["text"] = encode_text(model, all_items["text"], prompt_template, batch_size)
    elif r_type == "text":
        if show_progress:
            print(f"{progress_prefix}Encoding result-text features...")
        feats["text"] = encode_text(model, all_items["text"], prompt_template, batch_size)

    if q_type == "audio":
        if show_progress:
            print(f"{progress_prefix}Encoding query-audio features...")
        feats["audio"] = encode_audio(model, all_items["audio"], batch_size, device, sample_rate, aud_tf)
    elif r_type == "audio":
        if show_progress:
            print(f"{progress_prefix}Encoding result-audio features...")
        feats["audio"] = encode_audio(model, all_items["audio"], batch_size, device, sample_rate, aud_tf)

    if q_type == "image":
        if show_progress:
            print(f"{progress_prefix}Encoding query-image features...")
        feats["image"] = encode_image(model, all_items["image"], batch_size, device, img_tf)
    elif r_type == "image":
        if show_progress:
            print(f"{progress_prefix}Encoding result-image features...")
        feats["image"] = encode_image(model, all_items["image"], batch_size, device, img_tf)

    def select(mod: str, subset: list[Item]) -> torch.Tensor:
        key_to_idx = {it.key: i for i, it in enumerate(all_items[mod])}
        ids = [key_to_idx[it.key] for it in subset]
        return feats[mod][ids]

    qf = select(q_type, q_items)
    rf = select(r_type, r_items)
    scores = pair_logits(model, q_type, r_type, qf, rf).numpy()
    rel = relevance_with_progress(
        q_items,
        r_items,
        show_pair_progress=show_pair_progress,
        progress_prefix=progress_prefix,
    )

    retrieval_hit_by_k = {k: retrieval_hit_at_k(scores, rel, k) for k in K_VALUES}
    retrieval_r_by_k = {k: retrieval_r_at_k(scores, rel, k) for k in K_VALUES}
    supervised_acc = supervised_classification_accuracy_any(scores, q_items, r_items)
    map_score = mean_average_precision(scores, rel)
    sample_cls_acc = sample_classification_accuracy(scores, q_items, r_items, q_type, r_type)

    return retrieval_hit_by_k, retrieval_r_by_k, supervised_acc, len(q_items), len(r_items), map_score, sample_cls_acc


def main() -> None:
    args = parse_args()

    # --- Auto-detect fold from model filename if not provided ---
    if args.fold is None and args.dataset in {"esc50", "us8k"} and "fold" in str(args.model_path):
        m = re.search(r"fold(\d+)", str(args.model_path))
        if m:
            args.fold = int(m.group(1))
            print(f"[Info] Auto-detected --fold={args.fold} from model filename.")
    q_type, r_type = parse_pair(args.pair)

    validate_pair_for_dataset(args.dataset, q_type, r_type)

    if args.dataset == "coco":
        if args.coco_image_root is None or args.coco_audio_root is None:
            raise ValueError("--dataset coco requires both --coco-image-root and --coco-audio-root")
        image_root = args.coco_image_root.resolve()
        audio_root = args.coco_audio_root.resolve()
        if not image_root.exists():
            raise FileNotFoundError(f"COCO image root not found: {image_root}")
        if not audio_root.exists():
            raise FileNotFoundError(f"SpokenCOCO root not found: {audio_root}")

        text_items, audio_items, image_items = build_coco_combined(image_root, audio_root, args.coco_text_source)
        dataset_desc = f"coco (image={image_root}, audio={audio_root})"
    else:
        if args.dataset_root is None:
            raise ValueError(f"--dataset {args.dataset} requires --dataset-root")
        dataset_root = args.dataset_root.resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        text_items, audio_items, image_items = build_single_dataset(args.dataset, dataset_root)
        dataset_desc = f"{args.dataset} ({dataset_root})"

    # For Flickr8k: split 70/30 and evaluate only on 30% test set
    if args.dataset == "flickr8k":
        _train_text, _train_image, test_text, test_image = flickr_train_test_split(text_items, image_items)
        text_items = test_text
        image_items = test_image
        print(f"[Info] Flickr8k: evaluating on 30% test split ({len(test_image)} images, seed={FLICKR_SPLIT_SEED})")

    all_items = {"text": text_items, "audio": audio_items, "image": image_items}
    all_items = prepare_items_for_pair(args.dataset, q_type, r_type, all_items)

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_path = resolve_checkpoint_path(model_path)

    device = torch.device(args.device)
    model = AudioCLIP(pretrained=str(model_path)).to(device)
    model.eval()

    sample_rate = 44_100
    audio_len = default_audio_length(args.dataset) if args.audio_length is None else args.audio_length
    show_progress = not args.no_progress

    use_paper_folds = args.protocol == "paper" and args.dataset in {"esc50", "us8k"}
    if use_paper_folds:
        folds = [args.fold] if args.fold is not None else folds_for_dataset(args.dataset)
        if not folds:
            raise ValueError(f"No folds available for dataset {args.dataset}")

        fold_hit_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
        fold_r_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
        fold_sup_accs: list[float] = []
        fold_nq: list[int] = []
        fold_nr: list[int] = []
        fold_map: list[float] = []
        fold_sample_cls: list[float] = []

        fold_iter = folds
        if show_progress and tqdm is not None:
            fold_iter = tqdm(folds, desc="Evaluating folds", unit="fold")

        for fold in fold_iter:
            fold_audio = [it for it in all_items["audio"] if it.fold == fold]
            if not fold_audio:
                if show_progress:
                    print(f"[Fold {fold}] skipped (no items)")
                continue

            fold_labels = sorted({next(iter(it.labels)) for it in fold_audio})
            fold_text = [Item(key=f"class:{lb}", labels={lb}, text=lb) for lb in fold_labels]

            fold_items = {
                "text": fold_text if "text" in {q_type, r_type} else [],
                "audio": fold_audio if "audio" in {q_type, r_type} else [],
                "image": all_items["image"] if "image" in {q_type, r_type} else [],
            }

            fold_items = prepare_items_for_pair(args.dataset, q_type, r_type, fold_items)
            hit_by_k, r_by_k, sup_acc, nq, nr, map_score, sample_cls = evaluate_once_accuracy(
                model,
                q_type,
                r_type,
                fold_items,
                args.batch_size,
                device,
                sample_rate,
                audio_len,
                args.prompt_template,
                show_progress=show_progress,
                progress_prefix=f"[Fold {fold}] ",
                show_pair_progress=False,
            )
            for k in K_VALUES:
                fold_hit_by_k[k].append(hit_by_k[k])
                fold_r_by_k[k].append(r_by_k[k])
            if sup_acc is not None:
                fold_sup_accs.append(sup_acc)
            if sample_cls is not None:
                fold_sample_cls.append(sample_cls)
            fold_nq.append(nq)
            fold_nr.append(nr)
            fold_map.append(map_score)
            if show_progress:
                hit_txt = " ".join([f"retrieval_hit@{k}={hit_by_k[k] * 100.0:.2f}" for k in K_VALUES])
                r_txt = " ".join([f"retrieval_r@{k}={r_by_k[k] * 100.0:.2f}" for k in K_VALUES])
                line = f"[Fold {fold}] queries={nq} results={nr} {hit_txt} {r_txt}"
                if sup_acc is not None:
                    line += f" supervised_cls_acc={sup_acc * 100.0:.2f}"
                if sample_cls is not None:
                    line += f" sample_cls_acc={sample_cls * 100.0:.2f}"
                line += f" mAP={map_score * 100.0:.2f}"
                print(line)

        if not fold_hit_by_k[1]:
            raise RuntimeError("No fold metrics computed. Check fold metadata and modality pair.")

        retrieval_hit_by_k = {k: float(np.mean(fold_hit_by_k[k])) for k in K_VALUES}
        retrieval_r_by_k = {k: float(np.mean(fold_r_by_k[k])) for k in K_VALUES}
        supervised_acc = float(np.mean(fold_sup_accs)) if fold_sup_accs else None
        sample_cls_acc = float(np.mean(fold_sample_cls)) if fold_sample_cls else None
        nq = int(np.mean(fold_nq))
        nr = int(np.mean(fold_nr))
        map_score = float(np.mean(fold_map)) if fold_map else float('nan')
        mode_info = f"paper-fold-wise ({len(fold_hit_by_k[1])} folds)"
    else:
        retrieval_hit_by_k, retrieval_r_by_k, supervised_acc, nq, nr, map_score, sample_cls_acc = evaluate_once_accuracy(
            model,
            q_type,
            r_type,
            all_items,
            args.batch_size,
            device,
            sample_rate,
            audio_len,
            args.prompt_template,
            show_progress=show_progress,
            progress_prefix="[Global] ",
            show_pair_progress=show_progress,
        )
        mode_info = "global"

    model_label = "audio-head" if "partial" in model_path.name.lower() else "full-model"

    print("Zero-shot setting")
    print(f"  dataset: {dataset_desc}")
    print(f"  pair: {q_type} -> {r_type}")
    print(f"  protocol: {mode_info}")
    print(f"  model: {model_path}")
    print(f"  model-label: {model_label}")
    print(f"  queries(avg): {nq}")
    print(f"  results(avg): {nr}")
    print()
    print("Scores")
    print(f"  Retrieval Hit@1 / P@1 (query->result): {retrieval_hit_by_k[1] * 100.0:.2f}")
    print(f"  Retrieval Hit@5 / P@5 (query->result): {retrieval_hit_by_k[5] * 100.0:.2f}")
    print(f"  Retrieval Hit@10 / P@10 (query->result): {retrieval_hit_by_k[10] * 100.0:.2f}")
    print(f"  Retrieval R@1 (result->query): {retrieval_r_by_k[1] * 100.0:.2f}")
    print(f"  Retrieval R@5 (result->query): {retrieval_r_by_k[5] * 100.0:.2f}")
    print(f"  Retrieval R@10 (result->query): {retrieval_r_by_k[10] * 100.0:.2f}")
    print(f"  mAP (mean Average Precision): {map_score * 100.0:.2f}")
    if supervised_acc is None:
        print("  Supervised classification accuracy (result-label classes): N/A")
    else:
        print(f"  Supervised classification accuracy (result-label classes): {supervised_acc * 100.0:.2f}")
    if sample_cls_acc is None:
        print("  Per-sample classification accuracy: N/A")
    else:
        print(f"  Per-sample classification accuracy: {sample_cls_acc * 100.0:.2f}")

    if math.isnan(retrieval_hit_by_k[1]):
        raise RuntimeError("Retrieval Hit@1 is NaN. Check dataset integrity.")


if __name__ == "__main__":
    main()
