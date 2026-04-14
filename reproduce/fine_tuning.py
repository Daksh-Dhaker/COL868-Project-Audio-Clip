#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from table4 import AudioCLIP
from table4 import Item
from table4 import audio_transform
from table4 import default_audio_length
from table4 import image_transform
from table4 import load_audio
from table4 import load_image
from table4 import resolve_checkpoint_path

from zero_shot_eval import DATASET_CHOICES
from zero_shot_eval import DATASET_MODALITIES
from zero_shot_eval import K_VALUES
from zero_shot_eval import build_coco_combined
from zero_shot_eval import build_single_dataset
from zero_shot_eval import evaluate_once_accuracy
from zero_shot_eval import parse_pair
from zero_shot_eval import prepare_items_for_pair
from zero_shot_eval import validate_pair_for_dataset


ACC_PATTERN = re.compile(r"Val\. Acc\. Eval\.:\s*([0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tuning runner for AudioCLIP with the same core CLI style as zero_shot_eval.py. "
            "ESC-50/US8K can run in exact main.py-compatible mode."
        )
    )

    # Keep these aligned with zero_shot_eval.py.
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--pair", required=True, help="Query-result direction, e.g. audio-text.")
    p.add_argument("--model-path", required=True, type=Path)

    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--coco-image-root", type=Path, default=None)
    p.add_argument("--coco-audio-root", type=Path, default=None)
    p.add_argument("--coco-text-source", choices=["coco2014", "spokencoco", "union"], default="coco2014")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-length", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--protocol", choices=["paper", "global"], default="paper")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--prompt-template", default="{}")
    p.add_argument("--no-progress", action="store_true")

    # Fine-tuning options.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("reproduce") / "outputs" / "fine_tuning")
    p.add_argument(
        "--trainer-mode",
        choices=["auto", "exact", "generic"],
        default="auto",
        help="auto: exact for esc50/us8k, generic otherwise.",
    )

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_items_from_args(args: argparse.Namespace) -> tuple[dict[str, list[Item]], str]:
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

    all_items = {"text": text_items, "audio": audio_items, "image": image_items}
    return all_items, dataset_desc


class PairDataset(tud.Dataset):
    def __init__(
        self,
        query_items: list[Item],
        result_items: list[Item],
        seed: int,
        max_pairs: int | None = None,
    ):
        super().__init__()
        self.rng = random.Random(seed)

        self.label_to_result_ids: dict[str, list[int]] = {}
        for ridx, it in enumerate(result_items):
            for lb in it.labels:
                self.label_to_result_ids.setdefault(lb, []).append(ridx)

        self.query_items = [
            q for q in query_items if any(lb in self.label_to_result_ids for lb in q.labels)
        ]
        self.result_items = result_items

        if max_pairs is not None and max_pairs > 0 and len(self.query_items) > max_pairs:
            self.query_items = self.rng.sample(self.query_items, k=max_pairs)

        if not self.query_items:
            raise RuntimeError("No trainable query items with matching result labels.")

    def __len__(self) -> int:
        return len(self.query_items)

    def __getitem__(self, index: int) -> tuple[Item, Item]:
        q = self.query_items[index]
        labels = [lb for lb in q.labels if lb in self.label_to_result_ids]
        if not labels:
            raise RuntimeError("Query has no matching labels in result set.")

        chosen_label = self.rng.choice(labels)
        ridx = self.rng.choice(self.label_to_result_ids[chosen_label])
        r = self.result_items[ridx]
        return q, r


def collate_pairs(batch: list[tuple[Item, Item]]) -> tuple[list[Item], list[Item]]:
    q_items = [x[0] for x in batch]
    r_items = [x[1] for x in batch]
    return q_items, r_items


def text_of_item(item: Item, prompt_template: str) -> str:
    if item.text is not None and item.text.strip():
        return prompt_template.format(item.text)
    return prompt_template.format(next(iter(item.labels)))


def build_modal_payload(
    items: list[Item],
    modality: str,
    device: torch.device,
    sample_rate: int,
    aud_tfm,
    img_tfm,
    prompt_template: str,
):
    if modality == "text":
        return [[text_of_item(it, prompt_template)] for it in items]
    if modality == "audio":
        waves = [load_audio(it.path, sample_rate, aud_tfm) for it in items]
        return torch.stack(waves).to(device)
    if modality == "image":
        imgs = [load_image(it.path, img_tfm) for it in items]
        return torch.stack(imgs).to(device)
    raise ValueError(f"Unsupported modality: {modality}")


def extract_features(model: AudioCLIP, modality: str, payload) -> torch.Tensor:
    if modality == "text":
        ((_, _, f), _), _ = model(text=payload)
    elif modality == "audio":
        ((f, _, _), _), _ = model(audio=payload)
    elif modality == "image":
        ((_, f, _), _), _ = model(image=payload)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return f


def pair_scale(model: AudioCLIP, q_type: str, r_type: str) -> torch.Tensor:
    pair = {q_type, r_type}
    if pair == {"audio", "image"}:
        return torch.clamp(model.logit_scale_ai.exp(), min=1.0, max=100.0)
    if pair == {"audio", "text"}:
        return torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
    if pair == {"image", "text"}:
        return torch.clamp(model.logit_scale.exp(), min=1.0, max=100.0)
    return torch.tensor(1.0, device=model.device)


def mark_trainable(model: AudioCLIP, q_type: str, r_type: str) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False

    pair = {q_type, r_type}

    if "audio" in pair:
        for p in model.audio.parameters():
            p.requires_grad = True

    if "image" in pair:
        for p in model.visual.parameters():
            p.requires_grad = True

    if "text" in pair:
        for p in model.transformer.parameters():
            p.requires_grad = True
        for p in model.token_embedding.parameters():
            p.requires_grad = True
        model.positional_embedding.requires_grad = True
        for p in model.ln_final.parameters():
            p.requires_grad = True
        model.text_projection.requires_grad = True

    # Enable only the relevant logit scale for the selected modality pair.
    model.logit_scale.requires_grad = pair == {"image", "text"}
    model.logit_scale_ai.requires_grad = pair == {"audio", "image"}
    model.logit_scale_at.requires_grad = pair == {"audio", "text"}

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected for this modality pair.")
    return trainable


def run_exact_main_training(args: argparse.Namespace, q_type: str, r_type: str) -> None:
    if args.dataset not in {"esc50", "us8k"}:
        raise ValueError("Exact main.py mode supports only esc50 and us8k.")

    if {q_type, r_type} != {"audio", "text"}:
        raise ValueError("Exact main.py mode for esc50/us8k expects audio-text or text-audio pairs.")

    if args.dataset_root is None:
        raise ValueError(f"--dataset {args.dataset} requires --dataset-root")

    reproduce_root = Path(__file__).resolve().parent
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_path = resolve_checkpoint_path(model_path)

    run_fold_script = reproduce_root / "run_fold.py"
    if not run_fold_script.exists():
        raise FileNotFoundError(f"run_fold.py not found: {run_fold_script}")

    if args.protocol == "paper":
        if args.fold is not None:
            folds = [args.fold]
        else:
            folds = [1, 2, 3, 4, 5] if args.dataset == "esc50" else list(range(1, 11))
    else:
        # main.py training is fold-based; global is ambiguous, so use explicit fold if provided.
        folds = [args.fold] if args.fold is not None else [1]
        print("[Info] protocol=global is fold-ambiguous for main.py training; using fold(s):", folds)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir = output_dir / f"{args.dataset}_saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    parsed_accs: list[float] = []
    fold_iter = folds
    if not args.no_progress and tqdm is not None:
        fold_iter = tqdm(folds, desc="Fine-tuning folds (exact main.py)", unit="fold")

    for fold in fold_iter:
        cmd = [
            sys.executable,
            str(run_fold_script),
            "--dataset",
            args.dataset,
            "--dataset-root",
            str(dataset_root),
            "--checkpoint",
            str(model_path),
            "--fold",
            str(fold),
            "--epochs",
            str(args.epochs),
            "--batch-train",
            str(args.batch_size),
            "--batch-test",
            str(args.batch_size),
            "--workers-train",
            str(args.workers),
            "--workers-test",
            str(args.workers),
            "--saved-models-path",
            str(saved_models_dir),
            "--disable-visdom",
            "--enable-checkpoint-saving",
            "--suffix",
            f"fine-tune-fold{fold}",
        ]

        print(f"\n[Exact] Running fold {fold}")
        print(" ".join(shlex.quote(x) for x in cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=str(reproduce_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_acc = None
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            m = ACC_PATTERN.search(line)
            if m:
                last_acc = float(m.group(1))

        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Fold {fold} failed with exit code {code}")

        if last_acc is not None:
            parsed_accs.append(last_acc)
            print(f"[Exact] Parsed fold {fold} Val. Acc. Eval.: {last_acc:.4f}")

    if parsed_accs:
        mean = float(np.mean(parsed_accs))
        std = float(np.std(parsed_accs))
        print("\n[Exact] Summary")
        print(f"  dataset: {args.dataset}")
        print(f"  pair: {q_type} -> {r_type} (training objective is main.py audio-text)")
        print(f"  folds: {len(parsed_accs)}")
        print(f"  Mean Val. Acc. Eval.: {mean:.4f}")
        print(f"  Std Val. Acc. Eval.: {std:.4f}")
    else:
        print("\n[Exact] No Val. Acc. Eval. values were parsed from logs.")


def run_generic_training(args: argparse.Namespace, q_type: str, r_type: str) -> None:
    set_seed(args.seed)

    all_items, dataset_desc = load_items_from_args(args)
    all_items = prepare_items_for_pair(args.dataset, q_type, r_type, all_items)

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_path = resolve_checkpoint_path(model_path)

    device = torch.device(args.device)
    model = AudioCLIP(pretrained=str(model_path)).to(device)

    trainable_params = mark_trainable(model, q_type, r_type)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    sample_rate = 44_100
    audio_len = default_audio_length(args.dataset) if args.audio_length is None else args.audio_length
    aud_tfm = audio_transform(audio_len)
    img_tfm = image_transform()

    pair_ds = PairDataset(
        query_items=all_items[q_type],
        result_items=all_items[r_type],
        seed=args.seed,
        max_pairs=args.max_train_pairs,
    )
    train_loader = tud.DataLoader(
        pair_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pairs,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generic fine-tuning setup")
    print(f"  dataset: {dataset_desc}")
    print(f"  pair: {q_type} -> {r_type}")
    print(f"  epochs: {args.epochs}")
    print(f"  train_pairs: {len(pair_ds)}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")

    best_hit1 = -1.0
    best_path: Path | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []

        loader_iter = train_loader
        if not args.no_progress and tqdm is not None:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for q_batch, r_batch in loader_iter:
            q_payload = build_modal_payload(
                q_batch,
                modality=q_type,
                device=device,
                sample_rate=sample_rate,
                aud_tfm=aud_tfm,
                img_tfm=img_tfm,
                prompt_template=args.prompt_template,
            )
            r_payload = build_modal_payload(
                r_batch,
                modality=r_type,
                device=device,
                sample_rate=sample_rate,
                aud_tfm=aud_tfm,
                img_tfm=img_tfm,
                prompt_template=args.prompt_template,
            )

            optimizer.zero_grad(set_to_none=True)

            q_feat = extract_features(model, q_type, q_payload)
            r_feat = extract_features(model, r_type, r_payload)

            scale = pair_scale(model, q_type, r_type)
            logits = scale * (q_feat @ r_feat.T)
            target = torch.arange(logits.shape[0], dtype=torch.int64, device=logits.device)

            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            if not args.no_progress and tqdm is not None:
                loader_iter.set_postfix(loss=f"{loss_value:.4f}")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[Epoch {epoch}] train_loss={mean_loss:.4f}")

        ckpt_epoch = output_dir / f"fine_tune_{args.dataset}_{q_type}_{r_type}_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_epoch)

        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            model.eval()
            with torch.no_grad():
                hit_by_k, r_by_k, sup_acc, nq, nr = evaluate_once_accuracy(
                    model,
                    q_type,
                    r_type,
                    all_items,
                    batch_size=args.batch_size,
                    device=device,
                    sample_rate=sample_rate,
                    audio_len=audio_len,
                    prompt_template=args.prompt_template,
                    show_progress=not args.no_progress,
                    progress_prefix=f"[Epoch {epoch}] ",
                    show_pair_progress=False,
                )

            hit_line = " ".join([f"hit@{k}={hit_by_k[k] * 100.0:.2f}" for k in K_VALUES])
            r_line = " ".join([f"r@{k}={r_by_k[k] * 100.0:.2f}" for k in K_VALUES])
            sup_line = "N/A" if sup_acc is None else f"{sup_acc * 100.0:.2f}"
            print(f"[Epoch {epoch}] eval queries={nq} results={nr} {hit_line} {r_line} supervised_cls_acc={sup_line}")

            if hit_by_k[1] > best_hit1:
                best_hit1 = hit_by_k[1]
                best_path = output_dir / f"fine_tune_{args.dataset}_{q_type}_{r_type}_best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"[Epoch {epoch}] New best hit@1={best_hit1 * 100.0:.2f} -> {best_path}")

    print("\nGeneric fine-tuning complete")
    print(f"  dataset: {dataset_desc}")
    print(f"  pair: {q_type} -> {r_type}")
    print(f"  output_dir: {output_dir}")
    if best_path is not None:
        print(f"  best_checkpoint: {best_path}")


def main() -> None:
    args = parse_args()

    q_type, r_type = parse_pair(args.pair)
    if args.dataset not in DATASET_MODALITIES:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    validate_pair_for_dataset(args.dataset, q_type, r_type)

    trainer_mode = args.trainer_mode
    if trainer_mode == "auto":
        trainer_mode = "exact" if args.dataset in {"esc50", "us8k"} else "generic"

    print(f"Trainer mode: {trainer_mode}")

    if trainer_mode == "exact":
        run_exact_main_training(args, q_type, r_type)
    else:
        run_generic_training(args, q_type, r_type)


if __name__ == "__main__":
    main()
