#!/usr/bin/env python3
import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


ACC_PATTERN = re.compile(r"Val\. Acc\. Eval\.:\s*([0-9]*\.?[0-9]+)")


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def run_fold(
    reproduce_root: Path,
    dataset: str,
    dataset_root: Path,
    checkpoint: Path,
    saved_models_path: Path,
    visdom_host: str,
    visdom_port: int,
    visdom_env_path: Path,
    disable_visdom: bool,
    disable_checkpoint_saving: bool,
    fold: int,
    epochs: int,
    batch_train: int,
    batch_test: int,
    workers_train: int,
    workers_test: int,
    seed: int | None,
) -> float | None:
    run_fold_script = reproduce_root / "run_fold.py"

    cmd = [
        sys.executable,
        str(run_fold_script),
        "--dataset",
        dataset,
        "--dataset-root",
        str(dataset_root),
        "--checkpoint",
        str(checkpoint),
        "--saved-models-path",
        str(saved_models_path),
        "--visdom-host",
        visdom_host,
        "--visdom-port",
        str(visdom_port),
        "--visdom-env-path",
        str(visdom_env_path),
        "--fold",
        str(fold),
        "--epochs",
        str(epochs),
        "--batch-train",
        str(batch_train),
        "--batch-test",
        str(batch_test),
        "--workers-train",
        str(workers_train),
        "--workers-test",
        str(workers_test),
        "--suffix",
        f"reproduce-fold{fold}",
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if disable_visdom:
        cmd.append("--disable-visdom")
    else:
        cmd.append("--enable-visdom")

    if disable_checkpoint_saving:
        cmd.append("--disable-checkpoint-saving")
    else:
        cmd.append("--enable-checkpoint-saving")

    print(f"\n=== Fold {fold} ===")
    print(" ".join(shlex.quote(part) for part in cmd))

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
        match = ACC_PATTERN.search(line)
        if match:
            last_acc = float(match.group(1))

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Fold {fold} failed with exit code {code}")

    return last_acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-validation folds and aggregate Val. Acc. Eval.")
    parser.add_argument("--dataset", choices=["esc50", "us8k"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to AudioCLIP checkpoint .pt file. Defaults to AudioCLIP/assets/AudioCLIP-Full-Training.pt",
    )
    parser.add_argument(
        "--saved-models-path",
        type=Path,
        default=Path("/kaggle/working") / "outputs" / "saved_models",
        help="Where fold checkpoints are written. Use a writable path on Kaggle.",
    )
    parser.add_argument("--visdom-host", type=str, default="127.0.0.1")
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument(
        "--visdom-env-path",
        type=Path,
        default=Path("/kaggle/working") / "visdom_env",
        help="Writable directory for visdom env state on Kaggle.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-test", type=int, default=64)
    parser.add_argument("--workers-train", type=int, default=4)
    parser.add_argument("--workers-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--disable-visdom",
        action="store_true",
        default=True,
        help="Disable visdom and use console-only logging (default: enabled).",
    )
    parser.add_argument(
        "--enable-visdom",
        action="store_false",
        dest="disable_visdom",
        help="Enable visdom logging/server behavior.",
    )
    parser.add_argument(
        "--disable-checkpoint-saving",
        action="store_true",
        default=True,
        help="Disable saving model checkpoints (default: enabled).",
    )
    parser.add_argument(
        "--enable-checkpoint-saving",
        action="store_false",
        dest="disable_checkpoint_saving",
        help="Enable model checkpoint saving.",
    )

    args = parser.parse_args()

    reproduce_root = Path(__file__).resolve().parent

    dataset_root = resolve_path(args.dataset_root, Path.cwd())
    audioclip_root = reproduce_root.parent
    checkpoint_arg = args.checkpoint or (audioclip_root / "assets" / "AudioCLIP-Full-Training.pt")
    checkpoint = checkpoint_arg if checkpoint_arg.is_absolute() else resolve_path(checkpoint_arg, audioclip_root)
    saved_models_path = resolve_path(args.saved_models_path, Path.cwd())
    visdom_env_path = resolve_path(args.visdom_env_path, Path.cwd())

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    saved_models_path.mkdir(parents=True, exist_ok=True)
    visdom_env_path.mkdir(parents=True, exist_ok=True)

    folds = range(1, 6) if args.dataset == "esc50" else range(1, 11)
    accuracies: list[float] = []

    for fold in folds:
        acc = run_fold(
            reproduce_root=reproduce_root,
            dataset=args.dataset,
            dataset_root=dataset_root,
            checkpoint=checkpoint,
            saved_models_path=saved_models_path,
            visdom_host=args.visdom_host,
            visdom_port=args.visdom_port,
            visdom_env_path=visdom_env_path,
            disable_visdom=args.disable_visdom,
            disable_checkpoint_saving=args.disable_checkpoint_saving,
            fold=fold,
            epochs=args.epochs,
            batch_train=args.batch_train,
            batch_test=args.batch_test,
            workers_train=args.workers_train,
            workers_test=args.workers_test,
            seed=args.seed,
        )
        if acc is not None:
            accuracies.append(acc)
            print(f"Detected Fold {fold} Val. Acc. Eval.: {acc:.4f}")
        else:
            print(f"No Val. Acc. Eval. value detected for fold {fold}.")

    if accuracies:
        mean = sum(accuracies) / len(accuracies)
        var = sum((a - mean) ** 2 for a in accuracies) / len(accuracies)
        std = var ** 0.5

        print("\n=== Cross-Validation Summary ===")
        print(f"Dataset: {args.dataset}")
        print(f"Folds with parsed accuracy: {len(accuracies)}")
        print(f"Mean Val. Acc. Eval.: {mean:.4f}")
        print(f"Std Val. Acc. Eval.: {std:.4f}")
    else:
        print("\nNo accuracy values were parsed from run logs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
