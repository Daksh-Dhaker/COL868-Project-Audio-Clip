#!/usr/bin/env python3
import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SUMMARY_MEAN = re.compile(r"Mean Val\. Acc\. Eval\.:\s*([0-9]*\.?[0-9]+)")
SUMMARY_STD = re.compile(r"Std Val\. Acc\. Eval\.:\s*([0-9]*\.?[0-9]+)")


@dataclass
class RunResult:
    name: str
    mean: float | None
    std: float | None


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def run_cv_case(
    reproduce_root: Path,
    name: str,
    dataset: str,
    dataset_root: Path,
    checkpoint: Path,
    save_dir: Path,
    epochs: int,
    batch_train: int,
    batch_test: int,
    workers_train: int,
    workers_test: int,
    seed: int | None,
) -> RunResult:
    script = reproduce_root / "run_cv.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        dataset,
        "--dataset-root",
        str(dataset_root),
        "--checkpoint",
        str(checkpoint),
        "--enable-checkpoint-saving",
        "--saved-models-path",
        str(save_dir),
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
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"\n=== {name} ===")
    print(" ".join(shlex.quote(part) for part in cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(reproduce_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    mean = None
    std = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        m = SUMMARY_MEAN.search(line)
        s = SUMMARY_STD.search(line)
        if m:
            mean = float(m.group(1))
        if s:
            std = float(s.group(1))

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}")

    return RunResult(name=name, mean=mean, std=std)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple paper-claim reproduction cases (full/partial checkpoint across ESC-50 and US8K)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["esc50", "us8k"],
        default=["esc50", "us8k"],
        help="Datasets to run. Example: --datasets esc50 or --datasets esc50 us8k",
    )
    parser.add_argument("--esc50-root", type=Path, default=None)
    parser.add_argument("--us8k-root", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["full", "partial"],
        default=["full"],
        help="Checkpoint variants to run. Example: --models full or --models full partial",
    )
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--partial-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("reproduce") / "outputs" / "paper_claims")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-test", type=int, default=64)
    parser.add_argument("--workers-train", type=int, default=4)
    parser.add_argument("--workers-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    reproduce_root = Path(__file__).resolve().parent
    audioclip_root = reproduce_root.parent

    selected_datasets = set(args.datasets)
    selected_models = set(args.models)

    esc50_root = resolve_path(args.esc50_root, Path.cwd()) if args.esc50_root is not None else None
    us8k_root = resolve_path(args.us8k_root, Path.cwd()) if args.us8k_root is not None else None
    full_ckpt = args.full_checkpoint or (audioclip_root / "assets" / "AudioCLIP-Full-Training.pt")
    partial_ckpt = args.partial_checkpoint or (audioclip_root / "assets" / "AudioCLIP-Partial-Training.pt")

    full_ckpt = resolve_path(full_ckpt, audioclip_root)
    partial_ckpt = resolve_path(partial_ckpt, audioclip_root)
    output_root = resolve_path(args.output_root, Path.cwd())

    if "esc50" in selected_datasets:
        if esc50_root is None:
            raise ValueError("--esc50-root is required when esc50 is selected in --datasets")
        if not esc50_root.exists():
            raise FileNotFoundError(f"ESC-50 root not found: {esc50_root}")

    if "us8k" in selected_datasets:
        if us8k_root is None:
            raise ValueError("--us8k-root is required when us8k is selected in --datasets")
        if not us8k_root.exists():
            raise FileNotFoundError(f"UrbanSound8K root not found: {us8k_root}")

    if "full" in selected_models and not full_ckpt.exists():
        raise FileNotFoundError(f"Full-training checkpoint not found: {full_ckpt}")
    if "partial" in selected_models and not partial_ckpt.exists():
        raise FileNotFoundError(f"Partial-training checkpoint not found: {partial_ckpt}")

    results: list[RunResult] = []

    if "full" in selected_models and "esc50" in selected_datasets:
        results.append(
            run_cv_case(
                reproduce_root=reproduce_root,
                name="ESC-50 (Full Training Checkpoint)",
                dataset="esc50",
                dataset_root=esc50_root,
                checkpoint=full_ckpt,
                save_dir=output_root / "esc50_full",
                epochs=args.epochs,
                batch_train=args.batch_train,
                batch_test=args.batch_test,
                workers_train=args.workers_train,
                workers_test=args.workers_test,
                seed=args.seed,
            )
        )

    if "full" in selected_models and "us8k" in selected_datasets:
        results.append(
            run_cv_case(
                reproduce_root=reproduce_root,
                name="UrbanSound8K (Full Training Checkpoint)",
                dataset="us8k",
                dataset_root=us8k_root,
                checkpoint=full_ckpt,
                save_dir=output_root / "us8k_full",
                epochs=args.epochs,
                batch_train=args.batch_train,
                batch_test=args.batch_test,
                workers_train=args.workers_train,
                workers_test=args.workers_test,
                seed=args.seed,
            )
        )

    if "partial" in selected_models and "esc50" in selected_datasets:
        results.append(
            run_cv_case(
                reproduce_root=reproduce_root,
                name="ESC-50 (Partial Training Checkpoint)",
                dataset="esc50",
                dataset_root=esc50_root,
                checkpoint=partial_ckpt,
                save_dir=output_root / "esc50_partial",
                epochs=args.epochs,
                batch_train=args.batch_train,
                batch_test=args.batch_test,
                workers_train=args.workers_train,
                workers_test=args.workers_test,
                seed=args.seed,
            )
        )

    if "partial" in selected_models and "us8k" in selected_datasets:
        results.append(
            run_cv_case(
                reproduce_root=reproduce_root,
                name="UrbanSound8K (Partial Training Checkpoint)",
                dataset="us8k",
                dataset_root=us8k_root,
                checkpoint=partial_ckpt,
                save_dir=output_root / "us8k_partial",
                epochs=args.epochs,
                batch_train=args.batch_train,
                batch_test=args.batch_test,
                workers_train=args.workers_train,
                workers_test=args.workers_test,
                seed=args.seed,
            )
        )

    print("\n=== Aggregated Summary ===")
    for res in results:
        mean_text = f"{res.mean:.4f}" if res.mean is not None else "N/A"
        std_text = f"{res.std:.4f}" if res.std is not None else "N/A"
        print(f"{res.name}: mean={mean_text}, std={std_text}")

    print("\nReference values reported in the paper (supervised):")
    print("ESC-50: 97.15%")
    print("UrbanSound8K: 90.07%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
