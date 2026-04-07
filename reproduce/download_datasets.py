#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
US8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ESC-50 and/or UrbanSound8K datasets for AudioCLIP reproduction."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["esc", "esc50", "urban", "us8k", "all"],
        help="Dataset to download: esc/esc50, urban/us8k, or all (default).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/kaggle/working/datasets",
        help="Directory where archives are downloaded and extracted.",
    )
    return parser.parse_args()


def normalize_dataset_choice(choice: str) -> str:
    if choice in {"esc", "esc50"}:
        return "esc"
    if choice in {"urban", "us8k"}:
        return "urban"
    return "all"


def download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"Skipping existing file: {out_path}")
        return
    print(f"Downloading {out_path.name}")
    with urllib.request.urlopen(url) as response, out_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def ensure_esc50(output_dir: Path) -> Path:
    esc50_dir = output_dir / "ESC-50-master"
    esc50_zip = output_dir / "ESC-50-master.zip"

    download_if_missing(ESC50_URL, esc50_zip)

    if esc50_dir.exists():
        print(f"ESC-50 already extracted: {esc50_dir}")
    else:
        print("Extracting ESC-50...")
        with zipfile.ZipFile(esc50_zip, "r") as zf:
            zf.extractall(output_dir)

    return esc50_dir


def ensure_us8k(output_dir: Path) -> Path:
    us8k_dir = output_dir / "UrbanSound8K"
    us8k_tar = output_dir / "UrbanSound8K.tar.gz"

    download_if_missing(US8K_URL, us8k_tar)

    if us8k_dir.exists():
        print(f"UrbanSound8K already extracted: {us8k_dir}")
    else:
        print("Extracting UrbanSound8K...")
        with tarfile.open(us8k_tar, "r:gz") as tf:
            tf.extractall(output_dir)

    return us8k_dir


def main() -> None:
    args = parse_args()
    dataset_choice = normalize_dataset_choice(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots: list[Path] = []

    if dataset_choice in {"esc", "all"}:
        dataset_roots.append(ensure_esc50(output_dir))

    if dataset_choice in {"urban", "all"}:
        dataset_roots.append(ensure_us8k(output_dir))

    print("\nDone. Dataset roots:")
    for root in dataset_roots:
        print(str(root))

    print("\nUse with reproduce scripts:")
    for root in dataset_roots:
        print(f"  --dataset-root {root}")


if __name__ == "__main__":
    main()
