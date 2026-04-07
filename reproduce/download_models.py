#!/usr/bin/env python3
import argparse
import urllib.request
from pathlib import Path


def download_if_missing(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Skipping existing file: {out_path}")
        return
    print(f"Downloading {out_path.name}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AudioCLIP model files for reproduction.")
    parser.add_argument("--output-dir", type=Path, default=Path("reproduce") / "models")
    args = parser.parse_args()

    checkpoint_dir = args.output_dir / "checkpoints"
    base_dir = args.output_dir / "base"

    downloads = [
        (
            "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt",
            checkpoint_dir / "AudioCLIP-Full-Training.pt",
        ),
        (
            "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt",
            checkpoint_dir / "AudioCLIP-Partial-Training.pt",
        ),
        (
            "https://raw.githubusercontent.com/AndreyGuzhov/AudioCLIP/master/assets/CLIP.pt",
            base_dir / "CLIP.pt",
        ),
        (
            "https://raw.githubusercontent.com/AndreyGuzhov/AudioCLIP/master/assets/ESRNXFBSP.pt",
            base_dir / "ESRNXFBSP.pt",
        ),
        (
            "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
            base_dir / "bpe_simple_vocab_16e6.txt.gz",
        ),
    ]

    for url, out_path in downloads:
        download_if_missing(url, out_path)

    print("")
    print(f"Downloaded models to: {args.output_dir}")
    print(f"Checkpoint to use: {checkpoint_dir / 'AudioCLIP-Full-Training.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
