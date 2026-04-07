#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path


def infer_audioclip_root(script_path: Path) -> Path:
    """Infer AudioCLIP root when this script lives in AudioCLIP/reproduce."""
    candidate = script_path.resolve().parent.parent
    if not (candidate / "main.py").exists():
        raise FileNotFoundError(
            f"Could not find AudioCLIP root from script location: {script_path}. "
            "Expected main.py in the parent directory."
        )
    return candidate


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_bytes()[:120]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    release_map = {
        "AudioCLIP-Full-Training.pt": "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt",
        "AudioCLIP-Partial-Training.pt": "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt",
    }

    if not looks_like_lfs_pointer(checkpoint):
        return checkpoint

    if checkpoint.name not in release_map:
        raise RuntimeError(
            f"Checkpoint appears to be a Git LFS pointer and cannot be auto-downloaded: {checkpoint}\n"
            "Please provide a real checkpoint file path via --checkpoint."
        )

    out_dir = Path("/kaggle/working") / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / checkpoint.name

    if out_path.exists() and not looks_like_lfs_pointer(out_path):
        print(f"Using previously downloaded checkpoint: {out_path}")
        return out_path

    url = release_map[checkpoint.name]
    print(f"Detected Git LFS pointer for {checkpoint.name}. Downloading real file to: {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def build_command(args: argparse.Namespace, audioclip_root: Path, reproduce_root: Path) -> list[str]:
    protocol_map = {
        "esc50": reproduce_root / "protocols" / "audioclip-esc50.json",
        "us8k": reproduce_root / "protocols" / "audioclip-us8k.json",
    }

    protocol_path = protocol_map[args.dataset]
    command = [
        sys.executable,
        str(audioclip_root / "main.py"),
        "--config",
        str(protocol_path),
        "--Dataset.args.root",
        str(args.dataset_root),
        "--Dataset.args.fold",
        str(args.fold),
        "--Model.args.pretrained",
        str(args.checkpoint),
        "--Setup.saved_models_path",
        str(args.saved_models_path),
        "-e",
        str(args.epochs),
        "-b",
        str(args.batch_train),
        "-B",
        str(args.batch_test),
        "-w",
        str(args.workers_train),
        "-W",
        str(args.workers_test),
        "--visdom-host",
        args.visdom_host,
        "--visdom-port",
        str(args.visdom_port),
        "--visdom-env-path",
        str(args.visdom_env_path),
    ]

    if args.seed is not None:
        command.extend(["-R", str(args.seed)])

    if args.suffix:
        command.extend(["-s", args.suffix])

    if args.extra:
        command.extend(shlex.split(args.extra))

    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one AudioCLIP fold (Kaggle-friendly, no upstream code edits)."
    )
    parser.add_argument("--dataset", choices=["esc50", "us8k"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to AudioCLIP checkpoint .pt file. Defaults to AudioCLIP/assets/AudioCLIP-Full-Training.pt",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-test", type=int, default=64)
    parser.add_argument("--workers-train", type=int, default=4)
    parser.add_argument("--workers-test", type=int, default=4)
    parser.add_argument("--visdom-host", type=str, default="127.0.0.1")
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument(
        "--visdom-env-path",
        type=Path,
        default=Path("/kaggle/working") / "visdom_env",
        help="Writable directory for visdom env state on Kaggle.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suffix", type=str, default="reproduce")
    parser.add_argument(
        "--saved-models-path",
        type=Path,
        default=Path("/kaggle/working") / "outputs" / "saved_models",
        help="Used only if checkpoint saving is enabled.",
    )
    parser.add_argument("--extra", type=str, default="", help="Extra args passed to AudioCLIP main.py")
    parser.add_argument(
        "--disable-visdom",
        action="store_true",
        default=True,
        help="Disable visdom and keep console-only logging (default: enabled).",
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
    audioclip_root = infer_audioclip_root(Path(__file__))

    dataset_root = resolve_path(args.dataset_root, Path.cwd())
    checkpoint_arg = args.checkpoint or (audioclip_root / "assets" / "AudioCLIP-Full-Training.pt")
    checkpoint = checkpoint_arg if checkpoint_arg.is_absolute() else resolve_path(checkpoint_arg, audioclip_root)
    saved_models_path = resolve_path(args.saved_models_path, reproduce_root)
    visdom_env_path = resolve_path(args.visdom_env_path, Path.cwd())

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint = resolve_checkpoint_path(checkpoint)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    saved_models_path.mkdir(parents=True, exist_ok=True)
    visdom_env_path.mkdir(parents=True, exist_ok=True)

    args.dataset_root = dataset_root
    args.checkpoint = checkpoint
    args.saved_models_path = saved_models_path
    args.visdom_env_path = visdom_env_path

    cmd = build_command(args, audioclip_root, reproduce_root)

    print("Running command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print()

    env = os.environ.copy()
    if args.disable_visdom:
        env["AUDIOCLIP_DISABLE_VISDOM"] = "1"
    if args.disable_checkpoint_saving:
        env["AUDIOCLIP_DISABLE_CHECKPOINTS"] = "1"

    return subprocess.call(cmd, cwd=str(audioclip_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
