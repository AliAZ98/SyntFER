#!/usr/bin/env python3
"""Upload one SyntFER curated dataset variant as a Hugging Face configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Local curated dataset folder")
    parser.add_argument("--repo-id", required=True, help="HF repo, e.g. AliAZ98/SyntFER-Curated")
    parser.add_argument("--config-name", required=True, help="HF configuration/subset name")
    parser.add_argument("--private", action="store_true", help="Create/push as a private dataset")
    parser.add_argument("--max-shard-size", default="2GB")
    return parser.parse_args()


def validate_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    # Support either class folders directly or split/class folders.
    direct_classes = {p.name for p in root.iterdir() if p.is_dir()}
    if set(CLASSES).issubset(direct_classes):
        return

    split_dirs = [p for p in root.iterdir() if p.is_dir()]
    valid_split = False
    for split_dir in split_dirs:
        child_dirs = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if set(CLASSES).issubset(child_dirs):
            valid_split = True
            break

    if not valid_split:
        raise ValueError(
            "Expected the seven FER class folders either directly under the dataset root "
            "or inside split folders. Expected: " + ", ".join(CLASSES)
        )


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    validate_root(root)

    dataset = load_dataset("imagefolder", data_dir=str(root))
    print(dataset)

    dataset.push_to_hub(
        args.repo_id,
        config_name=args.config_name,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )

    print(
        f"Uploaded {args.config_name!r} to https://huggingface.co/datasets/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
