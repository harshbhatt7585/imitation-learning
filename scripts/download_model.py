"""Download a Hugging Face model snapshot into the local models/ directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PRESETS = {
    "qwen-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": "models/qwen2.5-0.5b-instruct",
    },
    "hrm-text-1b": {
        "repo_id": "sapientinc/HRM-Text-1B",
        "local_dir": "models/hrm-text-1b",
    },
}

ALLOW_PATTERNS = [
    "config.json",
    "configuration_*.py",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "README.md",
    "LICENSE",
]


def main():
    ap = argparse.ArgumentParser(description="Download a local model snapshot.")
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="qwen-0.5b",
        help="Known model shortcut.",
    )
    ap.add_argument("--repo-id", help="Override Hugging Face repo id.")
    ap.add_argument("--out-dir", help="Override local output directory.")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    repo_id = args.repo_id or preset["repo_id"]
    out_dir = Path(args.out_dir or preset["local_dir"])
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    print(f"Downloaded {repo_id} -> {path}")


if __name__ == "__main__":
    main()
