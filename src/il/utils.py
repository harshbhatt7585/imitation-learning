"""Shared helpers: config loading, device selection, seeding."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto") -> str:
    """Pick a torch device string: mps on Apple Silicon, else cuda, else cpu."""
    import torch

    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
