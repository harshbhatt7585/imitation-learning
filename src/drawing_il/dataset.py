"""Dataset utilities for drawing imitation from GIF/frame demos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageSequence
from torch.utils.data import Dataset


IMAGE_EXTS = {".gif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _ink(img: Image.Image, size: int) -> np.ndarray:
    arr = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255.0
    return 1.0 - arr


def _load_frames(path: Path, size: int) -> list[np.ndarray]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return [_ink(Image.open(p), size) for p in files]

    if path.suffix.lower() == ".gif":
        img = Image.open(path)
        return [_ink(frame.copy(), size) for frame in ImageSequence.Iterator(img)]

    return [_ink(Image.open(path), size)]


def iter_demo_paths(data_dir: str | Path):
    root = Path(data_dir)
    for path in sorted(root.iterdir()):
        if path.is_dir() or path.suffix.lower() in IMAGE_EXTS:
            yield path


class DrawingVideoDataset(Dataset):
    """Samples `(current_canvas, final_target) -> next ink cell` from demos."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 64,
        grid: int = 32,
        min_new_ink: float = 0.01,
    ):
        self.image_size = image_size
        self.grid = grid
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []

        for path in iter_demo_paths(data_dir):
            frames = _load_frames(path, image_size)
            if len(frames) < 2:
                continue
            target = np.maximum.reduce(frames)
            previous = np.zeros_like(target)
            for frame in frames:
                current = np.maximum(previous, frame)
                new_ink = np.clip(current - previous, 0.0, 1.0)
                if float(new_ink.sum()) >= min_new_ink:
                    y, x = np.argwhere(new_ink > 0.1).mean(axis=0)
                    gx = min(int(x / image_size * grid), grid - 1)
                    gy = min(int(y / image_size * grid), grid - 1)
                    self.samples.append((previous.copy(), target.copy(), gy * grid + gx))
                previous = current

        if not self.samples:
            raise FileNotFoundError(f"No usable drawing demos found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        canvas, target, cell = self.samples[idx]
        obs = np.stack([canvas, target], axis=0)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(cell, dtype=torch.long),
        )


def cell_to_xy(cell: int, grid: int, image_size: int) -> tuple[int, int]:
    gy, gx = divmod(int(cell), grid)
    x = round((gx + 0.5) / grid * image_size)
    y = round((gy + 0.5) / grid * image_size)
    return x, y
