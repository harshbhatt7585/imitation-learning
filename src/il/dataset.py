"""Torch Dataset over collected MiniWoB demonstrations.

Loads every ``<task>.npz`` under a directory, concatenates the frames, and
serves (image, text_ids, text_len, target_cell, task_index) samples for
instruction-conditioned behavioral cloning.

  * The click target (x, y) in 160x210 pixel space becomes a single class index
    over a ``grid x grid`` heatmap (matches the model's spatial head).
  * The instruction utterance is char-encoded to a fixed-length id sequence; the
    vocab is built here and exposed as ``self.vocab`` for the checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .env import TASK_HEIGHT, TASK_WIDTH
from .text import build_vocab, encode

MAX_TEXT_LEN_CAP = 64


class DemoDataset(Dataset):
    def __init__(self, data_dir: str | Path, resize_to: int, grid: int):
        self.resize_to = resize_to
        self.grid = grid
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz demos found in {data_dir}")

        images, coords, utts, tasks = [], [], [], []
        self.task_names: list[str] = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            if "utterances" not in d:
                raise KeyError(
                    f"{f} has no 'utterances' — re-run il.collect to record them."
                )
            task = str(d["task"])
            self.task_names.append(task)
            ti = len(self.task_names) - 1
            images.append(d["images"])
            coords.append(d["coords"])
            utts.append(np.asarray(d["utterances"], dtype=object))
            tasks.append(np.full(len(d["images"]), ti, dtype=np.int64))

        self.images = np.concatenate(images)            # (N,210,160,3) uint8
        self.coords = np.concatenate(coords)            # (N,2) float32, (x,y) px
        self.utterances = np.concatenate(utts)          # (N,) object (str)
        self.task_idx = np.concatenate(tasks)           # (N,)

        self.vocab = build_vocab(self.utterances)
        observed = max((len(str(u)) for u in self.utterances), default=1)
        self.max_text_len = min(observed, MAX_TEXT_LEN_CAP)

    def __len__(self) -> int:
        return len(self.images)

    def _coord_to_cell(self, x: float, y: float) -> int:
        """Map a pixel (x, y) to a flat grid-cell index in [0, grid*grid)."""
        gx = min(int(x / TASK_WIDTH * self.grid), self.grid - 1)
        gy = min(int(y / TASK_HEIGHT * self.grid), self.grid - 1)
        return gy * self.grid + gx

    def __getitem__(self, i: int):
        img = self.images[i]                            # (210,160,3) uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,210,160)
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(self.resize_to, self.resize_to),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        x, y = float(self.coords[i, 0]), float(self.coords[i, 1])
        cell = self._coord_to_cell(x, y)
        ids, length = encode(self.utterances[i], self.vocab, self.max_text_len)
        return (
            t,
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(cell, dtype=torch.long),
            int(self.task_idx[i]),
        )


def cell_to_pixel(cell: int, grid: int) -> tuple[float, float]:
    """Inverse of `_coord_to_cell`: grid-cell index -> pixel center (x, y)."""
    gy, gx = divmod(int(cell), grid)
    x = (gx + 0.5) / grid * TASK_WIDTH
    y = (gy + 0.5) / grid * TASK_HEIGHT
    return x, y
