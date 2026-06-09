"""Roll out a trained drawing policy and save an imitation GIF."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import _load_frames, cell_to_xy
from .model import DrawingPolicy


def _to_image(ink: np.ndarray) -> Image.Image:
    arr = ((1.0 - np.clip(ink, 0.0, 1.0)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Draw a target with a trained policy.")
    ap.add_argument("--checkpoint", default="runs/drawing/best.pt")
    ap.add_argument("--target", default="data/drawing/videos/circle.gif")
    ap.add_argument("--out", default="runs/drawing/rollout.gif")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--brush", type=int, default=3)
    ap.add_argument("--connect", action="store_true", help="Connect predicted brush points with lines.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    grid, image_size = ckpt["grid"], ckpt["image_size"]
    model = DrawingPolicy(grid=grid)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target = np.maximum.reduce(_load_frames(Path(args.target), image_size))
    canvas = np.zeros_like(target)
    frames: list[Image.Image] = []
    last_xy: tuple[int, int] | None = None

    for _ in range(args.steps):
        obs = torch.tensor(np.stack([canvas, target])[None], dtype=torch.float32)
        logits = model(obs).squeeze(0)
        remaining = np.clip(target - canvas, 0.0, 1.0)
        cell_scores = []
        for cell_idx in range(grid * grid):
            x, y = cell_to_xy(cell_idx, grid, image_size)
            r = max(1, args.brush)
            patch = remaining[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            cell_scores.append(float(patch.sum()))
        mask = torch.tensor(cell_scores, dtype=torch.float32)
        if float(mask.max()) > 0:
            logits = logits.masked_fill(mask <= 0, -1e9)
        cell = int(logits.argmax().item())
        xy = cell_to_xy(cell, grid, image_size)

        img = _to_image(canvas)
        draw = ImageDraw.Draw(img)
        if args.connect and last_xy is not None:
            draw.line([last_xy, xy], fill=(0, 0, 0), width=args.brush * 2)
        else:
            draw.ellipse(
                [xy[0] - args.brush, xy[1] - args.brush, xy[0] + args.brush, xy[1] + args.brush],
                fill=(0, 0, 0),
            )
        canvas = 1.0 - np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        frames.append(img)
        last_xy = xy

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=45, loop=0)
    print(f"Wrote rollout GIF to {out}")


if __name__ == "__main__":
    main()
