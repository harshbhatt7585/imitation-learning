"""Generate small drawing demonstration GIFs.

Each GIF is a "video" of an expert drawing one shape stroke-by-stroke. The
training code later infers the expert action from the new ink between frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


CANVAS = 96


def _line_points(points: list[tuple[int, int]], steps_per_segment: int = 12):
    out: list[tuple[int, int]] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for i in range(steps_per_segment):
            t = i / steps_per_segment
            out.append((round(x0 * (1 - t) + x1 * t), round(y0 * (1 - t) + y1 * t)))
    out.append(points[-1])
    return out


def _circle_points(cx: int, cy: int, r: int, steps: int = 72):
    import math

    return [
        (round(cx + r * math.cos(2 * math.pi * i / steps)),
         round(cy + r * math.sin(2 * math.pi * i / steps)))
        for i in range(steps + 1)
    ]


def _save_gif(path: Path, points: list[tuple[int, int]], brush: int = 4):
    frames: list[Image.Image] = []
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)
    last = points[0]
    for point in points[1:]:
        draw.line([last, point], fill=0, width=brush)
        frames.append(img.copy().convert("RGB"))
        last = point
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=45,
        loop=0,
    )


def main():
    ap = argparse.ArgumentParser(description="Create synthetic drawing demo GIFs.")
    ap.add_argument("--out-dir", default="data/drawing/videos")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demos = {
        "square.gif": _line_points([(24, 24), (72, 24), (72, 72), (24, 72), (24, 24)]),
        "triangle.gif": _line_points([(48, 18), (76, 72), (20, 72), (48, 18)]),
        "zigzag.gif": _line_points([(18, 26), (38, 70), (56, 26), (76, 70)]),
        "circle.gif": _circle_points(48, 48, 28),
        "check.gif": _line_points([(20, 52), (40, 70), (76, 28)]),
    }
    for name, points in demos.items():
        _save_gif(out_dir / name, points)
    print(f"Wrote {len(demos)} demo GIFs to {out_dir}")


if __name__ == "__main__":
    main()
