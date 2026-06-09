"""Collect behavioral-cloning demonstrations from scripted oracle experts.

For each task we roll out the oracle, recording (screenshot, click) pairs. Only
*successful* episodes (final reward > 0) are kept, so the saved dataset is clean
regardless of occasional oracle misses. Each task is written to a compressed
``.npz`` under ``<out_dir>/<task>.npz``:

    images       uint8   (N, 210, 160, 3)   screenshot seen before each click
    coords       float32 (N, 2)             (x, y) pixel that was clicked
    action_type  int64   (N,)               0 = CLICK (single supported type)

These are the "demonstration videos": each episode is a short frame sequence
with the expert's click annotated per frame.
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from . import env as E
from .experts import oracle_click
from .utils import load_config, resolve_path, set_seed

ACTION_CLICK = 0


def collect_task(task: str, n_episodes: int, max_steps: int, headless: bool):
    """Roll out the oracle on one task; return (images, coords) from successes."""
    e = E.make_env(task, headless=headless)
    images: list[np.ndarray] = []
    coords: list[tuple[float, float]] = []
    utterances: list[str] = []
    successes = 0
    attempts = 0
    seed = 0
    pbar = tqdm(total=n_episodes, desc=task, leave=True)
    try:
        while successes < n_episodes:
            attempts += 1
            obs = E.reset(e, seed=seed)
            seed += 1
            ep_images: list[np.ndarray] = []
            ep_coords: list[tuple[float, float]] = []
            ep_utts: list[str] = []
            solved = False
            for _ in range(max_steps):
                xy = oracle_click(obs)
                if xy is None:
                    break
                ep_images.append(E.get_screenshot(obs).copy())
                ep_coords.append(xy)
                ep_utts.append(E.get_utterance(obs))
                obs, reward, terminated, truncated, _info = E.step_click(e, *xy)
                if terminated or truncated:
                    solved = reward > 0
                    break
            if solved:
                images.extend(ep_images)
                coords.extend(ep_coords)
                utterances.extend(ep_utts)
                successes += 1
                pbar.update(1)
            # Guard against an oracle that can never solve a task.
            if attempts > 5 * n_episodes and successes == 0:
                raise RuntimeError(
                    f"Oracle failed to solve any '{task}' episode in {attempts} tries."
                )
    finally:
        pbar.close()
        e.close()
    rate = successes / max(attempts, 1)
    print(f"  {task}: {successes} episodes, {len(images)} frames "
          f"(oracle success rate {rate:.0%})")
    return (
        np.stack(images),
        np.asarray(coords, dtype=np.float32),
        np.asarray(utterances, dtype=object),
    )


def main():
    ap = argparse.ArgumentParser(description="Collect BC demos from oracle experts.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tasks", nargs="*", help="Override the task list.")
    ap.add_argument("--episodes-per-task", type=int, help="Override episode count.")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg.get("seed", 0))
    tasks = args.tasks or cfg["tasks"]
    n = args.episodes_per_task or cfg["collect"]["episodes_per_task"]
    max_steps = cfg["collect"]["max_steps"]
    headless = cfg["collect"]["headless"]

    out_dir = resolve_path(cfg["collect"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {n} episodes/task for: {', '.join(tasks)}")
    for task in tasks:
        images, coords, utterances = collect_task(task, n, max_steps, headless)
        out = out_dir / f"{task}.npz"
        np.savez_compressed(
            out,
            images=images,
            coords=coords,
            utterances=utterances,
            action_type=np.full(len(images), ACTION_CLICK, dtype=np.int64),
            task=task,
        )
        print(f"  saved -> {out} ({images.nbytes / 1e6:.1f} MB raw)")
    print("Done.")


if __name__ == "__main__":
    main()
