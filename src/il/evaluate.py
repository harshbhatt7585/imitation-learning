"""Evaluate a trained BC policy live in MiniWoB++.

For each task we roll out the learned click policy in the real browser and
measure task success rate, alongside a random-click baseline for context. This
is the metric that matters: can the agent actually navigate the task from pixels?
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import env as E
from .dataset import cell_to_pixel
from .model import ClickPolicy
from .text import encode
from .utils import get_device, load_config, resolve_path, set_seed


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ClickPolicy(
        vocab_size=len(ckpt["vocab"]) + 2,
        grid=ckpt["grid"],
        channels=tuple(ckpt["channels"]),
        text_emb=ckpt["text_emb"],
        text_hidden=ckpt["text_hidden"],
        head_dropout=ckpt.get("head_dropout", 0.10),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def encode_utterance(utterance, ckpt, device):
    ids, length = encode(utterance, ckpt["vocab"], ckpt["max_text_len"])
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    len_t = torch.tensor([length], dtype=torch.long, device=device)
    return ids_t, len_t


def screenshot_to_tensor(obs, resize_to, device):
    img = E.get_screenshot(obs)                          # (210,160,3) uint8
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(resize_to, resize_to),
        mode="bilinear", align_corners=False,
    )
    return t.to(device)


def run_task(model, ckpt, task, episodes, max_steps, headless, device, rng, policy="bc"):
    e = E.make_env(task, headless=headless)
    grid, resize_to = ckpt["grid"], ckpt["resize_to"]
    successes = 0
    try:
        for ep in range(episodes):
            obs = E.reset(e, seed=10_000 + ep)           # held-out seeds vs training
            solved = False
            for _ in range(max_steps):
                if policy == "random":
                    x = rng.uniform(0, E.TASK_WIDTH)
                    y = rng.uniform(0, E.TASK_HEIGHT)
                else:
                    t = screenshot_to_tensor(obs, resize_to, device)
                    ids_t, len_t = encode_utterance(
                        E.get_utterance(obs), ckpt, device
                    )
                    cell = int(model.predict_cell(t, ids_t, len_t).item())
                    x, y = cell_to_pixel(cell, grid)
                obs, reward, terminated, truncated, _ = E.step_click(e, x, y)
                if terminated or truncated:
                    solved = reward > 0
                    break
            successes += int(solved)
    finally:
        e.close()
    return successes / max(episodes, 1)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the BC policy live.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", help="Override checkpoint path.")
    ap.add_argument("--tasks", nargs="*", help="Override task list.")
    ap.add_argument("--baseline", action="store_true", help="Also run random-click baseline.")
    ap.add_argument("--watch", action="store_true",
                    help="Show a visible Chrome window instead of running headless.")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg.get("seed", 0))
    ecfg = cfg["eval"]
    device = get_device(cfg["train"]["device"])
    ckpt_path = resolve_path(args.checkpoint or ecfg["checkpoint"])
    model, ckpt = load_policy(ckpt_path, device)
    tasks = args.tasks or cfg["tasks"]
    headless = ecfg["headless"] and not args.watch
    rng = np.random.default_rng(cfg.get("seed", 0))

    print(f"Evaluating {ckpt_path.name} on {device} "
          f"({ecfg['episodes_per_task']} eps/task, held-out seeds)\n")
    header = f"{'task':<16} {'BC':>8}"
    if args.baseline:
        header += f" {'random':>8}"
    print(header)
    print("-" * len(header))

    bc_rates, rand_rates = [], []
    for task in tasks:
        bc = run_task(model, ckpt, task, ecfg["episodes_per_task"],
                      ecfg["max_steps"], headless, device, rng, "bc")
        bc_rates.append(bc)
        line = f"{task:<16} {bc:>7.0%}"
        if args.baseline:
            rnd = run_task(model, ckpt, task, ecfg["episodes_per_task"],
                           ecfg["max_steps"], headless, device, rng, "random")
            rand_rates.append(rnd)
            line += f" {rnd:>7.0%}"
        print(line)

    print("-" * len(header))
    summary = f"{'mean':<16} {np.mean(bc_rates):>7.0%}"
    if args.baseline:
        summary += f" {np.mean(rand_rates):>7.0%}"
    print(summary)


if __name__ == "__main__":
    main()
