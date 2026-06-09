"""Train a drawing policy from GIF/frame demonstrations."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import DrawingVideoDataset
from .model import DrawingPolicy


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, exact, n = 0.0, 0, 0
    with torch.no_grad():
        for obs, cell in loader:
            obs, cell = obs.to(device), cell.to(device)
            logits = model(obs)
            total_loss += crit(logits, cell).item() * len(cell)
            exact += (logits.argmax(1) == cell).sum().item()
            n += len(cell)
    return total_loss / max(n, 1), exact / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train drawing imitation policy.")
    ap.add_argument("--data-dir", default="data/drawing/videos")
    ap.add_argument("--out", default="runs/drawing/best.pt")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = DrawingVideoDataset(args.data_dir, image_size=args.image_size, grid=args.grid)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = DrawingPolicy(grid=args.grid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    print(f"Device: {device}")
    print(f"Dataset: {len(ds)} samples ({n_train} train / {n_val} val)")

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for obs, cell in train_loader:
            obs, cell = obs.to(device), cell.to(device)
            loss = crit(model(obs), cell)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(cell)
            n += len(cell)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1:3d} | train_loss {total / max(n, 1):.4f} | "
            f"val_loss {val_loss:.4f} | exact {val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "grid": args.grid,
                    "image_size": args.image_size,
                },
                out,
            )
    print(f"Saved best checkpoint to {out} (exact {best_acc:.3f})")


if __name__ == "__main__":
    main()
