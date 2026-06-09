"""Train a lightweight WebLINX prompt-to-action imitation model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DATASET_ID, WebLinxTextConfig, WebLinxTextDataset, load_pairs
from .model import WebActionSeq2Seq
from .text import PAD, build_vocab


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss, exact, n = 0.0, 0, 0
    with torch.no_grad():
        for src, src_len, dec_in, target in loader:
            src, src_len = src.to(device), src_len.to(device)
            dec_in, target = dec_in.to(device), target.to(device)
            logits = model(src, src_len, dec_in)
            loss = crit(logits.flatten(0, 1), target.flatten())
            total_loss += loss.item() * len(src)
            pred = logits.argmax(dim=-1)
            mask = target != PAD
            exact += (((pred == target) | ~mask).all(dim=1)).sum().item()
            n += len(src)
    return total_loss / max(n, 1), exact / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train WebLINX action-string BC baseline.")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--max-prompt-len", type=int, default=2048)
    ap.add_argument("--max-action-len", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/weblinx/best.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    cfg = WebLinxTextConfig(args.max_prompt_len, args.max_action_len)

    print("Loading WebLINX pairs...")
    train_pairs = load_pairs(args.train_split, args.train_limit, args.dataset)
    val_pairs = load_pairs(args.val_split, args.val_limit, args.dataset)
    vocab = build_vocab([s for pair in train_pairs for s in pair])
    train_ds = WebLinxTextDataset(train_pairs, vocab, cfg)
    val_ds = WebLinxTextDataset(val_pairs, vocab, cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = WebActionSeq2Seq(vocab_size=len(vocab) + 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.02)
    best = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device: {device} | train={len(train_ds)} val={len(val_ds)} "
        f"vocab={len(vocab) + 4}"
    )
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for src, src_len, dec_in, target in train_loader:
            src, src_len = src.to(device), src_len.to(device)
            dec_in, target = dec_in.to(device), target.to(device)
            logits = model(src, src_len, dec_in)
            loss = crit(logits.flatten(0, 1), target.flatten())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(src)
            n += len(src)
        val_loss, val_exact = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1:3d} | train_loss {total / max(n, 1):.4f} | "
            f"val_loss {val_loss:.4f} | exact_action {val_exact:.3f}"
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "max_prompt_len": args.max_prompt_len,
                    "max_action_len": args.max_action_len,
                },
                out,
            )
    print(f"Saved best checkpoint to {out}")


if __name__ == "__main__":
    main()
