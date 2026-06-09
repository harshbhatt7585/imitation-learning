"""Instruction-conditioned behavioral-cloning training loop.

Trains `ClickPolicy` with spatial cross-entropy over click-location grid cells.
Reports exact-cell accuracy and a tolerance accuracy (predicted cell within 1
cell of the target, i.e. "close enough to land on it"). Early-stops on
validation loss and saves the best checkpoint to ``best.pt`` (plus ``last.pt``).
The checkpoint embeds the vocab and shapes `evaluate.py` needs.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import DemoDataset
from .model import ClickPolicy
from .utils import get_device, load_config, resolve_path, set_seed


def _metric_value(name: str, val_loss: float, val_exact: float, val_tol: float):
    if name == "loss":
        return val_loss, False
    if name == "exact":
        return val_exact, True
    if name in {"within-1", "within_1", "tol"}:
        return val_tol, True
    raise ValueError(f"Unsupported checkpoint_metric: {name}")


def _cells_close(pred: torch.Tensor, target: torch.Tensor, grid: int, tol: int):
    """Chebyshev distance <= tol between pred/target cells on the grid."""
    pgx, pgy = pred % grid, pred // grid
    tgx, tgy = target % grid, target // grid
    return (torch.maximum((pgx - tgx).abs(), (pgy - tgy).abs()) <= tol)


def evaluate_cells(model, loader, device, grid: int, label_smoothing: float = 0.0):
    """Return (mean loss, exact cell acc, tolerance acc) over a loader."""
    model.eval()
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, exact, close, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for img, ids, length, cell, _task in loader:
            img, ids, length, cell = (
                img.to(device), ids.to(device), length.to(device), cell.to(device)
            )
            logits = model(img, ids, length)
            total_loss += crit(logits, cell).item() * len(cell)
            pred = logits.argmax(1)
            exact += (pred == cell).sum().item()
            close += _cells_close(pred, cell, grid, tol=1).sum().item()
            n += len(cell)
    return total_loss / max(n, 1), exact / max(n, 1), close / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train the instruction-conditioned BC policy.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg.get("seed", 0))
    tcfg, mcfg, icfg = cfg["train"], cfg["model"], cfg["image"]
    device = get_device(tcfg["device"])
    grid = mcfg["heatmap_size"]
    print(f"Device: {device}")

    ds = DemoDataset(resolve_path(tcfg["data_dir"]), resize_to=icfg["resize_to"], grid=grid)
    n_val = max(1, int(len(ds) * tcfg["val_frac"]))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    print(f"Dataset: {len(ds)} frames across {ds.task_names} "
          f"({n_train} train / {n_val} val) | vocab={len(ds.vocab)} "
          f"max_text_len={ds.max_text_len}")

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"],
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"])

    model = ClickPolicy(
        vocab_size=len(ds.vocab) + 2,        # +2 for PAD/UNK
        grid=grid,
        channels=tuple(mcfg["channels"]),
        text_emb=mcfg.get("text_emb", 32),
        text_hidden=mcfg.get("text_hidden", 64),
        head_dropout=mcfg.get("head_dropout", 0.10),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    label_smoothing = tcfg.get("label_smoothing", 0.0)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    out_dir = resolve_path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str):
        torch.save(
            {
                "model_state": model.state_dict(),
                "grid": grid,
                "channels": list(mcfg["channels"]),
                "resize_to": icfg["resize_to"],
                "task_names": ds.task_names,
                "vocab": ds.vocab,
                "max_text_len": ds.max_text_len,
                "text_emb": mcfg.get("text_emb", 32),
                "text_hidden": mcfg.get("text_hidden", 64),
                "head_dropout": mcfg.get("head_dropout", 0.10),
            },
            out_dir / name,
        )

    patience = tcfg.get("early_stop_patience", 10)
    checkpoint_metric = tcfg.get("checkpoint_metric", "loss")
    best_score, best_epoch, step = None, 0, 0
    best_loss = float("inf")
    for epoch in range(tcfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{tcfg['epochs']}", leave=False)
        run_loss, run_n = 0.0, 0
        for img, ids, length, cell, _task in pbar:
            img, ids, length, cell = (
                img.to(device), ids.to(device), length.to(device), cell.to(device)
            )
            logits = model(img, ids, length)
            loss = crit(logits, cell)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            run_loss += loss.item() * len(cell)
            run_n += len(cell)
            if step % tcfg["log_every"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = run_loss / max(run_n, 1)
        val_loss, val_exact, val_tol = evaluate_cells(
            model, val_loader, device, grid, label_smoothing=label_smoothing
        )
        print(f"epoch {epoch+1:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | exact {val_exact:.3f} | within-1 {val_tol:.3f}")
        save("last.pt")
        score, higher_is_better = _metric_value(
            checkpoint_metric, val_loss, val_exact, val_tol
        )
        improved = (
            best_score is None
            or (higher_is_better and score > best_score + 1e-4)
            or (not higher_is_better and score < best_score - 1e-4)
        )
        if improved:
            best_score, best_loss, best_epoch = score, val_loss, epoch
            save("best.pt")
        elif epoch - best_epoch >= patience:
            print(
                f"Early stop: no {checkpoint_metric} improvement "
                f"for {patience} epochs."
            )
            break
    print(
        f"Done. Best {checkpoint_metric} {best_score:.4f} "
        f"(val_loss {best_loss:.4f}) at epoch {best_epoch+1}."
    )
    print(f"Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
