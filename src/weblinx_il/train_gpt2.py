"""Fine-tune GPT-2 for WebLINX prompt -> action-string imitation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import DATASET_ID, load_pairs


ACTION_PREFIX = "\n\nAction: "


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install transformers first: uv sync or .venv/bin/python -m pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


class GPT2WebLinxDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length: int, max_action_tokens: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_action_tokens = max_action_tokens

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prompt, action = self.pairs[idx]
        prompt_text = str(prompt).strip() + ACTION_PREFIX
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]
        action_ids = self.tokenizer(
            str(action).strip() + self.tokenizer.eos_token,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"][: self.max_action_tokens]

        prompt_budget = max(1, self.max_length - len(action_ids))
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + action_ids
        labels = [-100] * len(prompt_ids) + action_ids
        attention_mask = [1] * len(input_ids)

        pad = self.max_length - len(input_ids)
        if pad > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad
            attention_mask += [0] * pad
            labels += [-100] * pad

        return (
            torch.tensor(input_ids[: self.max_length], dtype=torch.long),
            torch.tensor(attention_mask[: self.max_length], dtype=torch.long),
            torch.tensor(labels[: self.max_length], dtype=torch.long),
        )


def _device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total += loss.item() * len(input_ids)
            n += len(input_ids)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune GPT-2 on WebLINX actions.")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-action-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out-dir", default="runs/weblinx/gpt2")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = _device(args.device)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    print("Loading WebLINX pairs...")
    train_pairs = load_pairs(args.train_split, args.train_limit, args.dataset)
    val_pairs = load_pairs(args.val_split, args.val_limit, args.dataset)
    train_ds = GPT2WebLinxDataset(
        train_pairs, tokenizer, args.max_length, args.max_action_tokens
    )
    val_ds = GPT2WebLinxDataset(
        val_pairs, tokenizer, args.max_length, args.max_action_tokens
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    step = 0

    print(
        f"Device: {device} | model={args.model_name} | train={len(train_ds)} "
        f"val={len(val_ds)} | batch={args.batch_size} grad_accum={args.grad_accum}"
    )
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        opt.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            (loss / args.grad_accum).backward()
            total += loss.item() * len(input_ids)
            n += len(input_ids)
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        if step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        val_loss = evaluate(model, val_loader, device)
        train_loss = total / max(n, 1)
        print(f"epoch {epoch + 1:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
        if math.isfinite(val_loss) and val_loss < best:
            best = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    if best < float("inf"):
        print(f"Saved best GPT-2 checkpoint to {out_dir} (val_loss {best:.4f})")
    else:
        print("No finite validation loss produced; no GPT-2 checkpoint was saved.")


if __name__ == "__main__":
    main()
