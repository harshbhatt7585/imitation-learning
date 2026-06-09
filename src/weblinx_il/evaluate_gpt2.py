"""Evaluate a fine-tuned GPT-2 checkpoint on WebLINX action strings."""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data import DATASET_ID, load_pairs
from .train_gpt2 import GPT2WebLinxDataset, require_transformers


def _device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description="Evaluate GPT-2 WebLINX action model.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-action-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device).eval()

    pairs = load_pairs(args.split, args.limit, args.dataset)
    ds = GPT2WebLinxDataset(pairs, tokenizer, args.max_length, args.max_action_tokens)
    loader = DataLoader(ds, batch_size=args.batch_size)

    total_loss, total_tokens, n = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            tokens = int((labels != -100).sum().item())
            total_loss += out.loss.item() * max(tokens, 1)
            total_tokens += tokens
            n += len(input_ids)

    loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(loss)).item()
    print(f"split={args.split} examples={n} action_token_loss={loss:.4f} ppl={ppl:.2f}")


if __name__ == "__main__":
    main()
