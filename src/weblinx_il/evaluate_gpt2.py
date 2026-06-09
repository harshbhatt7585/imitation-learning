"""Evaluate a fine-tuned GPT-2 WebLINX agent with real, deployment-grade metrics.

Unlike a loss/perplexity check, this *generates* the next action for each held-out
turn and scores it the way WebLINX does: intent match, element (uid) accuracy,
text chrF, and an overall turn score. Also reports teacher-forced perplexity for
reference, and can dump a few qualitative examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DATASET_ID, load_examples
from .gpt2_common import (
    GPT2WebLinxDataset,
    generate_predictions,
    pick_device,
    require_transformers,
)
from .metrics import aggregate


def perplexity(model, loader, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            tokens = int((labels != -100).sum().item())
            total_loss += out.loss.item() * max(tokens, 1)
            total_tokens += tokens
    loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(loss)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate GPT-2 WebLINX agent (real metrics).")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-action-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--show", type=int, default=5, help="Print N qualitative examples.")
    ap.add_argument("--report-out", default=None, help="Optional path to write JSON report.")
    ap.add_argument("--no-perplexity", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    examples = load_examples(args.split, args.limit, args.dataset)
    print(f"Generating actions for {len(examples)} held-out turns on {device}...")
    preds, golds = generate_predictions(
        model, tokenizer, examples, device,
        max_new_tokens=args.max_action_tokens, max_length=args.max_length,
        max_action_tokens=args.max_action_tokens,
    )
    report = aggregate(preds, golds)

    print(f"\n=== WebLINX action metrics ({args.split}, n={len(examples)}) ===")
    print(report.pretty())

    if not args.no_perplexity:
        ds = GPT2WebLinxDataset(examples, tokenizer, args.max_length, args.max_action_tokens)
        loader = DataLoader(ds, batch_size=args.batch_size)
        print(f"\naction perplexity: {perplexity(model, loader, device):.2f}")

    for i in range(min(args.show, len(preds))):
        print(f"\n--- example {i} ---")
        print(f"gold: {golds[i]}")
        print(f"pred: {preds[i]}")

    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nSaved report -> {args.report_out}")


if __name__ == "__main__":
    main()
