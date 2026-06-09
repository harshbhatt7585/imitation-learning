"""Fine-tune GPT-2 for WebLINX prompt -> action imitation (deployment-grade).

Improvements over a plain LM fine-tune:
  * candidate-aware prompt assembly (keeps the gold element in context),
  * warmup + cosine LR schedule with gradient clipping,
  * checkpointing on the **task metric** (overall WebLINX-style score from
    generated actions), not teacher-forced loss,
  * early stopping on that metric and a saved JSON eval report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DATASET_ID, load_examples
from .gpt2_common import (
    ACTION_PREFIX,  # noqa: F401  (re-exported for predict/live imports)
    GPT2WebLinxDataset,
    generate_predictions,
    pick_device,
    require_transformers,
)
from .metrics import aggregate


def teacher_forced_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total += loss.item() * len(input_ids)
            n += len(input_ids)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a causal LM on WebLINX actions.")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-action-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--metric-eval-limit", type=int, default=200,
                    help="How many val examples to GENERATE on each epoch (slow).")
    ap.add_argument("--early-stop-patience", type=int, default=2)
    ap.add_argument("--out-dir", default="runs/weblinx/gpt2")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    from transformers import get_cosine_schedule_with_warmup

    device = pick_device(args.device)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    print("Loading WebLINX examples...")
    train_ex = load_examples(args.train_split, args.train_limit, args.dataset)
    val_ex = load_examples(args.val_split, args.val_limit, args.dataset)
    train_ds = GPT2WebLinxDataset(train_ex, tokenizer, args.max_length, args.max_action_tokens)
    val_ds = GPT2WebLinxDataset(val_ex, tokenizer, args.max_length, args.max_action_tokens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), total_steps
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, micro_step = -1.0, -1, 0

    print(
        f"Device: {device} | model={args.model_name} | train={len(train_ds)} "
        f"val={len(val_ds)} | batch={args.batch_size} grad_accum={args.grad_accum} "
        f"| optim_steps={total_steps}"
    )
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        opt.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for i, (input_ids, attention_mask, labels) in enumerate(pbar):
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            (loss / args.grad_accum).backward()
            total += loss.item() * len(input_ids)
            n += len(input_ids)
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                micro_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        # --- evaluate: teacher-forced loss + GENERATED task metric ---
        val_loss = teacher_forced_loss(model, val_loader, device)
        preds, golds = generate_predictions(
            model, tokenizer, val_ex, device,
            max_new_tokens=args.max_action_tokens, max_length=args.max_length,
            max_action_tokens=args.max_action_tokens, limit=args.metric_eval_limit,
        )
        print("Predictions::", preds)
        report = aggregate(preds, golds)
        print(
            f"epoch {epoch + 1:3d} | train_loss {total / max(n, 1):.4f} | "
            f"val_loss {val_loss:.4f} | OVERALL {report.overall:.3f} | "
            f"intent {report.intent_acc:.3f} | elem {report.element_acc:.3f} | "
            f"chrF {report.text_chrf:.3f}"
        )

        if report.overall > best_score:
            best_score, best_epoch = report.overall, epoch
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            payload = report.to_dict()
            payload.update({"epoch": epoch + 1, "val_loss": val_loss})
            (out_dir / "eval_report.json").write_text(json.dumps(payload, indent=2))
            print(f"  ✓ new best OVERALL {best_score:.3f} -> saved to {out_dir}")
        elif epoch - best_epoch >= args.early_stop_patience:
            print(f"Early stop: no OVERALL improvement for {args.early_stop_patience} epochs.")
            break

    if best_epoch >= 0:
        print(f"Done. Best OVERALL {best_score:.3f} at epoch {best_epoch + 1} -> {out_dir}")
    else:
        print("No checkpoint saved (no successful epoch).")


if __name__ == "__main__":
    main()
