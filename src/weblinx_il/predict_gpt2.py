"""Generate a WebLINX action with a fine-tuned GPT-2 checkpoint."""
from __future__ import annotations

import argparse

import torch

from .gpt2_common import generate_action, pick_device, require_transformers


def main():
    ap = argparse.ArgumentParser(description="Predict a WebLINX action with GPT-2.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device).eval()

    action = generate_action(
        model, tokenizer, args.prompt.strip(), device,
        max_new_tokens=args.max_new_tokens, max_action_tokens=args.max_new_tokens,
    )
    print(action)


if __name__ == "__main__":
    main()
