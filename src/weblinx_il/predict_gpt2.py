"""Generate a WebLINX action with a fine-tuned GPT-2 checkpoint."""
from __future__ import annotations

import argparse

import torch

from .train_gpt2 import ACTION_PREFIX, require_transformers


def main():
    ap = argparse.ArgumentParser(description="Predict a WebLINX action with GPT-2.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device).eval()

    text = args.prompt.strip() + ACTION_PREFIX
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
