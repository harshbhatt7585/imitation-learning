"""Generate a WebLINX-style action string from a prompt."""
from __future__ import annotations

import argparse

import torch

from .model import WebActionSeq2Seq
from .text import decode, encode


def main():
    ap = argparse.ArgumentParser(description="Predict one WebLINX action string.")
    ap.add_argument("--checkpoint", default="runs/weblinx/best.pt")
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    model = WebActionSeq2Seq(vocab_size=len(vocab) + 4)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src, src_len = encode(args.prompt, vocab, ckpt["max_prompt_len"])
    out = model.generate(
        torch.tensor([src], dtype=torch.long),
        torch.tensor([src_len], dtype=torch.long),
        max_len=ckpt["max_action_len"],
    )[0]
    print(decode(out.tolist(), inv_vocab))


if __name__ == "__main__":
    main()
