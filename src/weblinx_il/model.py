"""Small text action model for WebLINX-style prompt -> action imitation."""
from __future__ import annotations

import torch
import torch.nn as nn

from .text import BOS, EOS, PAD


class WebActionSeq2Seq(nn.Module):
    """GRU encoder-decoder baseline for WebLINX action strings."""

    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.enc_to_dec = nn.Linear(hidden * 2, hidden)
        self.decoder = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def encode(self, src: torch.Tensor, src_len: torch.Tensor) -> torch.Tensor:
        emb = self.embed(src)
        src_len = src_len.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len, batch_first=True, enforce_sorted=False
        )
        _out, h = self.encoder(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return torch.tanh(self.enc_to_dec(h_cat)).unsqueeze(0)

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        dec_in: torch.Tensor,
    ) -> torch.Tensor:
        h0 = self.encode(src, src_len)
        out, _h = self.decoder(self.embed(dec_in), h0)
        return self.head(out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        max_len: int = 192,
    ) -> torch.Tensor:
        h = self.encode(src, src_len)
        token = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
        out_ids = []
        for _ in range(max_len):
            out, h = self.decoder(self.embed(token), h)
            token = self.head(out[:, -1]).argmax(dim=1, keepdim=True)
            out_ids.append(token)
            if bool((token == EOS).all()):
                break
        return torch.cat(out_ids, dim=1) if out_ids else token.new_zeros((src.shape[0], 0))
