"""Instruction-conditioned BC policy: (screenshot, utterance) -> click heatmap.

A CNN encodes the frame; a char-level GRU encodes the instruction; the text
vector modulates the CNN feature map via FiLM (per-channel scale + shift) before
a 1x1 conv produces a ``grid x grid`` click-location logit map. Conditioning on
the instruction lets the policy pick the *correct* target when several are
present (e.g. which button/link the task names).

Training is spatial cross-entropy over grid cells; inference takes the argmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import PAD_IDX


def conv_block(cin: int, cout: int, downsample: bool = True) -> nn.Sequential:
    stride = 2 if downsample else 1
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TextEncoder(nn.Module):
    """Char embeddings -> GRU -> final hidden state (B, hidden)."""

    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD_IDX)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)                                    # (B,L,emb)
        lengths = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _out, h = self.gru(packed)                               # h: (1,B,hidden)
        return h.squeeze(0)                                      # (B,hidden)


class FiLM(nn.Module):
    """Produce per-channel (gamma, beta) from a conditioning vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_params = nn.Linear(cond_dim, 2 * channels)
        self.channels = channels

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_params(cond).chunk(2, dim=1)       # each (B,C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)                # (B,C,1,1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


class ClickPolicy(nn.Module):
    """Instruction-conditioned CNN -> flattened (grid*grid) click distribution."""

    def __init__(
        self,
        vocab_size: int,
        grid: int = 32,
        channels=(32, 64, 128, 256),
        text_emb: int = 32,
        text_hidden: int = 64,
        head_dropout: float = 0.10,
    ):
        super().__init__()
        self.grid = grid
        layers = []
        cin = 3
        for i, cout in enumerate(channels):
            layers.append(conv_block(cin, cout, downsample=i < 3))
            cin = cout
        self.encoder = nn.Sequential(*layers)          # input 128 -> 16x16 feature map
        self.text = TextEncoder(vocab_size, emb=text_emb, hidden=text_hidden)
        self.film = FiLM(text_hidden, cin)
        self.head = nn.Sequential(
            nn.Conv2d(cin + 2, cin, kernel_size=3, padding=1),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=head_dropout),
            nn.Conv2d(cin, 1, kernel_size=1),
        )

    def _coord_channels(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalized x/y coordinate planes for CoordConv-style localization."""
        b, _c, h, w = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).expand(b, -1, -1, -1)
        return torch.cat((feat, coords), dim=1)

    def forward(
        self, x: torch.Tensor, text_ids: torch.Tensor, text_len: torch.Tensor
    ) -> torch.Tensor:
        """Return flat logits of shape (B, grid*grid)."""
        feat = self.encoder(x)                          # (B,C,h,w)
        cond = self.text(text_ids, text_len)            # (B,text_hidden)
        feat = self.film(feat, cond)
        feat = self._coord_channels(feat)
        heat = self.head(feat)                          # (B,1,h,w)
        heat = F.interpolate(
            heat, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        return heat.flatten(1)                           # (B, grid*grid)

    @torch.no_grad()
    def predict_cell(
        self, x: torch.Tensor, text_ids: torch.Tensor, text_len: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, text_ids, text_len).argmax(dim=1)
