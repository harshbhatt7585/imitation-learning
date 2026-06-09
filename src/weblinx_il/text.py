"""Character-level utilities for WebLINX action modeling."""
from __future__ import annotations

PAD = 0
BOS = 1
EOS = 2
UNK = 3


def build_vocab(strings) -> dict[str, int]:
    chars = sorted({ch for s in strings for ch in str(s)})
    return {ch: i + 4 for i, ch in enumerate(chars)}


def encode(text: str, vocab: dict[str, int], max_len: int, add_eos: bool = False):
    ids = [vocab.get(ch, UNK) for ch in str(text)[:max_len]]
    if add_eos and len(ids) < max_len:
        ids.append(EOS)
    length = max(1, len(ids))
    ids = ids + [PAD] * (max_len - len(ids))
    return ids, length


def decode(ids, inv_vocab: dict[int, str]) -> str:
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS:
            break
        if idx >= 4:
            chars.append(inv_vocab.get(idx, ""))
    return "".join(chars)
