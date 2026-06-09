"""Character-level text encoding for instruction conditioning.

Utterances are short and use a small character set, so a char-level vocab is
robust (handles arbitrary target words, e.g. the random link text in click-link)
with no tokenizer dependency. The vocab is built from the training data and
stored in the checkpoint so evaluation encodes identically.

Index convention: 0 = PAD, 1 = UNK, real chars start at 2.
"""
from __future__ import annotations

PAD_IDX = 0
UNK_IDX = 1


def build_vocab(strings) -> dict[str, int]:
    """Build a char -> index map from an iterable of strings."""
    chars = sorted({c for s in strings for c in str(s)})
    return {c: i + 2 for i, c in enumerate(chars)}


def encode(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], int]:
    """Encode a string to a fixed-length list of ids plus its true length.

    Truncates to ``max_len``; shorter strings are PAD-padded on the right.
    Returns (ids, length) where length>=1 (GRU packing requires length>=1).
    """
    ids = [vocab.get(c, UNK_IDX) for c in str(text)[:max_len]]
    length = max(len(ids), 1)
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids, length
