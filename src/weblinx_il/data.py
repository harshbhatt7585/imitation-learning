"""Dataset helpers for lightweight WebLINX prompt-to-action training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .text import BOS, encode


DATASET_ID = "McGill-NLP/weblinx"


def require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install Hugging Face datasets first: "
            ".venv/bin/python -m pip install datasets"
        ) from exc
    return load_dataset


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def record_to_prompt_target(record: dict) -> tuple[str, str] | None:
    """Extract a prompt/action pair from known WebLINX record variants."""
    prompt = record.get("prompt") or record.get("input") or record.get("query")
    if prompt is None and "action_history" in record:
        pieces = [
            f"Viewport: {_stringify(record.get('viewport'))}",
            f"Dialogue: {_stringify(record.get('utterances'))}",
            f"History: {_stringify(record.get('action_history'))}",
            f"Candidates: {_stringify(record.get('candidates'))}",
        ]
        prompt = "\n".join(pieces)
    target = (
        record.get("output_target")
        or record.get("target")
        or record.get("action")
        or record.get("output")
    )
    if prompt is None or target is None:
        return None
    prompt, target = _stringify(prompt), _stringify(target)
    if not prompt.strip() or not target.strip():
        return None
    return prompt, target


def load_pairs(
    split: str,
    limit: int | None = None,
    dataset_id: str = DATASET_ID,
) -> list[tuple[str, str]]:
    load_dataset = require_datasets()
    ds = load_dataset(dataset_id, split=split, streaming=limit is not None)
    pairs: list[tuple[str, str]] = []
    for row in ds:
        pair = record_to_prompt_target(dict(row))
        if pair is not None:
            pairs.append(pair)
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(
            f"No prompt/action pairs found in {dataset_id}:{split}. "
            "Run `python -m weblinx_il.inspect` to inspect available fields."
        )
    return pairs


@dataclass
class WebLinxTextConfig:
    max_prompt_len: int = 2048
    max_action_len: int = 192


class WebLinxTextDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        vocab: dict[str, int],
        cfg: WebLinxTextConfig,
    ):
        self.pairs = pairs
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        prompt, action = self.pairs[idx]
        src, src_len = encode(prompt, self.vocab, self.cfg.max_prompt_len)
        action_ids, _ = encode(action, self.vocab, self.cfg.max_action_len, add_eos=True)
        dec_in = [BOS] + action_ids[:-1]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(src_len, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(action_ids, dtype=torch.long),
        )
