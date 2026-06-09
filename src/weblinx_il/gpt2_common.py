"""Shared GPT-2 helpers for WebLINX: tokenization, dataset, generation.

Centralizes everything that must be IDENTICAL across train / eval / serve:
  * how a record's fields become input token ids (prompt + action prefix),
  * head-truncation that preserves the leading (most relevant) candidates,
  * greedy action generation + decoding.

This is what keeps the model's training distribution and its deployment
distribution aligned.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .data import assemble_prompt

ACTION_PREFIX = "\n\nAction: "


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        return AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install transformers first: uv sync (or pip install transformers)"
        ) from exc


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _char_estimate(text: str) -> int:
    """Cheap token-count proxy (~1 token per 3 chars) for greedy assembly."""
    return max(1, len(text) // 3)


def assemble_text(fields: dict, max_length: int, max_action_tokens: int) -> str:
    """Prompt text (no action prefix) sized to roughly fit the context window."""
    # Reserve room for the action prefix + action; use a char-budget proxy.
    char_budget = max(64, (max_length - max_action_tokens) )
    return assemble_prompt(fields, _char_estimate, char_budget)


def build_input_ids(
    prompt_text: str,
    action_text: str | None,
    tokenizer,
    max_length: int,
    max_action_tokens: int,
):
    """Tokenize into input_ids (+ labels if action_text given).

    Truncates the PROMPT from the front-keeping side (drops trailing candidates,
    never the header or the action prefix). Returns lists.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, verbose=False)["input_ids"]
    prefix_ids = tokenizer(ACTION_PREFIX, add_special_tokens=False, verbose=False)["input_ids"]

    if action_text is None:
        # Inference: reserve room for generation.
        budget = max_length - len(prefix_ids) - max_action_tokens
        prompt_ids = prompt_ids[: max(1, budget)]
        return prompt_ids + prefix_ids, None

    action_ids = tokenizer(
        str(action_text).strip() + tokenizer.eos_token,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"][:max_action_tokens]
    budget = max_length - len(prefix_ids) - len(action_ids)
    prompt_ids = prompt_ids[: max(1, budget)]  # keep HEAD -> leading candidates
    input_ids = prompt_ids + prefix_ids + action_ids
    labels = [-100] * (len(prompt_ids) + len(prefix_ids)) + action_ids
    return input_ids, labels


class GPT2WebLinxDataset(Dataset):
    """Examples (field dicts from data.load_examples) -> padded LM tensors."""

    def __init__(self, examples, tokenizer, max_length: int, max_action_tokens: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_action_tokens = max_action_tokens

    def __len__(self):
        return len(self.examples)

    def prompt_text(self, idx: int) -> str:
        return assemble_text(self.examples[idx], self.max_length, self.max_action_tokens)

    def gold(self, idx: int) -> str:
        return self.examples[idx]["target"]

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt_text = self.prompt_text(idx)
        print("Prompt Text:", prompt_text)
        input_ids, labels = build_input_ids(
            prompt_text, ex["target"], self.tokenizer, self.max_length, self.max_action_tokens
        )
        attention_mask = [1] * len(input_ids)
        pad = self.max_length - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad
            attention_mask = attention_mask + [0] * pad
            labels = labels + [-100] * pad
        return (
            torch.tensor(input_ids[: self.max_length], dtype=torch.long),
            torch.tensor(attention_mask[: self.max_length], dtype=torch.long),
            torch.tensor(labels[: self.max_length], dtype=torch.long),
        )


@torch.no_grad()
def generate_action(
    model, tokenizer, prompt_text: str, device, max_new_tokens: int,
    max_length: int = 1024, max_action_tokens: int = 128,
) -> str:
    """Greedy-decode a single action string from a prompt."""
    input_ids, _ = build_input_ids(
        prompt_text, None, tokenizer, max_length, max(max_new_tokens, max_action_tokens)
    )
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    out = model.generate(
        input_ids=ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_predictions(
    model, tokenizer, examples, device, max_new_tokens: int,
    max_length: int, max_action_tokens: int, limit: int | None = None,
):
    """Generate predictions for examples; return (preds, golds) string lists."""
    model.eval()
    preds, golds = [], []
    items = examples if limit is None else examples[:limit]
    for ex in items:
        prompt_text = assemble_text(ex, max_length, max_action_tokens)
        pred = generate_action(
            model, tokenizer, prompt_text, device, max_new_tokens,
            max_length, max_action_tokens,
        )
        print("Predictions: ", pred)
        preds.append(pred)
        golds.append(ex["target"])
    return preds, golds
