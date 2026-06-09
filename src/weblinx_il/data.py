"""WebLINX data loading, canonical prompt construction, and truncation.

One place defines how a raw WebLINX record becomes a model prompt + gold action,
so training, evaluation, and the live browser controller stay in lockstep.

Key design point — **candidate-aware truncation**: the candidates list is ordered
roughly by relevance (the gold element is usually near the top). When the prompt
exceeds the model context we therefore keep the header + as many *leading*
candidate lines as fit, rather than blindly left-truncating (which would throw
away the very element the model must select).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .text import BOS, encode
from .actions import parse_action


DATASET_ID = "McGill-NLP/weblinx"

HISTORY_MAX_CHARS = 1500          # cap dialogue/history text before tokenizing
_CAND_UID_RE = re.compile(r"\(\s*uid\s*=\s*([^)\s]+)\s*\)")
_CAND_SECTION_RE = re.compile(r"\[\[(.*?)\]\]")
_BBOX_RE = re.compile(
    r"x=(?P<x>[-+]?\d+(?:\.\d+)?)\s+"
    r"y=(?P<y>[-+]?\d+(?:\.\d+)?)\s+"
    r"width=(?P<width>[-+]?\d+(?:\.\d+)?)\s+"
    r"height=(?P<height>[-+]?\d+(?:\.\d+)?)"
)


def require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install Hugging Face datasets first: uv sync (or pip install datasets)"
        ) from exc
    return load_dataset


def _stringify(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _gold_action(record: dict) -> str:
    return _stringify(
        record.get("action")
        or record.get("output_target")
        or record.get("target")
        or record.get("output")
    )


def record_fields(record: dict) -> dict | None:
    """Extract structured prompt fields + gold action from a WebLINX record."""
    target = _gold_action(record)
    if not target.strip():
        return None
    candidates = _stringify(record.get("candidates"))
    cand_lines = [ln for ln in candidates.splitlines() if ln.strip()]
    # Support both native WebLINX records and pre-built prompt records.
    dialogue = _stringify(record.get("utterances"))
    history = _stringify(record.get("action_history"))
    if not dialogue and not history and not cand_lines:
        pre = record.get("prompt") or record.get("input") or record.get("query")
        if not pre:
            return None
        return {
            "viewport": "",
            "dialogue": "",
            "history": "",
            "candidate_lines": [],
            "prebuilt_prompt": _stringify(pre),
            "target": target,
        }
    return {
        "viewport": _stringify(record.get("viewport")),
        "dialogue": dialogue,
        "history": history,
        "candidate_lines": cand_lines,
        "prebuilt_prompt": None,
        "target": target,
    }


def candidate_uids(cand_lines) -> list[str]:
    """Pull the candidate uids (in order) from candidate lines."""
    uids = []
    for ln in cand_lines:
        m = _CAND_UID_RE.search(ln)
        if m:
            uids.append(m.group(1))
    return uids


def _candidate_sections(line: str) -> dict[str, str]:
    matches = list(_CAND_SECTION_RE.finditer(line))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        sections[name] = line[start:end].strip()
    return sections


def parse_candidate_line(line: str, index: int, gold_uid: str | None = None) -> dict:
    """Parse one WebLINX candidate line into structured fields."""
    uid_match = _CAND_UID_RE.search(line)
    uid = uid_match.group(1) if uid_match else None
    sections = _candidate_sections(line)
    bbox = None
    bbox_match = _BBOX_RE.search(sections.get("bbox", ""))
    if bbox_match:
        bbox = {k: float(v) for k, v in bbox_match.groupdict().items()}
    return {
        "index": index,
        "uid": uid,
        "tag": sections.get("tag", ""),
        "xpath": sections.get("xpath", ""),
        "text": sections.get("text", ""),
        "bbox": bbox,
        "attributes": sections.get("attributes", ""),
        "children": sections.get("children", ""),
        "raw": line,
        "is_gold": bool(gold_uid and uid == gold_uid),
    }


def action_to_struct(raw_action: str) -> dict:
    """Parse a WebLINX action string into model-head-friendly fields."""
    action = parse_action(raw_action)
    if action is None:
        return {
            "raw": raw_action,
            "parse_ok": False,
            "intent": None,
            "uid": None,
            "text": None,
            "url": None,
            "args": {},
            "is_element": False,
            "is_text": False,
        }
    return {
        "raw": raw_action,
        "parse_ok": True,
        "intent": action.intent,
        "uid": action.uid,
        "text": action.text,
        "url": action.url,
        "args": action.args,
        "is_element": action.is_element,
        "is_text": action.is_text,
    }


def structured_example(fields: dict) -> dict:
    """Convert `record_fields` output into explicit supervised-action fields.

    The resulting record is suitable for an action-head model:

    - `dialogue`, `history`, `viewport`: language/context inputs
    - `candidates`: current DOM action choices with `is_gold` labels
    - `action`: parsed target action
    - `labels`: normalized targets for intent, element, text, and URL heads
    """
    action = action_to_struct(fields["target"])
    candidates = [
        parse_candidate_line(line, i, gold_uid=action["uid"])
        for i, line in enumerate(fields.get("candidate_lines", []))
    ]
    gold_candidate_index = None
    for cand in candidates:
        if cand["is_gold"]:
            gold_candidate_index = cand["index"]
            break
    prompt = fields.get("prebuilt_prompt") or full_prompt(fields)
    return {
        "viewport": fields.get("viewport", ""),
        "dialogue": fields.get("dialogue", ""),
        "history": fields.get("history", ""),
        "prompt": prompt,
        "candidates": candidates,
        "action": action,
        "labels": {
            "intent": action["intent"],
            "element_uid": action["uid"],
            "element_index": gold_candidate_index,
            "text": action["text"],
            "url": action["url"],
            "raw_action": fields["target"],
        },
    }


def load_structured_examples(
    split: str, limit: int | None = None, dataset_id: str = DATASET_ID
) -> list[dict]:
    return [structured_example(ex) for ex in load_examples(split, limit, dataset_id)]


def write_jsonl(rows, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(viewport: str, dialogue: str, history: str, candidates: str) -> str:
    """Canonical prompt text. Used identically at train / eval / serve time."""
    return "\n".join(
        [
            f"Viewport: {viewport}",
            f"Dialogue: {dialogue[:HISTORY_MAX_CHARS]}",
            f"History: {history[:HISTORY_MAX_CHARS]}",
            "Candidates:",
            candidates,
        ]
    )


def assemble_prompt(fields: dict, count_tokens, budget: int) -> str:
    """Build a prompt that fits `budget` tokens, keeping leading candidates.

    `count_tokens(str) -> int` lets this stay tokenizer-agnostic.
    """
    if fields.get("prebuilt_prompt"):
        return fields["prebuilt_prompt"]
    header = "\n".join(
        [
            f"Viewport: {fields['viewport']}",
            f"Dialogue: {fields['dialogue'][:HISTORY_MAX_CHARS]}",
            f"History: {fields['history'][:HISTORY_MAX_CHARS]}",
            "Candidates:",
        ]
    )
    used = count_tokens(header)
    kept: list[str] = []
    for line in fields["candidate_lines"]:
        cost = count_tokens(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return header + "\n" + "\n".join(kept)


def full_prompt(fields: dict) -> str:
    """Untruncated prompt (used by the char-GRU baseline / inspection)."""
    if fields.get("prebuilt_prompt"):
        return fields["prebuilt_prompt"]
    return build_prompt(
        fields["viewport"],
        fields["dialogue"],
        fields["history"],
        "\n".join(fields["candidate_lines"]),
    )


def load_examples(
    split: str, limit: int | None = None, dataset_id: str = DATASET_ID
) -> list[dict]:
    """Load structured WebLINX examples (fields + gold action)."""
    load_dataset = require_datasets()
    ds = load_dataset(dataset_id, split=split, streaming=limit is not None)
    out: list[dict] = []
    for row in ds:
        fields = record_fields(dict(row))
        if fields is not None:
            out.append(fields)
        if limit is not None and len(out) >= limit:
            break
    if not out:
        raise ValueError(
            f"No usable examples in {dataset_id}:{split}. "
            "Run `python -m weblinx_il.inspect` to inspect available fields."
        )
    return out


def load_pairs(
    split: str, limit: int | None = None, dataset_id: str = DATASET_ID
) -> list[tuple[str, str]]:
    """(prompt, action) pairs using the full (untruncated) prompt — legacy/GRU."""
    return [(full_prompt(f), f["target"]) for f in load_examples(split, limit, dataset_id)]


@dataclass
class WebLinxTextConfig:
    max_prompt_len: int = 2048
    max_action_len: int = 192


class WebLinxTextDataset(Dataset):
    def __init__(self, pairs, vocab, cfg: WebLinxTextConfig):
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
