"""Scripted oracle experts that solve simple MiniWoB++ click tasks.

These provide the demonstrations for behavioral cloning. The oracle inspects
the DOM in the observation to decide *where* to click; the BC model never sees
the DOM, only the screenshot, so it must learn to reproduce these clicks from
pixels alone.

Target-selection strategy (in priority order):
  1. The element MiniWoB explicitly flags as ``targeted`` (flags[2]).
  2. A leaf element whose visible text matches the instruction's ``target``
     field (case-insensitive).
  3. Fallback: the single clickable BUTTON / A leaf on the page.
"""
from __future__ import annotations

from typing import Any, Optional

from .env import element_center, get_fields

# DOM element flag indices (see miniwob.observation.serialize_dom_element).
FLAG_TARGETED = 2
FLAG_IS_LEAF = 3

CLICKABLE_TAGS = {"BUTTON", "A"}


def _is_targeted(element: dict[str, Any]) -> bool:
    return int(element["flags"][FLAG_TARGETED]) == 1


def _is_leaf(element: dict[str, Any]) -> bool:
    return int(element["flags"][FLAG_IS_LEAF]) == 1


def find_target_element(obs: dict) -> Optional[dict[str, Any]]:
    """Return the DOM element the oracle should click, or None if unsure."""
    elements = list(obs.get("dom_elements", ()))
    if not elements:
        return None

    # 1. Explicitly targeted element.
    targeted = [e for e in elements if _is_targeted(e)]
    if targeted:
        return targeted[0]

    # 2. Text match against the instruction's target field.
    target_text = get_fields(obs).get("target", "").strip().lower()
    if target_text:
        matches = [
            e for e in elements if e.get("text", "").strip().lower() == target_text
        ]
        leaves = [e for e in matches if _is_leaf(e)]
        if leaves:
            return leaves[0]
        if matches:
            return matches[0]

    # 3. Single clickable leaf fallback.
    clickable = [
        e
        for e in elements
        if e.get("tag", "").upper() in CLICKABLE_TAGS and _is_leaf(e)
    ]
    if clickable:
        return clickable[0]
    return None


def oracle_click(obs: dict) -> Optional[tuple[float, float]]:
    """Return the (x, y) pixel the oracle would click, or None if it can't."""
    target = find_target_element(obs)
    if target is None:
        return None
    return element_center(target)
