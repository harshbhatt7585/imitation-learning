"""Canonical WebLINX action representation, parsing, and normalization.

A single source of truth shared by training-data handling, evaluation metrics,
and the live browser controller, so they never disagree on what an action means.

WebLINX action strings look like::

    click(uid="a1b2-...")
    text_input(text="biotechnology", uid="a1b2-...")
    load(url="https://example.com/")
    say(speaker="navigator", utterance="Sure, on it")
    submit(uid="a1b2-...")
    scroll(x=0, y=400)

The parser is tolerant of arbitrary argument order, escaped quotes inside
strings, single- or double-quoted values, and unquoted numerics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Intent name: leading identifier before "(".
_INTENT_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# key="value" / key='value' with escaped-quote support, or key=<number>.
_KV_RE = re.compile(
    r"""
    (\w+)                        # key
    \s*=\s*
    (?:
        "((?:\\.|[^"\\])*)"      # 1: double-quoted value
      | '((?:\\.|[^'\\])*)'      # 2: single-quoted value
      | ([-+]?\d+(?:\.\d+)?)     # 3: bare number
    )
    """,
    re.VERBOSE,
)

# Intents whose primary argument is a target DOM element (carry a uid).
ELEMENT_INTENTS = frozenset(
    {"click", "text_input", "submit", "change", "hover", "clear", "select", "paste"}
)
# Intents whose primary argument is free text we score for similarity.
TEXT_INTENTS = frozenset({"text_input", "say", "paste", "change"})


def _unescape(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape") if "\\" in value else value


@dataclass
class Action:
    """A parsed WebLINX action."""

    intent: str
    args: dict[str, str] = field(default_factory=dict)

    @property
    def uid(self) -> Optional[str]:
        return self.args.get("uid")

    @property
    def text(self) -> Optional[str]:
        # text_input/paste use `text`; say uses `utterance`.
        return self.args.get("text", self.args.get("utterance"))

    @property
    def url(self) -> Optional[str]:
        return self.args.get("url")

    @property
    def is_element(self) -> bool:
        return self.intent in ELEMENT_INTENTS

    @property
    def is_text(self) -> bool:
        return self.intent in TEXT_INTENTS

    def __repr__(self) -> str:  # concise, for logs
        inner = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.intent}({inner})"


def parse_action(raw: str) -> Optional[Action]:
    """Parse a (possibly multi-line) action string into an `Action`, or None.

    Only the first line is considered, matching how models emit one action.
    """
    if not raw or not raw.strip():
        return None
    first = raw.strip().splitlines()[0].strip()
    m = _INTENT_RE.match(first)
    if not m:
        return None
    intent = m.group(1)
    args: dict[str, str] = {}
    for key, dq, sq, num in _KV_RE.findall(first):
        if dq != "":
            args[key] = _unescape(dq)
        elif sq != "":
            args[key] = _unescape(sq)
        elif num != "":
            args[key] = num
        else:
            args[key] = ""
    return Action(intent=intent, args=args)
