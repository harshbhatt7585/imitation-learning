"""WebLINX-style turn-level evaluation metrics.

These are the metrics that actually reflect deployment quality, mirroring the
WebLINX benchmark's scoring philosophy:

  * Intent Match (IM)  — did we predict the right action type?
  * Element accuracy   — for element actions, did we pick the right `uid`?
  * Text similarity    — chrF between predicted/gold text (text_input, say, ...).
  * URL match          — exact URL for `load`.
  * Overall score      — per turn: 0 if intent wrong, else the argument score for
                         that intent; averaged over turns. This is the headline
                         "can the agent do the task" number.

`chrf` is implemented dependency-free (character n-gram F-score, beta=2), the
same family WebLINX uses for text arguments.

All functions take *parsed* `Action`s (see actions.py) or raw strings.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .actions import Action, parse_action


# --------------------------------------------------------------------------- #
# Text similarity (chrF)
# --------------------------------------------------------------------------- #
def _char_ngrams(text: str, n: int) -> Counter:
    text = text or ""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1)) if len(text) >= n else Counter()


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score in [0, 1]. Returns 1.0 if both empty."""
    hyp, ref = (hyp or ""), (ref or "")
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / max(sum(h.values()), 1))
        recalls.append(overlap / max(sum(r.values()), 1))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


# --------------------------------------------------------------------------- #
# Per-turn scoring
# --------------------------------------------------------------------------- #
def _as_action(x) -> Optional[Action]:
    if x is None or isinstance(x, Action):
        return x
    return parse_action(str(x))


def argument_score(pred: Action, gold: Action) -> float:
    """Score the arguments of `pred` against `gold`, assuming intents match.

    - element + text (text_input): mean(uid match, text chrF)
    - element only (click/submit/...): uid match
    - text only (say): text chrF
    - load: url exact match
    - otherwise (scroll, etc.): 1.0 (intent match is the whole signal)
    """
    intent = gold.intent
    if intent == "load":
        return 1.0 if (pred.url or "") == (gold.url or "") else 0.0
    if gold.is_element and gold.is_text:
        uid_ok = 1.0 if pred.uid is not None and pred.uid == gold.uid else 0.0
        return 0.5 * uid_ok + 0.5 * chrf(pred.text or "", gold.text or "")
    if gold.is_element:
        return 1.0 if pred.uid is not None and pred.uid == gold.uid else 0.0
    if gold.is_text:
        return chrf(pred.text or "", gold.text or "")
    return 1.0


def turn_score(pred, gold) -> dict:
    """Return per-turn metric dict for a (pred, gold) pair (Actions or strings)."""
    g = _as_action(gold)
    p = _as_action(pred)
    if g is None:
        return {}  # ungradeable gold; caller should skip
    parse_ok = p is not None
    intent_match = 1.0 if (parse_ok and p.intent == g.intent) else 0.0
    arg = argument_score(p, g) if intent_match else 0.0
    return {
        "intent_match": intent_match,
        "argument_score": arg,
        "overall": intent_match * arg,
        "parse_ok": 1.0 if parse_ok else 0.0,
        "gold_intent": g.intent,
        "is_element": g.is_element,
        "is_text": g.is_text,
        "uid_correct": (
            1.0 if (parse_ok and g.uid is not None and p.uid == g.uid) else 0.0
        ),
        "text_chrf": (
            chrf(p.text or "", g.text or "") if (parse_ok and g.is_text) else None
        ),
    }


@dataclass
class MetricReport:
    n: int
    overall: float
    intent_acc: float
    parse_rate: float
    element_acc: float          # uid accuracy over element-action turns
    text_chrf: float            # mean chrF over text-action turns
    n_element: int
    n_text: int
    per_intent: dict            # intent -> {n, overall, intent_acc}

    def pretty(self) -> str:
        lines = [
            f"turns           : {self.n}",
            f"OVERALL score   : {self.overall:.3f}",
            f"intent accuracy : {self.intent_acc:.3f}",
            f"parse rate      : {self.parse_rate:.3f}",
            f"element acc(uid): {self.element_acc:.3f}  (n={self.n_element})",
            f"text chrF       : {self.text_chrf:.3f}  (n={self.n_text})",
            "per-intent (n / intent_acc / overall):",
        ]
        for intent, d in sorted(self.per_intent.items(), key=lambda kv: -kv[1]["n"]):
            lines.append(
                f"  {intent:<12} {d['n']:>5}  {d['intent_acc']:.3f}  {d['overall']:.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "overall": self.overall,
            "intent_acc": self.intent_acc,
            "parse_rate": self.parse_rate,
            "element_acc": self.element_acc,
            "text_chrf": self.text_chrf,
            "n_element": self.n_element,
            "n_text": self.n_text,
            "per_intent": self.per_intent,
        }


def aggregate(preds, golds) -> MetricReport:
    """Aggregate metrics over parallel lists of predicted/gold action strings."""
    rows = []
    for p, g in zip(preds, golds):
        row = turn_score(p, g)
        if row:
            rows.append(row)
    n = len(rows)
    if n == 0:
        return MetricReport(0, 0, 0, 0, 0, 0, 0, 0, {})

    def mean(key):
        return sum(r[key] for r in rows) / n

    elem = [r for r in rows if r["is_element"]]
    text = [r for r in rows if r["is_text"] and r["text_chrf"] is not None]

    per_intent: dict = {}
    for r in rows:
        d = per_intent.setdefault(
            r["gold_intent"], {"n": 0, "intent_acc": 0.0, "overall": 0.0}
        )
        d["n"] += 1
        d["intent_acc"] += r["intent_match"]
        d["overall"] += r["overall"]
    for d in per_intent.values():
        d["intent_acc"] /= d["n"]
        d["overall"] /= d["n"]

    return MetricReport(
        n=n,
        overall=mean("overall"),
        intent_acc=mean("intent_match"),
        parse_rate=mean("parse_ok"),
        element_acc=(sum(r["uid_correct"] for r in elem) / len(elem)) if elem else 0.0,
        text_chrf=(sum(r["text_chrf"] for r in text) / len(text)) if text else 0.0,
        n_element=len(elem),
        n_text=len(text),
        per_intent=per_intent,
    )
