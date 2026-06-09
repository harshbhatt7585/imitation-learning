"""Export WebLINX records as structured JSONL for action-head training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DATASET_ID, load_structured_examples, write_jsonl


def main():
    ap = argparse.ArgumentParser(description="Export structured WebLINX examples.")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="data/weblinx/train_structured.jsonl")
    ap.add_argument("--preview", type=int, default=1)
    args = ap.parse_args()

    rows = load_structured_examples(args.split, args.limit, args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out)
    print(f"Wrote {len(rows)} structured examples to {out}")

    for row in rows[: args.preview]:
        print("\n--- preview ---")
        keep = {
            "dialogue": row["dialogue"],
            "history": row["history"][:500],
            "action": row["action"],
            "labels": row["labels"],
            "candidates": row["candidates"][:5],
        }
        print(json.dumps(keep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
