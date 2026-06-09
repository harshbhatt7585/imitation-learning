"""Inspect WebLINX records from Hugging Face."""
from __future__ import annotations

import argparse

from .data import DATASET_ID, require_datasets


def main():
    ap = argparse.ArgumentParser(description="Inspect raw WebLINX dataset records.")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    load_dataset = require_datasets()
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    for i, row in enumerate(ds):
        print(f"\n--- record {i} ---")
        for key, value in dict(row).items():
            text = str(value)
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"{key}: {text}")
        if i + 1 >= args.n:
            break


if __name__ == "__main__":
    main()
