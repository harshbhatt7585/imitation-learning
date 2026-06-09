"""Generate a WebLINX action with a fine-tuned GPT-2 checkpoint."""
from __future__ import annotations

import argparse

import torch

from .gpt2_common import generate_action, pick_device, require_transformers
from .live_browser import get_candidates, make_driver, make_prompt


def main():
    ap = argparse.ArgumentParser(description="Predict a WebLINX action with GPT-2.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--prompt", help="Full WebLINX prompt including candidates.")
    ap.add_argument("--url", help="Open this URL and scrape candidates into the prompt.")
    ap.add_argument("--instruction", help="Task instruction used with --url.")
    ap.add_argument("--candidate-limit", type=int, default=40)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if bool(args.prompt) == bool(args.url):
        ap.error("provide exactly one of --prompt or --url")
    if args.url and not args.instruction:
        ap.error("--instruction is required when using --url")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    if args.url:
        driver = make_driver(headless=True)
        try:
            driver.get(args.url)
            candidates, _uids, _candidate_rows = get_candidates(driver, args.candidate_limit)
            prompt = make_prompt(
                driver,
                args.instruction,
                [
                    f'say(speaker="instructor", utterance="{args.instruction}")',
                    f'load(url="{args.url}")',
                ],
                candidates,
            )
        finally:
            driver.quit()
    else:
        prompt = args.prompt.strip()

    if args.show_prompt:
        print("PROMPT:\n" + prompt + "\n")

    action = generate_action(
        model, tokenizer, prompt, device,
        max_new_tokens=args.max_new_tokens, max_action_tokens=args.max_new_tokens,
    )
    print(action)


if __name__ == "__main__":
    main()
