"""Run a WebLINX-style GPT-2 action model in a live Selenium browser.

This is a lightweight demo controller, not the official WebLINX environment. It
extracts visible DOM candidates, asks the action model for the next WebLINX-style
action string, parses it, and executes supported browser actions.
"""
from __future__ import annotations

import argparse
import ast
import re
import time
from dataclasses import dataclass

import torch

from .train_gpt2 import ACTION_PREFIX, require_transformers


UID_RE = re.compile(r'uid="([^"]+)"')
TEXT_RE = re.compile(r'text=("(?:\\.|[^"])*")')
URL_RE = re.compile(r'url=("(?:\\.|[^"])*")')
INTENT_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*\(")


@dataclass
class ParsedAction:
    intent: str
    uid: str | None = None
    text: str | None = None
    url: str | None = None


def _device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _decode_quoted(match: re.Match | None) -> str | None:
    if match is None:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return match.group(1).strip('"')


def parse_action(raw: str) -> ParsedAction | None:
    first = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    match = INTENT_RE.match(first)
    if not match:
        return None
    return ParsedAction(
        intent=match.group(1),
        uid=(UID_RE.search(first).group(1) if UID_RE.search(first) else None),
        text=_decode_quoted(TEXT_RE.search(first)),
        url=_decode_quoted(URL_RE.search(first)),
    )


def load_model(checkpoint: str, device: torch.device):
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    return model, tokenizer


def make_driver(headless: bool):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    return webdriver.Chrome(options=opts)


def get_candidates(driver, limit: int) -> str:
    rows = driver.execute_script(
        """
        const selector = [
          'a', 'button', 'input', 'textarea', 'select',
          '[role="button"]', '[onclick]', '[contenteditable="true"]'
        ].join(',');
        const els = Array.from(document.querySelectorAll(selector));
        const out = [];
        let i = 0;
        for (const el of els) {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          if (rect.width < 2 || rect.height < 2) continue;
          if (style.visibility === 'hidden' || style.display === 'none') continue;
          if (rect.bottom < 0 || rect.right < 0 ||
              rect.top > window.innerHeight || rect.left > window.innerWidth) continue;
          const uid = `e${i++}`;
          el.setAttribute('data-agent-uid', uid);
          const attrs = [];
          for (const name of ['type', 'name', 'value', 'placeholder', 'aria-label', 'title']) {
            const value = el.getAttribute(name);
            if (value) attrs.push(`${name}='${value.slice(0, 80)}'`);
          }
          const text = (el.innerText || el.value || el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') || '').replace(/\\s+/g, ' ').trim();
          out.push({
            uid,
            tag: el.tagName.toLowerCase(),
            text: text.slice(0, 120),
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            attrs: attrs.join(' ')
          });
          if (out.length >= arguments[0]) break;
        }
        return out;
        """,
        limit,
    )
    formatted = []
    for row in rows:
        formatted.append(
            f"(uid = {row['uid']}) [[tag]] {row['tag']} [[bbox]] "
            f"x={row['x']} y={row['y']} width={row['width']} height={row['height']} "
            f"[[attributes]] {row['attrs']} [[text]] {row['text']}"
        )
    return "\n".join(formatted)


def build_prompt(driver, instruction: str, history: list[str], candidates: str) -> str:
    size = driver.get_window_size()
    viewport = f"{size.get('height')}h x {size.get('width')}w"
    return "\n".join(
        [
            f"Viewport: {viewport}",
            f"Dialogue: {instruction}",
            f"History: {' '.join(history[-8:])}",
            f"Candidates: {candidates}",
        ]
    )


def generate_action(model, tokenizer, device, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(
        prompt.strip() + ACTION_PREFIX,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def execute_action(driver, action: ParsedAction) -> bool:
    if action.intent == "load" and action.url:
        driver.get(action.url)
        return True

    if action.intent in {"click", "text_input", "paste"} and action.uid:
        from selenium.webdriver.common.by import By

        elements = driver.find_elements(By.CSS_SELECTOR, f'[data-agent-uid="{action.uid}"]')
        if not elements:
            print(f"No element found for uid={action.uid}")
            return False
        el = elements[0]
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
        if action.intent == "click":
            el.click()
            return True
        if action.text is not None:
            el.click()
            try:
                el.clear()
            except Exception:
                pass
            el.send_keys(action.text)
            return True

    if action.intent == "say":
        print(f"Model says: {action.text or ''}")
        return False

    print(f"Unsupported or incomplete action: {action}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Watch a GPT-2 WebLINX-style browser agent.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--url", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--candidate-limit", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = _device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)
    driver = make_driver(headless=args.headless)
    history = [f'load(url="{args.url}")']

    try:
        driver.get(args.url)
        for step in range(args.steps):
            time.sleep(args.pause)
            candidates = get_candidates(driver, args.candidate_limit)
            prompt = build_prompt(driver, args.instruction, history, candidates)
            raw = generate_action(model, tokenizer, device, prompt, args.max_new_tokens)
            action = parse_action(raw)
            print(f"\nstep {step + 1}/{args.steps}")
            print(f"predicted: {raw.strip()}")
            if action is None:
                print("Could not parse action; stopping.")
                break
            history.append(raw.strip().splitlines()[0])
            if not execute_action(driver, action):
                break
    finally:
        if args.headless:
            driver.quit()
        else:
            input("\nPress Enter to close Chrome...")
            driver.quit()


if __name__ == "__main__":
    main()
