"""Run a WebLINX-style action model in a live Selenium browser.

A lightweight deployment controller (not the official WebLINX environment). It
extracts visible DOM candidates *in the same format the model trained on*, asks
the model for the next action, and executes it — with deployment safeguards:

  * shared prompt template + action parser (no train/serve drift),
  * the predicted ``uid`` is validated against the real candidates and snapped to
    the closest match when the model emits a near-miss,
  * unsupported / unparseable / ungrounded actions stop the run instead of doing
    something arbitrary.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .actions import Action, parse_action
from .data import build_prompt
from .gpt2_common import generate_action, pick_device, require_transformers
from .model import WebActionSeq2Seq
from .text import decode, encode


class ActionPredictor:
    def predict(self, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError


class GPT2Predictor(ActionPredictor):
    def __init__(self, checkpoint: str, device: torch.device, max_length: int = 1024):
        AutoModelForCausalLM, AutoTokenizer = require_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
        self.device = device
        self.max_length = max_length

    def predict(self, prompt: str, max_new_tokens: int) -> str:
        return generate_action(
            self.model, self.tokenizer, prompt, self.device,
            max_new_tokens=max_new_tokens, max_length=self.max_length,
            max_action_tokens=max_new_tokens,
        )


class GRUPredictor(ActionPredictor):
    def __init__(self, checkpoint: str, device: torch.device):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.vocab = ckpt["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_prompt_len = ckpt["max_prompt_len"]
        self.max_action_len = ckpt["max_action_len"]
        self.device = device
        self.model = WebActionSeq2Seq(vocab_size=len(self.vocab) + 4)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()

    def predict(self, prompt: str, max_new_tokens: int) -> str:
        src, src_len = encode(prompt, self.vocab, self.max_prompt_len)
        out = self.model.generate(
            torch.tensor([src], dtype=torch.long, device=self.device),
            torch.tensor([src_len], dtype=torch.long, device=self.device),
            max_len=min(max_new_tokens, self.max_action_len),
        )[0]
        return decode(out.tolist(), self.inv_vocab)


def load_predictor(checkpoint: str, device: torch.device) -> ActionPredictor:
    path = Path(checkpoint)
    if path.is_file() or checkpoint.endswith(".pt"):
        print(f"Loading GRU action model from {checkpoint}")
        return GRUPredictor(checkpoint, device)
    print(f"Loading GPT-2 action model from {checkpoint}")
    return GPT2Predictor(checkpoint, device)


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


# JS: collect visible, interactable elements + a short xpath, tag them with uid.
_COLLECT_JS = r"""
const limit = arguments[0];
function xpath(el) {
  if (el === document.body) return '/html/body';
  let ix = 0; const sibs = el.parentNode ? el.parentNode.childNodes : [];
  for (let i = 0; i < sibs.length; i++) {
    const s = sibs[i];
    if (s === el) {
      const p = el.parentNode && el.parentNode !== document
        ? xpath(el.parentNode) : '';
      return `${p}/${el.tagName.toLowerCase()}[${ix + 1}]`;
    }
    if (s.nodeType === 1 && s.tagName === el.tagName) ix++;
  }
  return '';
}
const selector = ['a','button','input','textarea','select',
  '[role="button"]','[onclick]','[contenteditable="true"]'].join(',');
const els = Array.from(document.querySelectorAll(selector));
const out = []; let i = 0;
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
  for (const name of ['type','name','value','placeholder','aria-label','title']) {
    const v = el.getAttribute(name);
    if (v) attrs.push(`${name}='${v.slice(0, 80)}'`);
  }
  const text = (el.innerText || el.value || el.getAttribute('aria-label') ||
                el.getAttribute('placeholder') || '').replace(/\s+/g, ' ').trim();
  out.push({uid, tag: el.tagName.toLowerCase(), xpath: xpath(el),
            x: Math.round(rect.x), y: Math.round(rect.y),
            width: Math.round(rect.width), height: Math.round(rect.height),
            attrs: attrs.join(' '), text: text.slice(0, 120)});
  if (out.length >= limit) break;
}
return out;
"""


def get_candidates(driver, limit: int):
    """Return (candidates_text, uid_list) in the WebLINX candidate format."""
    rows = driver.execute_script(_COLLECT_JS, limit)
    lines, uids = [], []
    for r in rows:
        uids.append(r["uid"])
        lines.append(
            f"(uid = {r['uid']}) [[tag]] {r['tag']} [[xpath]] {r['xpath']} "
            f"[[bbox]] x={r['x']} y={r['y']} width={r['width']} height={r['height']} "
            f"[[attributes]] {r['attrs']} [[text]] {r['text']}"
        )
    return "\n".join(lines), uids


def make_prompt(driver, instruction: str, history: list[str], candidates: str) -> str:
    size = driver.get_window_size()
    viewport = f"{size.get('height')}h x {size.get('width')}w"
    return build_prompt(viewport, instruction, " ".join(history[-8:]), candidates)


def ground_uid(action: Action, uids: list[str]) -> str | None:
    """Snap the predicted uid to a real candidate, or None if impossible."""
    if action.uid is None:
        return None
    if action.uid in uids:
        return action.uid
    # near-miss: predicted uid is a prefix/substring of a real candidate (or v.v.)
    for u in uids:
        if action.uid in u or u in action.uid:
            return u
    return None


def execute_action(driver, action: Action, uids: list[str]) -> bool:
    if action.intent == "load" and action.url:
        driver.get(action.url)
        return True

    if action.intent == "say":
        print(f"Model says: {action.text or ''}")
        return False

    if action.intent in {"click", "text_input", "paste", "submit", "change"}:
        from selenium.webdriver.common.by import By

        uid = ground_uid(action, uids)
        if uid is None:
            print(f"  ! predicted uid {action.uid!r} is not a real candidate; stopping.")
            return False
        if uid != action.uid:
            print(f"  ~ snapped uid {action.uid!r} -> {uid!r}")
        els = driver.find_elements(By.CSS_SELECTOR, f'[data-agent-uid="{uid}"]')
        if not els:
            print(f"  ! no live element for uid={uid}; stopping.")
            return False
        el = els[0]
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
        if action.intent in {"click", "submit"}:
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

    print(f"  ! unsupported or incomplete action: {action}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Watch a WebLINX-style browser agent.")
    ap.add_argument("--checkpoint", default="runs/weblinx/gpt2")
    ap.add_argument("--url", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--candidate-limit", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    predictor = load_predictor(args.checkpoint, device)
    driver = make_driver(headless=args.headless)
    history = [f'load(url="{args.url}")']

    try:
        driver.get(args.url)
        for step in range(args.steps):
            time.sleep(args.pause)
            candidates, uids = get_candidates(driver, args.candidate_limit)
            prompt = make_prompt(driver, args.instruction, history, candidates)
            raw = predictor.predict(prompt, args.max_new_tokens)
            action = parse_action(raw)
            print(f"\nstep {step + 1}/{args.steps}")
            print(f"predicted: {raw.strip()!r}")
            if action is None:
                print("  ! could not parse an action; stopping.")
                break
            history.append(repr(action))
            if not execute_action(driver, action, uids):
                break
    finally:
        if args.headless:
            driver.quit()
        else:
            input("\nPress Enter to close Chrome...")
            driver.quit()


if __name__ == "__main__":
    main()
