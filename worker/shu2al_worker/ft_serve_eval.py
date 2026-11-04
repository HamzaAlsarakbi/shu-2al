"""Translates a test set through a running llama-server with the fine-tune's
exact prompt and decoding, to check that an adapter survives the move to GGUF.

    python -m shu2al_worker.ft_serve_eval --url http://127.0.0.1:8089 \\
        --test /data/ft/test.json --brief /data/ft/brief.md \\
        --out /data/ft/qwen3-8b-r3/gguf-q4.json --grammar

Writes the English per line to --out and the raw replies next to it
(<out>.raw.json). Prints chrF++ when the test set has references.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from itertools import groupby
from pathlib import Path

from .ft_llm import WINDOW, chrf, messages, parse, windows


def line_grammar(n: int) -> str:
    """GBNF for exactly `n` `i. ar:` / `i. en:` pairs, with literal local ids."""
    pairs = ' "\\n" '.join(f"pair{i + 1}" for i in range(n))
    rules = [f"root ::= {pairs}"]
    rules += [f'pair{i + 1} ::= "{i + 1}. ar: " text "\\n{i + 1}. en: " text' for i in range(n)]
    rules.append("text ::= [^\\u0000-\\u001f\\u007f]{0,300}")
    return "\n".join(rules)


def complete(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 - local server
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"llama-server said {e.code}: {e.read().decode()[:500]}") from e


def scenes(items: list[dict]) -> list[list[dict]]:
    """Consecutive lines per scene, so context never crosses a scene boundary."""
    return [list(group) for _, group in groupby(items, key=lambda i: i.get("scene"))]


def translate(args, brief: str, items: list[dict]) -> tuple[list[str], list[dict]]:
    out: list[str] = []
    raw: list[dict] = []
    started = time.monotonic()
    tokens = 0
    batches = [window for scene in scenes(items) for window in windows(scene, WINDOW)]
    for k, (context, chunk) in enumerate(batches):
        # Context uses the model's own earlier output, as in ft_llm.translate.
        ctx = [
            {"ar": c["ar"], "en": out[len(out) - len(context) + j]} for j, c in enumerate(context)
        ]
        body = {
            "messages": messages(brief, ctx, chunk, False),
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "repeat_penalty": args.repeat_penalty,
            "repeat_last_n": 8192,
            "dry_multiplier": 0,
            # Adapter-on and adapter-off passes must not share cached prompts.
            "cache_prompt": False,
            "max_tokens": 120 * len(chunk),
        }
        if args.lora_scale is not None:
            # An empty list leaves the server's default scale in place.
            body["lora"] = [{"id": 0, "scale": args.lora_scale}]
        if args.grammar:
            body["grammar"] = line_grammar(len(chunk))
        reply = complete(args.url, body)
        choice = reply["choices"][0]
        text = choice["message"]["content"]
        tokens += reply.get("usage", {}).get("completion_tokens", 0)
        out += parse(text, len(chunk))
        raw.append({"window": k, "finish": choice.get("finish_reason"), "text": text})
        print(f"  window {k + 1}, {len(out)}/{len(items)} lines", flush=True)
    elapsed = time.monotonic() - started
    print(f"  {elapsed:.0f}s, {tokens / max(elapsed, 1):.0f} tokens/s", flush=True)
    return out, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grammar", action="store_true", help="constrain replies to the line format")
    ap.add_argument(
        "--lora-scale", type=float, help="per-request adapter scale; 0 disables the adapter"
    )
    ap.add_argument("--repeat-penalty", type=float, default=1.05)
    args = ap.parse_args()

    items = [i for i in json.loads(args.test.read_text(encoding="utf-8")) if i["ar"].strip()]
    brief = args.brief.read_text(encoding="utf-8")
    out, raw = translate(args, brief, items)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    args.out.with_suffix(".raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    cut = sum(r["finish"] != "stop" for r in raw)
    empty = sum(not line.strip() for line in out)
    print(f"chrF++ {chrf(out, items)}; {empty} empty lines, {cut} cut-off windows", flush=True)


if __name__ == "__main__":
    main()
