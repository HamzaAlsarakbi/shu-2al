"""`mt` stage: translate cue lines with a seq2seq model (fine-tuned NLLB).

Input `lines`: JSON list of Arabic strings. Output `mt.json`: {"en": [...]},
one English string per input line, same order. Empty inputs stay empty.
"""

from __future__ import annotations

import json
import time

from .asr import peak_vram_mb, pick_device
from .protocol import Reporter, Request, write_json

SRC_LANG = "apc_Arab"
TGT_LANG = "eng_Latn"


def run(req: Request, report: Reporter) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    started = time.monotonic()
    lines: list[str] = json.loads(req.input("lines").read_text(encoding="utf-8"))
    p = req.params
    device = pick_device(req.device, torch)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(p["model"], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(p["model"], dtype=dtype).to(device).eval()
    report.progress(5, "model loaded", force=True)

    todo = [i for i, line in enumerate(lines) if line.strip()]
    out = [""] * len(lines)
    batch_size = int(p.get("batch_size", 32))
    for n in range(0, len(todo), batch_size):
        idx = todo[n : n + batch_size]
        enc = tokenizer(
            [lines[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=192,
        ).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                num_beams=int(p.get("beams", 4)),
                max_new_tokens=160,
                no_repeat_ngram_size=4,
            )
        for i, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True), strict=True):
            out[i] = text.strip()
        report.progress(5 + 95 * min(n + batch_size, len(todo)) / max(len(todo), 1), "translating")

    out_path = req.output_dir / "mt.json"
    write_json(out_path, {"model": p["model"], "en": out})
    return {
        "outputs": {"mt": str(out_path)},
        "stats": {
            "lines": len(lines),
            "elapsed_s": round(time.monotonic() - started, 1),
            "peak_vram_mb": peak_vram_mb(torch, device),
        },
    }
