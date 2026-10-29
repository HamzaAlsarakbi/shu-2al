"""`align` stage: word timestamps via CTC forced alignment.

Each ASR segment is aligned against its own slice of audio with an Arabic
wav2vec2 CTC model, so one bad segment can't drift the rest. Adds
`words: [{"w", "start", "end", "score"}]` to every segment it can align and
`align_error` to the ones it can't (the cue builder then falls back to the
segment's own timing).
"""

from __future__ import annotations

import time

from .asr import SR, load_audio, peak_vram_mb, pick_device
from .protocol import Reporter, Request, write_json

DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"


def run(req: Request, report: Reporter) -> dict:
    import json

    import torch
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_alignment_model,
        postprocess_results,
        preprocess_text,
    )

    started = time.monotonic()
    transcript = json.loads(req.input("transcript").read_text(encoding="utf-8"))
    audio = load_audio(req.input("audio"))
    device = pick_device(req.device, torch)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = load_alignment_model(
        device, model_path=req.params.get("model", DEFAULT_MODEL), dtype=dtype
    )
    report.progress(5, "model loaded", force=True)

    segments = transcript["segments"]
    failed = 0
    for n, seg in enumerate(segments):
        seg.pop("words", None)
        seg.pop("align_error", None)
        words = seg["text"].split()
        if not words:
            continue
        clip = audio[int(seg["start"] * SR) : int(seg["end"] * SR)]
        try:
            wav = torch.from_numpy(clip).to(device=device, dtype=dtype)
            with torch.inference_mode():
                emissions, stride = generate_emissions(model, wav, batch_size=4)
            tokens, text_starred = preprocess_text(seg["text"], romanize=False, language="ara")
            path, scores, blank = get_alignments(emissions, tokens, tokenizer)
            spans = get_spans(tokens, path, blank)
            aligned = postprocess_results(text_starred, spans, stride, scores)
        except Exception as e:  # noqa: BLE001 - one bad segment must not sink the episode
            seg["align_error"] = f"{type(e).__name__}: {e}"[:300]
            failed += 1
            continue
        if len(aligned) != len(words):
            seg["align_error"] = f"word count mismatch: {len(aligned)} aligned vs {len(words)}"
            failed += 1
            continue
        seg["words"] = [
            {
                "w": w,
                "start": round(seg["start"] + a["start"], 3),
                "end": round(seg["start"] + a["end"], 3),
                "score": round(float(a["score"]), 3) if "score" in a else None,
            }
            for w, a in zip(words, aligned, strict=True)
        ]
        report.progress(5 + 95 * (n + 1) / len(segments), f"segment {n + 1}/{len(segments)}")

    transcript["align_model"] = req.params.get("model", DEFAULT_MODEL)
    out_path = req.output_dir / "transcript.aligned.json"
    write_json(out_path, transcript)
    return {
        "outputs": {"transcript": str(out_path)},
        "stats": {
            "segments": len(segments),
            "failed": failed,
            "elapsed_s": round(time.monotonic() - started, 1),
            "peak_vram_mb": peak_vram_mb(torch, device),
        },
    }
