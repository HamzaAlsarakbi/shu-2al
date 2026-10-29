"""`asr` stage: Silero VAD finds speech, Cohere Transcribe Arabic transcribes it.

Input `audio`: 16 kHz mono FLAC/WAV. Output `transcript.json`:
    {"version": 1, "audio_s", "model", "speech": [[start, end], ...],
     "segments": [{"start", "end", "text"}, ...]}
Times are seconds. Each segment is one ASR chunk; the `align` stage adds words.
"""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from .chunking import batches_by_length, max_new_tokens, merge_regions
from .protocol import Reporter, Request, StageError, write_json

SR = 16_000
DEFAULT_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"


def run(req: Request, report: Reporter) -> dict:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    p = req.params
    started = time.monotonic()
    audio = load_audio(req.input("audio"))
    audio_s = len(audio) / SR

    report.progress(0, "detecting speech", force=True)
    stamps = get_speech_timestamps(
        torch.from_numpy(audio),
        load_silero_vad(),
        sampling_rate=SR,
        threshold=p.get("vad_threshold", 0.5),
        min_silence_duration_ms=p.get("min_silence_ms", 300),
        speech_pad_ms=p.get("speech_pad_ms", 100),
        return_seconds=True,
    )
    speech = [(float(s["start"]), float(s["end"])) for s in stamps]
    chunks = merge_regions(speech, p.get("max_chunk_s", 20.0), p.get("max_gap_s", 0.5))
    report.progress(5, f"{len(chunks)} speech chunks", force=True)

    device = pick_device(req.device, torch)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_id = p.get("model", DEFAULT_MODEL)
    processor = AutoProcessor.from_pretrained(model_id)
    model = CohereAsrForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    report.progress(10, "model loaded", force=True)

    texts: list[str] = [""] * len(chunks)
    batches = batches_by_length(chunks, p.get("batch_size", 8))
    for n, batch in enumerate(batches):
        clips = [audio[int(chunks[i][0] * SR) : int(chunks[i][1] * SR)] for i in batch]
        longest = max(chunks[i][1] - chunks[i][0] for i in batch)
        inputs = processor(clips, sampling_rate=SR, return_tensors="pt", language="ar")
        inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens(longest))
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        for i, text in zip(batch, decoded, strict=True):
            texts[i] = text.strip()
        report.progress(10 + 90 * (n + 1) / len(batches), f"batch {n + 1}/{len(batches)}")

    segments = [
        {"start": round(s, 3), "end": round(e, 3), "text": t}
        for (s, e), t in zip(chunks, texts, strict=True)
    ]
    out_path = req.output_dir / "transcript.json"
    write_json(
        out_path,
        {
            "version": 1,
            "audio_s": round(audio_s, 3),
            "model": model_id,
            "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
            "segments": segments,
        },
    )
    elapsed = time.monotonic() - started
    return {
        "outputs": {"transcript": str(out_path)},
        "stats": {
            "audio_s": round(audio_s, 1),
            "speech_s": round(sum(e - s for s, e in speech), 1),
            "chunks": len(chunks),
            "elapsed_s": round(elapsed, 1),
            "rtf": round(elapsed / audio_s, 4) if audio_s else None,
            "peak_vram_mb": peak_vram_mb(torch, device),
        },
    }


def load_audio(path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise StageError("bad_input", f"{path}: expected {SR} Hz audio, got {sr} Hz")
    return np.ascontiguousarray(audio.mean(axis=1))


def pick_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def peak_vram_mb(torch, device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated() / 2**20)
