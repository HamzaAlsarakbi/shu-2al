"""Turning VAD speech regions into ASR chunks (pure, no torch)."""

from __future__ import annotations

Region = tuple[float, float]


def merge_regions(regions: list[Region], max_chunk_s: float, max_gap_s: float) -> list[Region]:
    """Greedily joins consecutive speech regions into chunks for the ASR model.

    Regions merge while the silence between them is at most `max_gap_s` and the
    merged chunk stays within `max_chunk_s`. A single region longer than
    `max_chunk_s` is split into equal parts, since the ASR model loops on long
    inputs. Chunk boundaries become the coarse subtitle timestamps.
    """
    chunks: list[Region] = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        for part in _split(start, end, max_chunk_s):
            close = chunks and part[0] - chunks[-1][1] <= max_gap_s
            if close and part[1] - chunks[-1][0] <= max_chunk_s:
                chunks[-1] = (chunks[-1][0], part[1])
            else:
                chunks.append(part)
    return chunks


def _split(start: float, end: float, max_len: float) -> list[Region]:
    n = max(1, -int(-(end - start) // max_len))  # ceil
    step = (end - start) / n
    return [(start + i * step, start + (i + 1) * step) for i in range(n)]


def batches_by_length(chunks: list[Region], batch_size: int) -> list[list[int]]:
    """Groups chunk indices into batches of similar length, longest first, so
    padding stays small and an OOM shows up on the first batch."""
    order = sorted(range(len(chunks)), key=lambda i: chunks[i][1] - chunks[i][0], reverse=True)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def max_new_tokens(duration_s: float) -> int:
    """Generous token budget for a chunk: fast Arabic speech is about 3 words/s,
    and about 2 tokens per word. Caps runaway decoding loops."""
    return min(448, 32 + int(duration_s * 12))
