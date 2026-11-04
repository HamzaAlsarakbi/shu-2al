"""Compares two systems' translations with a paired block bootstrap on chrF++.

    python -m shu2al_worker.ft_score --test /data/ft/test.json \\
        --a /data/ft/qwen3-8b/base.json --b /data/ft/qwen3-8b-r3/tuned.json

Lines are resampled in scenes (or contiguous blocks) because neighbouring
subtitle lines are not independent. A difference is only real if its interval
excludes 0.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def blocks(kept: list[dict], size: int) -> list[list[int]]:
    """Whole scenes when the test set has them, else fixed runs of lines."""
    if kept and "scene" in kept[0]:
        groups: dict[str, list[int]] = {}
        for k, item in enumerate(kept):
            groups.setdefault(item["scene"], []).append(k)
        return list(groups.values())
    return [list(range(i, min(i + size, len(kept)))) for i in range(0, len(kept), size)]


def score(metric, hyps: list[str], refs: list[str]) -> float:
    return metric.corpus_score(hyps, [refs]).score


def main() -> None:
    from sacrebleu.metrics import CHRF

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument(
        "--block", type=int, default=15, help="lines per block if the test set has no scenes"
    )
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    test = [i for i in json.loads(args.test.read_text(encoding="utf-8")) if i["ar"].strip()]
    a_out = json.loads(args.a.read_text(encoding="utf-8"))
    b_out = json.loads(args.b.read_text(encoding="utf-8"))
    kept = [k for k, item in enumerate(test) if item.get("en")]
    refs = [test[k]["en"] for k in kept]
    a = [a_out[k] for k in kept]
    b = [b_out[k] for k in kept]

    metric = CHRF(word_order=2)
    score_a, score_b = score(metric, a, refs), score(metric, b, refs)
    groups = blocks([test[k] for k in kept], args.block)
    rng = random.Random(args.seed)
    deltas = []
    for _ in range(args.resamples):
        picked = [k for block in rng.choices(groups, k=len(groups)) for k in block]
        sample_refs = [refs[k] for k in picked]
        delta = score(metric, [b[k] for k in picked], sample_refs) - score(
            metric, [a[k] for k in picked], sample_refs
        )
        deltas.append(delta)
    deltas.sort()
    low = deltas[int(0.025 * len(deltas))]
    high = deltas[int(0.975 * len(deltas)) - 1]
    print(f"{len(kept)} lines, {len(groups)} blocks")
    print(f"a {score_a:.2f}  b {score_b:.2f}  b-a {score_b - score_a:+.2f}")
    print(f"95% interval for b-a: [{low:+.2f}, {high:+.2f}]")


if __name__ == "__main__":
    main()
