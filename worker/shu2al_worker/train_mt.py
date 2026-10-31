"""Fine-tune NLLB-200 for Levantine (Syrian) Arabic → English subtitles.

    python -m shu2al_worker.train_mt --data /data/mt --out /models/mt/nllb-levantine

Training data (personal, non-commercial licences):
  * UFAL Parallel Corpus of North Levantine 1.0: OpenSubtitles lines rewritten
    in North Levantine by native speakers, with the English side (apc.txt / eng.txt).
  * Levanti: human-written dialect sentences with English (levanti_v1.csv);
    synthetic and Egyptian rows are skipped.

Trains LoRA adapters (fits an 8 GB GPU), reports chrF on a held-out UFAL split
before and after, and saves a merged model for the `mt` stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

SRC_LANG = "apc_Arab"
TGT_LANG = "eng_Latn"


def load_pairs(data: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Returns (ufal_pairs, levanti_pairs)."""
    apc = (data / "ufal" / "apc.txt").read_text(encoding="utf-8").splitlines()
    eng = (data / "ufal" / "eng.txt").read_text(encoding="utf-8").splitlines()
    if len(apc) != len(eng):
        raise SystemExit(f"UFAL files are not aligned: {len(apc)} vs {len(eng)} lines")
    ufal = [
        (a.strip(), e.strip()) for a, e in zip(apc, eng, strict=True) if a.strip() and e.strip()
    ]

    levanti: list[tuple[str, str]] = []
    csv_path = data / "levanti" / "levanti_v1.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames or []}
            ar_col = next(
                (cols[c] for c in ("arabic", "ar", "levantine", "text") if c in cols), None
            )
            en_col = next((cols[c] for c in ("english", "en", "translation") if c in cols), None)
            synth_col = next((cols[c] for c in ("synthesized", "synthetic") if c in cols), None)
            dialect_col = cols.get("dialect")
            if ar_col is None or en_col is None:
                print(f"levanti: unknown columns {reader.fieldnames}; skipping", flush=True)
            else:
                for row in reader:
                    flag = (row.get(synth_col) or "").strip().lower() if synth_col else ""
                    if flag in ("1", "true", "yes"):
                        continue
                    if dialect_col and (row.get(dialect_col) or "").strip().lower() == "egyptian":
                        continue
                    a, e = (row[ar_col] or "").strip(), (row[en_col] or "").strip()
                    if a and e:
                        levanti.append((a, e))
    return ufal, levanti


def chrf(model, tokenizer, pairs, device, batch_size=32, beams=4) -> float:
    import sacrebleu
    import torch

    hyps = []
    model.eval()
    for i in range(0, len(pairs), batch_size):
        src = [a for a, _ in pairs[i : i + batch_size]]
        enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                num_beams=beams,
                max_new_tokens=128,
            )
        hyps += tokenizer.batch_decode(out, skip_special_tokens=True)
    return sacrebleu.corpus_chrf(hyps, [[e for _, e in pairs]], word_order=2).score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--max-train", type=int, default=0, help="cap training pairs (0 = all)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(13)
    ufal, levanti = load_pairs(args.data)
    random.shuffle(ufal)
    test, ufal_train = ufal[: args.test_size], ufal[args.test_size :]
    train = ufal_train + levanti
    random.shuffle(train)
    if args.max_train:
        train = train[: args.max_train]
    print(
        f"pairs: ufal {len(ufal)} (test {len(test)}), levanti {len(levanti)}, train {len(train)}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base, dtype=torch.bfloat16).to(device)

    base_chrf = chrf(model, tokenizer, test[:300], device)
    print(f"baseline chrF++ (300 held-out UFAL lines): {base_chrf:.1f}", flush=True)
    torch.cuda.empty_cache()

    # NLLB's 256k-token vocabulary makes the logits the memory hog; recompute
    # activations instead of storing them so 8 GB is enough.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            task_type="SEQ_2_SEQ_LM",
        ),
    )
    model.print_trainable_parameters()

    per_step = args.batch_size * args.accum
    steps = int(len(train) * args.epochs / per_step)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.03), steps)
    model.train()
    started = time.monotonic()
    cursor = 0
    for step in range(steps):
        for _ in range(args.accum):
            batch = train[cursor : cursor + args.batch_size]
            cursor = (cursor + args.batch_size) % max(len(train) - args.batch_size, 1)
            enc = tokenizer(
                [a for a, _ in batch],
                text_target=[e for _, e in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc).loss / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == steps - 1:
            rate = (step + 1) / (time.monotonic() - started)
            eta = (steps - step - 1) / rate / 60
            print(
                f"step {step + 1}/{steps} loss {loss.item() * args.accum:.3f} "
                f"({rate:.2f} it/s, eta {eta:.0f} min, "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)",
                flush=True,
            )

    model = model.merge_and_unload()
    tuned_chrf = chrf(model, tokenizer, test[:300], device)
    print(f"fine-tuned chrF++: {tuned_chrf:.1f} (baseline {base_chrf:.1f})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (args.out / "shu2al-train.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "train_pairs": len(train),
                "test_pairs": len(test),
                "chrf_baseline": round(base_chrf, 2),
                "chrf_finetuned": round(tuned_chrf, 2),
                "minutes": round((time.monotonic() - started) / 60, 1),
            },
            indent=1,
        )
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
