"""QLoRA fine-tune of a chat LLM on reference subtitle translations, with a
before/after evaluation on a held-out episode.

    python -m shu2al_worker.ft_llm --base Qwen/Qwen3-8B \\
        --train /data/ft/train.json --test /data/ft/test.json \\
        --brief /data/ft/brief.md --out /data/ft/qwen3-8b

train/test JSON: [{"ar": "...", "fix": "...", "en": "..."}, ...] in episode
order. `ar` is the raw ASR line; `fix` (optional) is the annotated corrected
Arabic; `en` is the annotated English. The model learns to output both the
corrected Arabic and the English for each line. Test items without `en` are
translated but not scored. Writes base.json, tuned.json and scores.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

SYSTEM = """You translate Syrian Arabic TV drama dialogue (Damascene dialect, some Standard
Arabic narration) into natural, idiomatic English subtitles.
The Arabic comes from speech recognition and may contain errors; infer the intended words
from context. Translate meaning and tone, not word for word. Keep lines short.
Use the character names from the show notes.
A line may contain two speakers; mark each with "- ".
For every numbered line, first write the corrected Arabic (fix recognition errors, keep the
dialect), then the English, exactly like this and nothing else:
1. ar: <corrected Arabic>
1. en: <English>

Show notes:
"""

WINDOW = 8
CONTEXT = 3


def windows(items: list[dict], stride: int) -> list[tuple[list[dict], list[dict]]]:
    out = []
    for start in range(0, len(items), stride):
        chunk = items[start : start + WINDOW]
        if chunk:
            out.append((items[max(0, start - CONTEXT) : start], chunk))
    return out


def user_prompt(context: list[dict], chunk: list[dict]) -> str:
    parts = []
    if context:
        parts.append("Previous lines, for context only:")
        parts += [f"- {c['ar']} → {c.get('en', '')}" for c in context]
        parts.append("")
    parts.append("Translate:")
    parts += [f"{i + 1}. {c['ar']}" for i, c in enumerate(chunk)]
    return "\n".join(parts)


def corrected(c: dict) -> str:
    return c.get("fix", c["ar"])


def answer(chunk: list[dict]) -> str:
    return "\n".join(
        f"{i + 1}. ar: {corrected(c)}\n{i + 1}. en: {c['en']}" for i, c in enumerate(chunk)
    )


def parse(text: str, n: int) -> list[str]:
    """English per line from `N. en: ...` (falls back to bare `N. ...`)."""
    en, bare = {}, {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(?:(ar|en)\s*:)?\s*(.*)", line, re.IGNORECASE)
        if not m or not 1 <= int(m.group(1)) <= n:
            continue
        k, tag, body = int(m.group(1)), (m.group(2) or "").lower(), m.group(3).strip()
        if tag == "en":
            en.setdefault(k, body)
        elif not tag:
            bare.setdefault(k, body)
    return [en.get(i + 1, bare.get(i + 1, "")) for i in range(n)]


def messages(brief: str, context, chunk, with_answer: bool) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM + brief},
        {"role": "user", "content": user_prompt(context, chunk)},
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": answer(chunk)})
    return msgs


def translate(model, tok, brief: str, items: list[dict], label: str) -> list[str]:
    import torch

    out: list[str] = []
    model.eval()
    started = time.monotonic()
    for k, (context, chunk) in enumerate(windows(items, WINDOW)):
        # Context uses the model's own earlier output, as in real use.
        ctx = [
            {"ar": c["ar"], "en": out[len(out) - len(context) + j]} for j, c in enumerate(context)
        ]
        prompt = tok.apply_chat_template(
            messages(brief, ctx, chunk, False),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=120 * len(chunk), do_sample=False, repetition_penalty=1.05
            )
        text = tok.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out += parse(text, len(chunk))
        print(f"  {label}: window {k + 1}, {len(out)}/{len(items)} lines", flush=True)
    print(f"  {label}: {time.monotonic() - started:.0f}s", flush=True)
    return out


def chrf(hyps: list[str], items: list[dict]) -> float | None:
    import sacrebleu

    pairs = [(h, it["en"]) for h, it in zip(hyps, items, strict=True) if it.get("en")]
    if not pairs:
        return None
    return round(
        sacrebleu.corpus_chrf([h for h, _ in pairs], [[r for _, r in pairs]], word_order=2).score, 2
    )


def load_model(base: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    return model, tok


def answer_loss(model, ids, n_answer: int, slice_len: int = 64):
    """Mean cross-entropy over the answer tokens only.

    The decoder runs once; its hidden states then go through the 151k-wide
    output layer a slice at a time, and each slice's logits are recomputed in
    backward instead of kept. Full fp32 logits for a ~400-token answer don't
    fit next to an 8B model in 8 GB.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids[None]).last_hidden_state[0, -n_answer - 1 : -1]
    targets = ids[-n_answer:]

    def piece(h, t):
        return torch.nn.functional.cross_entropy(inner.lm_head(h).float(), t, reduction="sum")

    total = sum(
        checkpoint(
            piece, hidden[i : i + slice_len], targets[i : i + slice_len], use_reentrant=False
        )
        for i in range(0, n_answer, slice_len)
    )
    return total / n_answer


def lr_factor(update: int, total: int, warmup: int, cosine: bool) -> float:
    if update < warmup:
        return (update + 1) / warmup
    if not cosine:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * (update - warmup) / max(1, total - warmup)))


def train(
    model,
    tok,
    brief: str,
    items: list[dict],
    epochs: int,
    lr: float,
    rank: int,
    extra=(),
    warmup: float = 0.0,
    cosine: bool = False,
    accum: int = 1,
    save_at: tuple[float, ...] = (),
    out: Path | None = None,
):
    import torch
    from peft import LoraConfig, get_peft_model

    # Not peft's prepare_model_for_kbit_training: it upcasts the embeddings and
    # output layer to fp32 (2.3 GB extra on an 8B model), which 8 GB can't hold.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    examples = []
    # Episode windows with context, plus context-free batches of general
    # dialect sentences (already clean, so their Arabic is unchanged).
    batches = windows(items, WINDOW // 2) + [
        ([], list(extra[i : i + WINDOW])) for i in range(0, len(extra), WINDOW)
    ]
    for context, chunk in batches:
        full = tok.apply_chat_template(
            messages(brief, context, chunk, True), tokenize=False, enable_thinking=False
        )
        prompt = tok.apply_chat_template(
            messages(brief, context, chunk, False),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(full, return_tensors="pt")["input_ids"][0]
        n_answer = len(ids) - len(tok(prompt)["input_ids"])
        examples.append((ids, n_answer))
    print(
        f"training examples: {len(examples)}, longest {max(len(i) for i, _ in examples)} tokens",
        flush=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step, total = 0, epochs * len(examples)
    updates = math.ceil(total / accum)
    warmup_updates = int(warmup * updates)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: lr_factor(u, updates, warmup_updates, cosine)
    )
    save_steps = {round(epoch * len(examples)): epoch for epoch in save_at}
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for i in order:
            ids, n_answer = examples[i]
            loss = answer_loss(model, ids.to(model.device), n_answer)
            (loss / accum).backward()
            step += 1
            if step % accum == 0 or step == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step in save_steps and out is not None:
                model.save_pretrained(out / f"lora-e{save_steps[step]:g}")
            if step % 20 == 0 or step == total:
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"  epoch {epoch + 1} step {step}/{total} loss {loss.item():.3f} "
                    f"({time.monotonic() - started:.0f}s, peak {peak:.1f} GB)",
                    flush=True,
                )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--warmup", type=float, default=0.0, help="fraction of updates to warm up over")
    ap.add_argument("--cosine", action="store_true", help="decay the learning rate to 0")
    ap.add_argument("--accum", type=int, default=1, help="examples per optimizer update")
    ap.add_argument(
        "--save-at", type=float, nargs="*", default=[], help="epochs to also save the adapter at"
    )
    ap.add_argument("--skip-eval", action="store_true", help="only train and save the adapter")
    ap.add_argument("--reuse-base", action="store_true", help="reuse base.json from an earlier run")
    ap.add_argument(
        "--extra", type=Path, help="extra [{ar, en}] sentence pairs (e.g. Levanti) to mix in"
    )
    ap.add_argument("--extra-n", type=int, default=2000)
    args = ap.parse_args()

    train_items = [
        i
        for i in json.loads(args.train.read_text(encoding="utf-8"))
        if i["ar"].strip() and i.get("en")
    ]
    test_items = [i for i in json.loads(args.test.read_text(encoding="utf-8")) if i["ar"].strip()]
    brief = args.brief.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{args.base}: train {len(train_items)} lines, test {len(test_items)} lines", flush=True)

    import torch

    torch.manual_seed(args.seed)
    model, tok = load_model(args.base)
    base_path = args.out / "base.json"
    if args.skip_eval:
        base_out = []
    elif args.reuse_base and base_path.is_file():
        base_out = json.loads(base_path.read_text(encoding="utf-8"))
    else:
        base_out = translate(model, tok, brief, test_items, "base")
        base_path.write_text(json.dumps(base_out, ensure_ascii=False, indent=1), encoding="utf-8")

    extra = []
    if args.extra:
        import random

        extra = json.loads(args.extra.read_text(encoding="utf-8"))
        random.Random(13).shuffle(extra)
        extra = extra[: args.extra_n]
    model = train(
        model,
        tok,
        brief,
        train_items,
        args.epochs,
        args.lr,
        args.rank,
        extra,
        warmup=args.warmup,
        cosine=args.cosine,
        accum=args.accum,
        save_at=tuple(args.save_at),
        out=args.out,
    )
    model.save_pretrained(args.out / "lora")
    if args.skip_eval:
        return
    tuned_out = translate(model, tok, brief, test_items, "tuned")
    (args.out / "tuned.json").write_text(
        json.dumps(tuned_out, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    scores = {"base": chrf(base_out, test_items), "tuned": chrf(tuned_out, test_items)}
    (args.out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(f"chrF++ vs reference: base {scores['base']}, fine-tuned {scores['tuned']}", flush=True)


if __name__ == "__main__":
    main()
