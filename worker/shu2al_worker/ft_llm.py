"""QLoRA fine-tune of a chat LLM on reference subtitle translations, with a
before/after evaluation on a held-out episode.

    python -m shu2al_worker.ft_llm --base Qwen/Qwen3-8B \\
        --train /data/ft/train.json --test /data/ft/test.json \\
        --brief /data/ft/brief.md --out /data/ft/qwen3-8b

train/test JSON: [{"ar": "...", "en": "..."}, ...] in episode order; `en` is
the reference (e.g. a frontier model's translation). Test items without `en`
are translated but not scored. Writes base.json, tuned.json and scores.json.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

SYSTEM = """You translate Syrian Arabic TV drama dialogue (Damascene dialect, some Standard
Arabic narration) into natural, idiomatic English subtitles.
The Arabic comes from speech recognition and may contain errors; infer the intended words
from context. Translate meaning and tone, not word for word. Keep lines short.
Use the character names from the show notes.
A line may contain two speakers; mark each with "- ".
Answer with exactly one numbered English line per numbered Arabic line, nothing else.

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


def answer(chunk: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {c['en']}" for i, c in enumerate(chunk))


def parse(text: str, n: int) -> list[str]:
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= n:
            found.setdefault(int(m.group(1)), m.group(2).strip())
    return [found.get(i + 1, "") for i in range(n)]


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
                **enc, max_new_tokens=60 * len(chunk), do_sample=False, repetition_penalty=1.05
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


def train(model, tok, brief: str, items: list[dict], epochs: int, lr: float, rank: int):
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
    for context, chunk in windows(items, WINDOW // 2):
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
    loss_fn = torch.nn.CrossEntropyLoss()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step, total = 0, epochs * len(examples)
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for i in order:
            ids, n_answer = examples[i]
            ids = ids.to(model.device)
            # Only the answer is learned, so only compute logits for it: with a
            # 151k-token vocabulary, logits over the whole prompt don't fit in 8 GB.
            logits = model(input_ids=ids[None], logits_to_keep=n_answer + 1).logits[0]
            loss = loss_fn(logits[:-1].float(), ids[-n_answer:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--reuse-base", action="store_true", help="reuse base.json from an earlier run")
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

    model, tok = load_model(args.base)
    base_path = args.out / "base.json"
    if args.reuse_base and base_path.is_file():
        base_out = json.loads(base_path.read_text(encoding="utf-8"))
    else:
        base_out = translate(model, tok, brief, test_items, "base")
        base_path.write_text(json.dumps(base_out, ensure_ascii=False, indent=1), encoding="utf-8")

    model = train(model, tok, brief, train_items, args.epochs, args.lr, args.rank)
    tuned_out = translate(model, tok, brief, test_items, "tuned")
    (args.out / "tuned.json").write_text(
        json.dumps(tuned_out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    model.save_pretrained(args.out / "lora")

    scores = {"base": chrf(base_out, test_items), "tuned": chrf(tuned_out, test_items)}
    (args.out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(f"chrF++ vs reference: base {scores['base']}, fine-tuned {scores['tuned']}", flush=True)


if __name__ == "__main__":
    main()
