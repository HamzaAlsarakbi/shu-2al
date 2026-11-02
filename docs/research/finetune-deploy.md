# Deploying the Syrian-drama QLoRA adapter

Research date: **2026-11-02**. Research only: no remote execution, model conversion, training, inference benchmark, package installation, or production change was performed. Commands and code below are proposed procedures. **Measured locally** means existing project results supplied in the task or recorded in DESIGN, not measurements made for this report. **Estimate / inference** identifies recommendations that still need a 3070 experiment.

## Recommendation

Keep llama.cpp. First deploy **the exact Qwen/Qwen3-8B base, Q4_K_M, plus a separate F16 GGUF LoRA**, using the trained `N. ar:` / `N. en:` format with a per-request grammar. Give it eight input cues, three previous raw-ASR/English pairs, the training system prompt plus the same show brief, and thinking disabled. Start at **8,192 context tokens and one server slot**, reducing to 4,096 only after measuring token lengths. Make this an opt-in translation profile; preserve `qwen9b` as the default until episode-level comparison passes.

This is a recommendation about the smallest useful deployment experiment, **not evidence that separate Q4 + LoRA preserves the 45.7 chrF++ score**. Compare it with a clean full-precision-base merge followed by Q4_K_M and Q5_K_M quantization. There is no established quality ordering for this adapter. Separate adapters make diagnosis and disabling the adapter for helper tasks easier; merged weights can eliminate adapter inference work but cannot switch back to the base per request.

The existing advantage is promising: round 3's 45.7 versus production's 39.9 is +5.8 chrF++, but it compares different prompts, models, runtimes and postprocessing. The 165 scored E2 lines, shared annotator style and round 2→3 change of only +0.2 do not establish a general +6 gain. Treat the NF4 HF result as the conversion reference and the current complete pipeline as the product baseline. See [DESIGN, round 1/2](../DESIGN.md#annotate-then-fine-tune-round-2-2026-11-02-five-episodes-corrected-arabic) and the task's round 3 results.

## What this checkout actually does

These are observations from the files, rather than assumptions from the older design narrative:

| Area | Current behavior and deployment implication |
|---|---|
| [ft_llm.py](../../worker/shu2al_worker/ft_llm.py) | `load_model()` quantizes the base with BnB NF4, double quantization, BF16 compute. PEFT targets seven linear projections, r=16 by default, alpha=2r. `save_pretrained(out / "lora")` saves an adapter; it does not save a standalone merged model or tokenizer. |
| Training examples | `WINDOW=8`, `CONTEXT=3`, stride 4; local cue numbers restart at 1. Context Arabic is raw `ar`, context English is reference `en`. Show brief is appended to `SYSTEM`. `corrected()` falls back to raw ASR if `fix` is absent. Levanti examples also receive the show brief. |
| HF evaluation | Stride 8, previous English comes from the model's own predictions. Greedy generation, repetition penalty 1.05, `max_new_tokens=120*n`, `enable_thinking=False`. `parse()` scores English only, tolerates bare numbered output and takes the first duplicate. Arabic correctness and strict structure are not evaluated. |
| CLI defaults | `ft_llm.py --epochs` defaults to **3**, despite the reported successful runs using **1**. Any reproduction must explicitly use `--epochs 1`. |
| [llm.rs](../../crates/server/src/llm.rs) | `Lease::start()` runs a separate on-demand Docker llama-server, mounts models read-only, uses `-ngl 99 -c <ctx> -fa on --jinja`, accepts extra flags, and tears it down after translate. `chat_json()` hardcodes JSON-schema output, temperature 0.3 and DRY 0.8; thinking is already disabled through `chat_template_kwargs`. |
| [translate.rs](../../crates/server/src/translate.rs) | Default batches are 10–25 cues, breaking at ≥4 s gaps after the minimum. Previous context is **8 corrected-Arabic/English** cues. User JSON contains `show`, `summary`, `previous`, `lines`, global IDs and `max_chars`; the brief is **not** in the system prompt in this implementation. |
| Output and helper calls | JSON requires `lines[{id,ar,en,uncertain}]` plus `scene_summary`. Translation uses up to two attempts, a JSON mismatch check, recursive batch splitting, and ASR/empty-English/uncertain fallback. Final Arabic change is flagged above CER 0.4. The same loaded model handles JSON verify and shorten calls. |
| Helper limitations | `verify()` compares English with **raw ASR**, not corrected Arabic; verifier errors become an empty mismatch result through `unwrap_or_default()`. `shorten()` triggers on the line-wrapping predicate, not every CPS violation, and accepts a rewrite if nonempty and shorter without re-verifying meaning. `fits_two_lines()` relies on `segment::break_lines()`, which already returns at most two lines. |
| [pipeline.rs](../../crates/server/src/pipeline.rs) | Existing `Llm::{Docker, External, Seq2Seq}` and `Options` are enough to host another llama.cpp model. The saved contract is still `translation.json: {lines:[{ar,en,uncertain,overcorrected}]}` and two SRTs. Output existence controls resumption; model/prompt changes do not invalidate old results. |
| Dashboard | [queue.rs](../../crates/server/src/queue.rs) separately constructs Docker settings for `qwen9b` / `moe`. It appends glossary and up to **12** approved examples from [editor.rs](../../crates/server/src/editor.rs) to the brief. CLI flags alone will not configure this path. |

DESIGN §5.5 describes historical API defaults, extra JSON fields and behaviors not implemented here; the later local-only decision and the Rust implementation govern this plan. Earlier [translation research](translation.md) and [ASR research](asr-models.md) supply domain context, not proof of adapter compatibility. Keep ASR, alignment and segmentation fixed during deployment evaluation.

## 1. PEFT → llama.cpp

### Conversion requirements and artifacts

PEFT's checkpoint consists of `adapter_config.json`, adapter tensors, and usually a README, with no original base weights. Keep the entire `lora/` directory and record the **exact training base revision**, tokenizer/chat-template revision, data split, package versions and training arguments. The present script does not pin `revision=` or save this manifest, so recover the training HF snapshot identity before calling a fresh download equivalent. [PEFT checkpoint specification](https://huggingface.co/docs/peft/developer_guides/checkpoint).

Use the converter and server from one tested llama.cpp revision. The upstream head checked for this report was [`1b6935b227b9f9aa89deea8016c0e208f547446f`](https://github.com/ggml-org/llama.cpp/commit/1b6935b227b9f9aa89deea8016c0e208f547446f), dated November 2; this is a **source reference, not a validated deployment build**. Pin the eventual CUDA image digest after testing, since `server-cuda` in this repo is a moving tag.

The conversion environment needs CPU PyTorch, Transformers/AutoConfig, safetensors, and the checkout's `gguf-py` plus conversion dependencies. Use an isolated environment, not the ML worker environment: upstream conversion currently pins Transformers 4.57.6 and Torch 2.11.0, whereas this worker specifies Transformers ≥5.4 and Torch ≥2.15. Follow that checkout's requirement files rather than assembling arbitrary versions. [Converter requirements](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_lora_to_gguf.txt), [HF converter requirements](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_hf_to_gguf.txt), [shared requirements](https://github.com/ggml-org/llama.cpp/blob/master/requirements/requirements-convert_legacy_llama.txt), [worker dependencies](../../worker/pyproject.toml).

`convert_lora_to_gguf.py` reads adapter tensors and the **base model configuration**, not the NF4 weights. `--base` takes an HF directory with configuration/tokenizer files; original weight shards are unnecessary for this conversion. Without it the script obtains AutoConfig from `base_model_name_or_path`; `--base-model-id Qwen/Qwen3-8B` overrides that lookup and `--base`. For reproducibility use a local snapshot: the remote override has no explicit revision argument. [Converter source](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py).

Proposed commands, run from the pinned llama.cpp checkout in a future isolated conversion environment:

```sh
# QWEN3_SNAPSHOT: configuration/tokenizer files from the exact training revision.
# ADAPTER_DIR: this experiment's out/lora directory.
python convert_lora_to_gguf.py \
  --base "$QWEN3_SNAPSHOT" \
  --outtype f16 \
  --outfile "$MODELS/llm/alnadam-r3-lora-f16.gguf" \
  "$ADAPTER_DIR"
```

Use F16 adapter tensors first; do not quantize the small adapter to Q4. The converter also offers BF16/F32/Q8_0, but F16 minimizes memory without adding an integer quantizer. **Computed** from the [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) and this script's seven targets: 43,646,976 LoRA parameters, approximately **83.25 MiB at F16** (metadata/runtime allocations extra). Check actual tensor dtypes and output size; PEFT may have kept adapter parameters in FP32 during training. Conversion stores alpha; runtime scale **1.0** should mean the trained adapter, not 2.0: alpha/r=32/16 is already part of LoRA's internal scaling. [Runtime scaling implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp).

The base GGUF must represent **Qwen/Qwen3-8B**, not Qwen3.5-9B, Qwen3-8B-Base, a later instruct variant, a different fine-tune, or another tokenizer revision. The matching public [official Qwen3-8B GGUF repository](https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main) exists, including Q4_K_M and Q5_K_M. Safest artifact provenance is to convert/quantize the training snapshot yourself; a public GGUF is acceptable after verifying its base provenance and template. Qwen documents the GGUF path in its [llama.cpp guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md).

### Qwen3 issues: distinguish architectures

- Current conversion explicitly registers `Qwen3ForCausalLM`, including Qwen3-8B. This adapter uses ordinary linear LoRA targets, with no added vocabulary, `modules_to_save`, DoRA, rsLoRA or per-layer rank patterns in the recipe. That is a relatively simple conversion case, but successful loading still needs an output comparison. [Qwen converter implementation](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py).
- Qwen3.5 conversion reports include missing `LoraTorchTensor.contiguous()` ([#20450](https://github.com/ggml-org/llama.cpp/issues/20450)) and unsupported reshaping during V-head reordering ([#21125](https://github.com/ggml-org/llama.cpp/issues/21125)). These concern **Qwen3.5 hybrid attention**, not proof that this dense Qwen3-8B adapter is broken. The #20450 reporter explicitly says Qwen3 previously worked. Neither report validates this checkpoint.
- Qwen3-8B MLX-fused exports have reported unmappable `lm_head.biases` ([#14467](https://github.com/ggml-org/llama.cpp/issues/14467)); this is a different export route from the current PEFT adapter. Avoid treating MLX or Qwen3-VL failures as this recipe's expected behavior.
- A multi-adapter memory-pool failure reproduced on Qwen3 was closed via a fix in December 2025 ([#18050](https://github.com/ggml-org/llama.cpp/issues/18050)). Start with one adapter and a current tested build. Old examples also use a different `--lora-scaled` syntax; inspect the pinned binary's help.

Before an episode run, inspect conversion logs for all **36×7 target modules** and both halves, verify rank/alpha, and test adapter-on versus adapter-off on a known improved line. A load-success message alone is insufficient. If conversion fails, retain the original checkpoint and try the clean HF merge route below; do not silently drop failing target tensors.

### Serving, per-request selection, and what “hot swap” means

The current argument parser accepts `--lora PATH`, and `--lora-scaled PATH:SCALE` with comma-separated entries for multiple adapters. Historical two-argument examples are not the current syntax. [Argument parser](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp).

```sh
llama-server \
  -m /models/llm/Qwen3-8B-Q4_K_M.gguf \
  --lora /models/llm/alnadam-r3-lora-f16.gguf \
  --lora-init-without-apply \
  -ngl 99 -c 8192 -np 1 -fa on --jinja \
  --host 127.0.0.1 --port 8081
# Alternative startup scaling: --lora-scaled /models/llm/alnadam-r3-lora-f16.gguf:1.0
```

With Docker, preserve the existing host loopback port mapping and use the container's listening address/port as `Lease::start()` already does. The command above describes a host process.

**Yes, per-request LoRA is supported by llama-server**, including the OpenAI-compatible chat path as a llama.cpp extension. Supply this alongside messages and grammar:

```json
{"lora": [{"id": 0, "scale": 1.0}]}
```

To run a JSON helper on the same **Qwen3-8B base without the adapter**, send `"lora": []`. Do not omit the field and assume it disables LoRA. Listed IDs receive their scale; unlisted adapters receive zero. `GET /lora-adapters` supplies IDs. The shared [request-schema implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-schema.cpp) defines these semantics; [chat request parsing](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp) passes through server-specific fields. This is not portable OpenAI API behavior: capability-check the pinned server and external URLs.

`POST /lora-adapters` changes **global scales of already loaded adapters**. `--lora-init-without-apply` starts their scales at zero. These mechanisms permit toggling/mixing preloaded adapters without restarting; the documented endpoint does **not** load an arbitrary new file from a request path. Restart the lease for a new adapter artifact or base. Different adapter configurations are not batched together. [Server adapter API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#lora-adapters). For this sequential pipeline, explicit per-request selection is simpler than mutating global scales. Do not reuse a saved KV/prompt state across different adapter configurations unless the pinned server establishes its validity.

### Clean merge and quantization

The standard alternative is **original FP16/BF16 HF base + adapter → merged HF model → F16 GGUF → quantize once**. `merge_and_unload()` removes PEFT runtime adapter handling. [PEFT merge API](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#peft.LoraModel.merge_and_unload).

Illustrative CPU-only merge, in a separate environment compatible with the training checkpoint:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    QWEN3_SNAPSHOT, dtype=torch.float16, device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged = tuned.merge_and_unload(safe_merge=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
AutoTokenizer.from_pretrained(QWEN3_SNAPSHOT).save_pretrained(MERGED_DIR)
```

```sh
python convert_hf_to_gguf.py "$MERGED_DIR" \
  --outtype f16 --outfile "$MODELS/llm/alnadam-r3-f16.gguf"
llama-quantize "$MODELS/llm/alnadam-r3-f16.gguf" \
  "$MODELS/llm/alnadam-r3-Q4_K_M.gguf" Q4_K_M
llama-quantize "$MODELS/llm/alnadam-r3-f16.gguf" \
  "$MODELS/llm/alnadam-r3-Q5_K_M.gguf" Q5_K_M
# Optional fourth-bit variants, always from the same F16 input:
llama-quantize "$MODELS/llm/alnadam-r3-f16.gguf" \
  "$MODELS/llm/alnadam-r3-IQ4_NL.gguf" IQ4_NL
llama-quantize "$MODELS/llm/alnadam-r3-f16.gguf" \
  "$MODELS/llm/alnadam-r3-IQ4_XS.gguf" IQ4_XS
```

These are instructions for later, not executed steps. **Estimate:** an 8.2B 16-bit model needs roughly 16.4 GB of host weight storage; allow 32–48 GB free RAM for loading/merging copies and roughly 40–60 GB scratch disk beyond the original snapshot. The project's documented 62 GB host RAM should make CPU merging practical; it does not fit in the 3070's VRAM. CPU conversion/quantization takes no GPU reservation, but elapsed time depends on that host and was not measured.

Do not start this standard route by calling `merge_and_unload()` on the live NF4-loaded training object and assume its output is equivalent to merging the original base. Do not dequantize an existing Q4 GGUF, add the adapter and requantize it as the default export. Those preserve/add different quantization errors. The [quantizer's help](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp) explicitly warns about quality loss from requantization. `llama-export-lora` also exists, but if used for this route, supply an original high-precision base GGUF and quantize afterward; it does not recover precision missing from a low-bit base. [Export-LoRA tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/export-lora/README.md).

For a later imatrix comparison, collect activation statistics on **training-only** Arabic/English subtitle prompts and answers, including the brief and noisy ASR. Use the merged F16 model when calibrating the merged candidate. Never include E2 references or the independent final test set. `llama-imatrix` supports model/data inputs and `llama-quantize --imatrix` consumes the result; CPU/partial offload can avoid an F16 8B GPU OOM, at a substantial unmeasured time cost. [Imatrix tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md). IQ4_NL / IQ4_XS are distinct nonlinear quantizers, **not BnB NF4**; neither guarantees better subtitles than Q4_K_M. Test them only after the main paths work.

### Which route preserves quality?

Let `W` be original weights, `D_N(W)` the dequantized training NF4 base, `D_G(W)` a GGUF-quantized base, and `Δ=(alpha/r)BA` the learned update. Ignoring arithmetic rounding, the deployed layer weights differ as follows:

| Route | Effective weights | Change from the trained layer |
|---|---|---|
| HF training/eval reference | `D_N(W) + Δ` | Reference |
| Separate GGUF base + adapter | `D_G(W) + Δ` | `D_G(W) - D_N(W)` |
| Original base merge, then GGUF | `D_G(W + Δ)` | `D_G(W + Δ) - (D_N(W) + Δ)` |
| Diagnostic NF4-dequantized merge, then GGUF | `D_G(D_N(W) + Δ)` | An additional quantization of the trained effective weights |

**Inference from the algebra:** the separate route retains the floating adapter contribution but changes the base's error pattern; the clean merge changes the base and rounds the adapter into quantization bins. Neither error is uniformly smaller. The diagnostic fourth route can reveal dependence on NF4 but compounds quantization and is not the initial production recommendation. Matching the exact NF4 checkpoint in HF is the closest numerical reference. Even that requires matching tokenizer, template, decoding and kernels before interpreting changed text.

QLoRA established that NF4 adapter training can match 16-bit fine-tuning on its evaluated tasks. It did **not** measure this Qwen3 adapter transferred to GGUF or prove that NF4→FP16-merge→Q4 is lossless. [QLoRA paper](https://arxiv.org/abs/2305.14314). Recent [Scale-QLoRA, September 2026](https://arxiv.org/abs/2609.04526) studies merge damage and merge-aware training for **native microscaling FP4**; it reinforces that merge behavior matters but is not a Q4_K_M/NF4 result and is not a reason to add FP4 training tooling to an Ampere 3070.

I found **no controlled published measurement of this exact NF4-trained Qwen3-8B LoRA applied separately to Q4_K_M/Q5_K_M/IQ4 bases**, nor a Damascene translation benchmark settling separate versus merged. There is adjacent first-party project evidence: Slips reports a fine-tuned Qwen2.5-1.5B security summarizer on 47 incidents, comparing its BnB NF4 serving reference with merged GGUF variants:

| Variant | Mean judge score |
|---|---:|
| NF4 reference | 4.70 |
| Q8_0 | 4.59 |
| Q4_K_M | 4.28 |
| Q5_K_M | 4.14 |

Their [evaluation/deployment report](https://stratospherelinuxips.readthedocs.io/en/develop/immune/finetuning_quantization.html#performance-by-quantization) identifies the NF4 reference and gpt-oss-120b judge; the [training code](https://github.com/stratosphereips/Slips-tools/blob/main/unsloth-scripts/train_qwen.py) uses 4-bit loading and PEFT/Unsloth LoRA. This is small, task-specific, LLM-judged evidence with export/runtime confounds, **not an isolated quantizer experiment or separate-adapter benchmark**. It shows that a conversion regression is possible and that a small measured set need not rank Q5 above Q4. Its percentage judge-score losses cannot be converted to chrF loss here.

## 2. Eight-GB fit, throughput and alternative servers

### Budget this model rather than inheriting 16K

Official Qwen3-8B GGUF files are approximately **5.03 GB / 4.68 GiB Q4_K_M** and **5.85 GB / 5.45 GiB Q5_K_M**. Q8_0 is 8.71 GB, already too large before KV/cache overhead. These are model-file sizes, not measured VRAM peaks. [Official file listing/API](https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF?blobs=true).

**Computed** for Qwen3-8B's 36 layers, 8 KV heads and head dimension 128: ordinary F16 K+V storage is `36*8*128*2*2 = 147456 bytes/token` (144 KiB). This is different from assuming the current Qwen3.5 model's cache geometry. [Base configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json).

| Total allocated context, one slot | F16 KV | Q4 weights + F16 adapter + KV | Q5 weights + F16 adapter + KV |
|---|---:|---:|---:|
| 4,096 | 0.56 GiB | 5.33 GiB | 6.09 GiB |
| 8,192 | 1.13 GiB | 5.89 GiB | 6.66 GiB |
| 16,384 | 2.25 GiB | 7.01 GiB | 7.78 GiB |

CUDA context, graph/work buffers, allocator slack, display usage and server implementation details are **additional**. At 16K the Q5 sum alone nearly exhausts 8 GiB, and Q4 is fragile. At 8K Q4 should have workable headroom; Q5 must be measured on the actual desktop. These are estimates, not fit guarantees. Keep `--llm-ctx`; explicitly use `-np 1` to avoid surprises from parallel defaults. If 4K covers the tokenized prompt plus up to 960 output tokens and safety margin, use it. Q8 KV is a later memory knob, not part of the first quality-preservation comparison.

`--cpu-moe` already has a role for the existing expert model; **dense Qwen3-8B has no MoE experts to offload**. It is not a way to make its weights fit. Keep full GPU layers for this profile; falling back to CPU layers changes speed substantially and must be recorded.

### Alternatives

The following support claims were checked against current primary sources. No verified apples-to-apples 3070 throughput table exists for all these servers with **this adapter and workload**.

| Server | Applicable LoRA/quantization path | Decision for this project |
|---|---|---|
| **llama.cpp** | GGUF base + GGUF LoRA; per-request scales; existing grammar and JSON client. | Preferred. Preserve the lease/GPU queue and test exported quality before changing runtime. |
| **vLLM** | Qwen3ForCausalLM has LoRA support. `--enable-lora --max-lora-rank 16 --lora-modules alnadam=/path/lora`; API selects the registered adapter via `model: "alnadam"`. Dynamic loading APIs also exist. BnB NF4 can avoid the GGUF base change; current BnB support is an out-of-tree plugin. AWQ/GPTQ changes the base error pattern again. | Worth a bounded fallback trial **only if GGUF loses meaningful quality** and BnB retains it. Single-sequence dependent windows underuse the scheduler. Limit context/sequences and graph memory; no assumption an 8B BF16 example fits. |
| **ExLlamaV2 + TabbyAPI** | EXL2/GPTQ low-bit base and PEFT LoRA; dynamic adapter selection exists. | V2 is currently archived in favor of V3. Do not introduce a new archived backend for this deployment. |
| **ExLlamaV3 + TabbyAPI** | V3 exists; latest release checked is **v1.6.6, 2026-10-31**. Supports Qwen3 and EXL3. That release includes a PEFT LoRA loader applying float updates to quantized linears. | Technically interesting, but another quantization transfer and deployment stack. Library loading is verified; TabbyAPI's exact per-request V3 LoRA/grammar integration for this adapter is **unverified**. No demonstrated speed/quality win here. |
| **SGLang** | `--enable-lora --lora-paths alnadam=/path/lora`; OpenAI API uses `model:adapter-name`, native API `lora_path`. Supports adapter pools and structured output. Low-bit quantizers exist. | A quantizer being supported and LoRA being supported separately is not proof every combination works. Exact Qwen3/BnB double-quant or AWQ+LoRA on 8 GB is **unverified**. Too much integration cost without a measured gain. |
| **Ollama ADAPTER** | Older instructions show `FROM <matching-base>` plus `ADAPTER <adapter>`. Current source deprecates LoRA adapters and disallows creating new ones; old adapter layers can be carried over. Existing merged GGUF import remains. | Do not choose the historical ADAPTER route. Importing a merged GGUF is possible but adds model packaging without solving NF4 mismatch or the existing client/grammar work. |

Evidence: [vLLM Qwen3 model support](https://docs.vllm.ai/en/latest/models/supported_models/), [vLLM LoRA API](https://docs.vllm.ai/en/latest/features/lora/), [vLLM BnB plugin requirement](https://docs.vllm.ai/en/latest/features/quantization/bnb/); [ExLlamaV2 archive notice](https://github.com/turboderp-org/exllamav2), [V2 author's hot-swap explanation](https://github.com/turboderp-org/exllamav2/discussions/65), [V3 release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.6.6), [V3 architecture support](https://github.com/turboderp-org/exllamav3), [V3 LoRA implementation at that release](https://github.com/turboderp-org/exllamav3/blob/v1.6.6/exllamav3/model/lora.py); [SGLang LoRA serving](https://docs.sglang.io/docs/advanced_features/lora), [SGLang quantization arguments](https://docs.sglang.io/docs/advanced_features/server_arguments); [Ollama deprecation in source](https://github.com/ollama/ollama/blob/main/server/model.go), [September 15 removal PR](https://github.com/ollama/ollama/pull/14969), [current GGUF import documentation](https://docs.ollama.com/import).

For a future vLLM trial, an **unvalidated starting configuration** is `--quantization bitsandbytes --enable-lora --max-lora-rank 16 --lora-modules alnadam=/path/lora --max-model-len 4096 --max-num-seqs 1 --enforce-eager`, with an explicit memory-utilization limit that leaves desktop headroom. Confirm the installed BnB plugin's NF4/double-quant parameters against training, preferably by loading a saved quantization configuration. Generic `--quantization bitsandbytes` is not proof those settings match. Configure the real model/adapter name in the client: existing `chat_json()` hardcodes `model: "local"`. This is a diagnostic candidate, not a tested launch recipe.

### Throughput evidence and a realistic target

- **Project measurements:** DESIGN records current Qwen3.5-9B Q4 translation at 63 output tok/s in a short test, and about 20 minutes for a complete episode with helper passes. HF generation for this experiment takes about 18 minutes for 434 lines. These have different denominators and should not be compared as raw kernel benchmarks. [Existing hardware measurements](../DESIGN.md).
- **External first-party measurement:** a September 2026 runner benchmark reports **81.6 decode tok/s and 2,224.5 prefill tok/s** for llama.cpp b10754, Qwen2.5-7B Q4_K_M on an RTX 3070 8 GB / Windows 11. That is a useful nearby anchor, not Qwen3-8B with LoRA or Arabic grammar. [Benchmark author's table](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/docs/benchmarks.md).
- **Planning estimate:** allow roughly **40–80 decode tok/s** for an all-GPU 8B Q4 experiment, with separate LoRA, prompt length and grammar determining where it lands. At 30,000 generated tokens that alone is 6.25–12.5 minutes; prefill, ~55 eight-line calls for 434 cues, helpers, retries and startup add time. Do not promise an episode time from `llama-bench tg128` alone.
- There is no trustworthy current **3070 + Qwen3-8B + this LoRA** vLLM/EXL3/SGLang speed figure in the sources checked. EXL3 is not automatically faster on Ampere: its author explains the more expensive unpacking and difficulty saturating Ampere bandwidth in a [2026 performance issue](https://github.com/turboderp-org/exllamav3/issues/144). Do not extrapolate H100 concurrency benchmarks to one rolling-context episode.

Keep episodes serialized on the existing GPU lane; release the server at stage completion for ASR and games. Across episodes there is potential concurrency, but 8 GB and one user's queue favor predictable memory and release behavior over maximizing server occupancy. Merging's removal of adapter matmuls is a plausible speed benefit, **not a measured speedup**.

## 3. Output structure and helper passes

### Keep the trained lines first

Constrain structure in llama-server and parse both languages strictly. This preserves the successful target sequence and its Arabic-before-English order. Line format avoids JSON escaping and auxiliary summary/uncertainty tokens, though its exact token savings must be measured. A grammar constrains **syntax, not meaning**; it cannot establish that the Arabic correction is justified, the translation is faithful or an empty line represents actual noise.

Generate a grammar for the actual `n`, using literal local IDs instead of a permissive integer rule. For a full eight-cue request:

```gbnf
root ::= pair-one "\n" pair-two "\n" pair-three "\n" pair-four "\n" pair-five "\n" pair-six "\n" pair-seven "\n" pair-eight
pair-one   ::= "1. ar: " text "\n1. en: " text
pair-two   ::= "2. ar: " text "\n2. en: " text
pair-three ::= "3. ar: " text "\n3. en: " text
pair-four  ::= "4. ar: " text "\n4. en: " text
pair-five  ::= "5. ar: " text "\n5. en: " text
pair-six   ::= "6. ar: " text "\n6. en: " text
pair-seven ::= "7. ar: " text "\n7. en: " text
pair-eight ::= "8. ar: " text "\n8. en: " text
text ::= [^\u0000-\u001f\u007f]{0,300}
```

This admits Unicode Arabic and empty tracks, excludes embedded newlines/control characters, fixes IDs/order/pair count, and caps each field at 300 code points. For a short final batch or split retry emit only its `n` pair rules, restarting at 1. Two-speaker cues remain one physical field, e.g. `- … - …`; rendering can wrap afterward. Flatten any multiline training targets the same way for the next run, because the present `answer()` does not enforce this restriction.

Use `grammar` on each `/v1/chat/completions` request, **without** `response_format`/`json_schema`; leave verify/shorten on their own JSON constraints. A server-global line grammar would conflict with those helper tasks. llama.cpp documents Unicode classes, bounded repetitions, request grammars and a validator in its [GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md). The bounded repetition avoids long chains of optionals that the guide warns can slow sampling.

Draft translation request, omitting the long system/user text:

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "<ft_llm.SYSTEM + exact brief>"},
    {"role": "user", "content": "<ft_llm.user_prompt; 3 context pairs, up to 8 raw ASR cues>"}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "lora": [{"id": 0, "scale": 1.0}],
  "temperature": 0,
  "repeat_penalty": 1.05,
  "dry_multiplier": 0,
  "max_tokens": 960,
  "grammar": "<generated GBNF above>"
}
```

This approximates the HF evaluation sampler; llama.cpp repetition-window semantics and numerical kernels can still differ. Inspect the actual rendered chat template (including the disabled-thinking assistant prefix), not merely the visible messages. Qwen's [model card](https://huggingface.co/Qwen/Qwen3-8B) supports `enable_thinking=False` and recommends sampled non-thinking defaults; use greedy first to reproduce **this project's measured evaluator**, then test sampling separately. Do not mistake the model card's warning about greedy **thinking mode** for evidence that the current non-thinking result is invalid.

**Operational limits:** a grammar cannot guarantee completion before `max_tokens`, timeout or context exhaustion. Check `finish_reason`, reject truncation, require exactly `2*n` fields, unique IDs, valid UTF-8 and no trailing content, then map local ID `j` to global cue `range.start+j`. Reject duplicate IDs rather than taking the first as HF `parse()` does. Keep retry/split/fallback and report its rate. Empty English for an input believed to contain speech must be reviewed/flagged; grammar acceptance alone must not hide it. Unit examples should cover Arabic, quotes/backslashes, two speakers, empty noise, short batches, truncation and duplicates.

### JSON retraining is feasible, but is a separate experiment

There is no need to regenerate linguistic labels merely to change serialization: transform the same `fix/ar/en` pairs into JSON. Keep corrected Arabic before English. The current adapter has not learned global IDs, JSON input, duration budgets, `uncertain`, or `scene_summary`; forcing it through today's schema is a useful ablation, not a quality-preserving deployment assumption.

For a next training run aimed at production JSON, use this **actual current contract**, rather than the larger historical DESIGN example. This concrete example is for `n=8` and global IDs 101–108; generate lengths and enum from the batch:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["lines", "scene_summary"],
  "properties": {
    "lines": {
      "type": "array", "minItems": 8, "maxItems": 8,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "ar", "en", "uncertain"],
        "properties": {
          "id": {"type": "integer", "enum": [101,102,103,104,105,106,107,108]},
          "ar": {"type": "string", "maxLength": 300},
          "en": {"type": "string", "maxLength": 300},
          "uncertain": {"type": "boolean"}
        }
      }
    },
    "scene_summary": {"type": "string", "maxLength": 400}
  }
}
```

Length plus an ID enum still permits duplicates/reordering. Validate exact IDs, count and order in Rust; a generated literal-ID JSON grammar can enforce these too. Avoid relying on positional JSON-schema features until verified against the pinned llama.cpp subset. For training labels, set `scene_summary: ""` and keep it unused until trustworthy summary targets exist. Current skipped garbled lines do not provide honest uncertainty supervision: do not label every retained line `false` and claim calibrated confidence. Keep host-derived review flags; add explicit labeled unrecoverable examples if teaching abstention. Existing runtime JSON requires the Boolean, but it should remain a heuristic unless independently evaluated.

**Inference:** JSON may cost tokens and training memory; line format may need more custom parsing and cannot emit calibrated uncertainty/summary as currently trained. Neither format is inherently more accurate in Arabic translation. The important measured win came with joint corrected-Arabic/English supervision and more data, not evidence favoring JSON or line syntax alone.

### Verify and shorten

For the first export-fidelity test, save raw first-pass translations with both helpers disabled. Then test the product pipeline with helpers separately. A line-format translation does not require line-format helpers: normalize it to `Translated` and reuse helper JSON. With separate adapters, send `lora: []` for verify/shorten so their untrained task/schema does not activate the translation adapter. This uses **base Qwen3-8B**, not Qwen3.5-9B; helper quality needs its own comparison. Retain fail-open versus fail-closed behavior explicitly in metrics; never count a verifier error as proof of no slips.

In particular, the current verifier can reject a good repair because it only sees raw ASR. A follow-up prompt should include raw ASR **and** corrected Arabic and English, judging faithfulness without requiring a translation of the recognizer's mistake. Do not make this prompt change in the first quantization comparison. For shorten, test both ≤84 characters/two lines and **duration-derived CPS**, keep before/after text, and count semantic damage. A short-duration subtitle can violate CPS while fitting two lines, and therefore never trigger today's shorten pass. These recommendations follow the actual [verify/shorten implementations](../../crates/server/src/translate.rs) and [two-line renderer](../../crates/subs/src/segment.rs).

For a merged model the adapter cannot be disabled; verify/shorten must work on that fine-tune, be trained as auxiliary tasks, or run on a separately loaded base. Loading both 8B models concurrently is outside this VRAM budget. Sequentially reloading Qwen3.5 for helpers is possible but adds startup/GPU occupancy; do not make it the initial solution. This is a concrete operational advantage of the separate-adapter experiment.

## 4. Make training and serving agree

| Mismatch | Change for this adapter now | Training feedback after deployment is stable |
|---|---|---|
| 8-cue training vs 10–25 production | Dedicated profile uses fixed stride-8 windows, up to eight cues, no output overlap. Reproduce HF boundaries first. | If scene-aware 4–8 cue windows help, train/evaluate that distribution. Do not assume 25-cue training fits: the existing run already peaks at 7.2 GB. |
| 3 previous pairs vs 8 | Supply three previous **raw-ASR + model English** pairs; do not replace context Arabic with corrected output. | Mix reference and generated previous-English context, cached offline, to reduce exposure mismatch. Label generated-context runs separately. |
| System brief vs user `show` | Copy the training system text and identical brief into the system message; plain numbered user prompt, not JSON. No rolling scene summary. | One versioned prompt renderer/fixture contract across Python and Rust, including brief/glossary order and whitespace. Train the eventual deployed prompt. |
| Static brief vs dashboard learning context | Freeze a brief snapshot for the controlled A/B. Dashboard normally adds glossary and up to 12 examples; that is another variable. | Include realistic bounded glossary/verified-example sections during training or separately prove inference-only additions help. Keep evaluation episode examples out. |
| Cue numbering | Reset local numbers 1…n per request, then map back to global cue IDs. | If moving to JSON/global IDs, train those exact IDs and inputs. |
| Thinking | Already off on both paths; preserve `enable_thinking=False` / `chat_template_kwargs.enable_thinking=false`, `--jinja`, correct Qwen3 template and stop token. | Save tokenizer/template revision, rendered prompt samples and answer-loss token boundary assertions. |
| Sampler | First reproduce greedy + repetition penalty 1.05, no DRY; keep current production sampler as its own baseline. | Select/checkpoint on the deployed sampler, grammar and GGUF, not only HF chrF. |
| Arabic orthography | Training says keep dialect; production asks e.g. كتير→كثير and عالبيت→على البيت. Preserve trained behavior in the adapter profile and review Arabic. | Pick one spelling policy, normalize targets consistently, then retrain; do not silently push dialect correction toward MSA at inference. |
| Durations / max_chars | Trained inputs contain no reading budgets. Start with current short-style instruction, report CPS and use measured shortening. | Add durations/budgets only with sensible condensed targets; a budget field beside an over-budget target teaches a contradiction. |
| Unrecoverable/noise/repetition | Keep existing filters and conservative host flags; score on these cases, not only the 165 recoverable references. | Curate explicit uncertain/unrecoverable/noise and ASR-loop examples. Preserve noisy raw ASR as input, corrected Arabic + meaning as output; never manufacture recoverable dialogue for corrupt input. |
| Episode boundaries | Ensure three-line context and stride windows do not cross episodes. Current `windows()` has no episode-ID logic. | Store episode IDs or build windows per episode; concatenating the flat input format otherwise creates artificial cross-episode context. Split train/dev/test by episode before windowing. |

The code observations above are directly supported by [Python prompt/window code](../../worker/shu2al_worker/ft_llm.py), [Rust prompts and batching](../../crates/server/src/translate.rs), and [queue brief construction](../../crates/server/src/queue.rs).

**Training priority:** retain the successful 8-GB NF4 recipe while validating the exported artifact. Nothing in the research requires retraining directly on GGUF, and this script/PEFT path does not provide GGUF training. If a clean merge wins, make merge+quantize+GGUF dev evaluation part of checkpoint selection; if separate adapters win, evaluate that exact base quant and scale. Only a substantial, repeatable export gap justifies changing the training quantizer/runtime. LoftQ initializes adapters to compensate quantization error; that is useful in its supported quantizer setup, not an automatic fix for a different GGUF base. [PEFT quantization/LoftQ guidance](https://huggingface.co/docs/peft/developer_guides/quantization).

Do not increase rank, sequence length and epochs together in response to the plateau. First make export quality measurable, then spend annotation effort on idioms/names/unrecoverable cases and independent human-verified editor examples. Preserve one epoch explicitly, deterministic seeds and split manifests. The current stride-4 training overlap means the number of training windows is not the number of independent examples; the 2,000 Levanti pairs also have a different clean-input distribution. Record both rather than treating a lower training loss as production progress.

## 5. Concrete integration and A/B plan

### Smallest implementation surface

Use existing `Llm::Docker` / `Llm::External`; **a fourth serving-backend variant is unnecessary**. Add a translation profile independent of model transport:

```rust
enum TranslationProfile {
    ProductionJson,    // existing behavior and default
    Qwen3LoraLines,     // training prompt, 8/3 context, strict line grammar
}
// Options gains translation_profile; Settings receives it.
// CLI proposal: --translation-profile production-json|qwen3-lora-lines
```

The profile selects the prompt/format/window contract. Adapter artifacts remain llama-server startup flags through the **existing** repeatable `--llm-arg`; `--llm-ctx` remains the context control. Add a small ablation switch for helper policy (`off` versus `base`) for evaluation; default the promoted adapter profile to the policy that wins the helper comparison. No new CPU-offload flag is needed.

| File | Proposed change and reason |
|---|---|
| `crates/server/src/llm.rs` | Factor HTTP chat response/content/usage handling into a small internal helper; support per-request grammar, LoRA selection and sampler for line translation. Keep `chat_json` for production/helpers. Check finish reason and server errors. Send explicit adapter-off on adapter-profile JSON helpers; leave unrelated external providers untouched. Inspect loaded adapter IDs once at startup and reject a missing adapter. |
| `crates/server/src/translate.rs` | Add the profile's exact Python-compatible prompt builder, generated n-pair grammar, strict Arabic+English parser, local→global mapping, fixed windows/context. Convert results to existing `Translated`. Keep CER flags and retry/split/fallback. Make verifier failure visible and retain raw/pre-helper output for evaluation. Helpers keep JSON and use base-only selection. |
| `crates/server/src/pipeline.rs` | Thread profile/helper policy through `Options` → `Settings`; keep Docker lease and existing output/rendering contract. Record model/adapter hashes, image digest, context, profile, sampler, prompt/brief hash and helper policy in run metadata. Existing files otherwise silently survive model changes. |
| `crates/server/src/main.rs` | Parse profile/helper ablation flags, pass them into `Options`, reject nonsensical seq2seq/profile combinations. Continue using `--llm-model`, `--llm-image`, `--llm-ctx`, `--llm-arg` and `--llm-url`; no replacement backend framework. |
| `worker/shu2al_worker/ft_llm.py` | Later: save manifest/tokenizer/template identity; add export-evaluation ingestion or a small offline evaluation helper, strict dual-track parsing and episode-safe windows. Share golden prompt fixtures with Rust rather than executing Python for every request. Change training target format only in the JSON experiment. |
| Tests near existing translate/LLM tests | Cover prompt equivalence, exact n/IDs, local/global mapping, Unicode fields, malformed/truncated output, adapter selection for helpers, short batches and retry fallback. A small fake HTTP endpoint can check request fields without a GPU. |
| Dashboard promotion: `queue.rs`, `api.rs`, `web/src/lib/api.ts`, `web/src/NewRequest.svelte`, `web/src/QueueView.svelte` | Add opt-in translator key such as `qwen3ft`, map it to Qwen3-8B plus adapter/profile at 8K, admit it in `check_translator`, expose the choice and label. CLI options alone do not reach the queue. Existing `episodes.translator` is text, so this needs no schema migration or change to the `qwen9b` default. |
| `docs/DESIGN.md` | After a successful implementation, record selected artifact/profile and reconcile the stale §5.5 narrative. This research task edits only this report. |

For CLI testing, the existing extra arguments can already carry `--lora` and `--lora-init-without-apply`; the **new** profile flag below would supply the prompt/parser/request selection. Illustrative future command:

```sh
shu2al run "$EPISODE_MEDIA" \
  --work "$AB_ROOT/e2-qwen3ft" --name AlNadam-E02 --from translate \
  --models "$MODELS" --brief "$FROZEN_BRIEF" \
  --llm-image "$PINNED_LLAMA_IMAGE" \
  --llm-model llm/Qwen3-8B-Q4_K_M.gguf --llm-ctx 8192 \
  --llm-arg=--parallel --llm-arg=1 \
  --llm-arg=--lora --llm-arg=/models/llm/alnadam-r3-lora-f16.gguf \
  --llm-arg=--lora-init-without-apply \
  --translation-profile qwen3-lora-lines
```

If `--llm-url` is used, start/manage the adapter server externally and confirm its capabilities; that path does not own or release GPU memory through `Lease::stop()`. It is useful for experiments but the Docker lease is the production fit for the gaming/shared-GPU constraint.

### A/B without changing the episode under test

1. **Freeze inputs before branching:** original media identity, ASR transcript, aligned transcript, `cues.ar.srt`, reference mapping, brief/glossary snapshot and train/test episode membership. Create separate work directories outside the live episode. For `run --from translate`, copy or link **all four earlier stage outputs** (`audio.16k.flac`, `asr/transcript.json`, `asr/transcript.aligned.json`, `cues.ar.srt`). The runner executes missing earlier stages even with `--from translate`; copying cues alone can accidentally rerun ASR. Alternatively an offline harness can call `run_stage(Stage::Translate)` then Render directly.
2. **Measure export fidelity:** saved HF NF4+LoRA predictions versus separate GGUF+LoRA under the same eight-cue/three-context prompt, greedy decoding, no helpers, first without then with grammar. Run adapter-off Qwen3-8B with the same prompt too. Score 165 E2 references using the same `sacrebleu.corpus_chrf(..., word_order=2)` and exact stable cue mapping, but retain all 434 generated lines and Arabic. Never align outputs to references by list position if skipped/unscored lines have changed indices.
3. **Measure product improvement:** current Qwen3.5-9B Q4 production profile, with today's verify/shorten, versus tuned Qwen3-8B with base helpers. Use identical episode cues and brief; retain first pass and final output for each. This is the decision comparison, although each profile deliberately has its own trained prompt. A helper-off run and logged before/after changes separate translation improvements from postprocessing damage.
4. **Measure quantization choice:** at identical prompts/boundaries compare separate Q4+LoRA, merged Q4 and merged Q5; optionally separate Q5+LoRA to isolate base precision from merge. Use Q5 at 4K if needed, but rerun the paired Q4 at the same context and verify no truncation. Record differing VRAM/context as part of the result, not a hidden confound.
5. **Validate outside the optimization set:** reserve an entire untrained episode or human-verified scene collection with independent references. E2 has already informed several rounds, so it is now a development set. Blind-review faithfulness, invented content, names, the three cited idioms, unrecoverable ASR, Arabic correction and subtitle reading quality. Report paired differences and scene-level/bootstrap uncertainty; do not infer significance from +0.2 on 165 correlated lines.

Record completion/prompt tokens separately, end-to-end translate time including startup/helpers, server decode/prefill timings, peak GPU memory, exact context, retries/splits, incomplete/empty output, overcorrection flags, helper errors, helper changes, and SRT CPS/CPL/line-count violations. Existing `pipeline.rs`'s aggregate `completion_tokens / elapsed_s` includes waits/helpers and is not pure decoding speed. Proposed initial gates: exact cue coverage/order on every accepted result; no unexplained >1 chrF++ export drop versus HF on E2; sustained meaningful improvement over 39.9 plus independent human preference; no increase in serious hallucinations; and enough measured VRAM headroom to coexist with the normal desktop. The one-point export threshold is a triage rule, not a statistical claim.

### Rollback

Keep current Qwen3.5-9B GGUF, tested image digest and `ProductionJson` untouched, and add a separate `qwen3ft` key/artifact path. Roll back by selecting `qwen9b` / `production-json`, omitting adapter flags, and rerunning translate+render from **preserved original cues** into a fresh work directory. Disabling LoRA alone produces base Qwen3-8B and is **not** rollback to Qwen3.5-9B. Never overwrite live editor corrections or promote raw A/B outputs into the library automatically. Retain each candidate's artifact hashes and results so a new adapter can be rolled back independently of the server binary.

## Recommended next experiments

Ranked by information gained per GPU hour; expected effects are hypotheses unless identified as existing measurements.

1. **Export the existing round 3 adapter; reproduce its exact prompt in llama.cpp.** Change: Qwen3-8B Q4_K_M + F16 adapter, scale 1, eight-cue/three-context windows, frozen system brief, thinking off, 8K/one slot, greedy/repetition 1.05, helpers off; compare unconstrained and the generated GBNF. **Expected effect:** retain most of the 45.7 while establishing strict cue structure and faster inference. **3070 cost:** conversion uses CPU; two E2 inference passes, approximately 10–25 minutes total as a planning allowance, Q4 weight/adapter/KV sum 5.89 GiB plus runtime. **Worked if:** accepted outputs have exact coverage, no truncation/loops, Arabic survives parsing, and chrF++ is within the proposed one-point export tolerance with reviewed differences. If it fails, check template/scaling before training again.

2. **Compare clean merges at Q4_K_M and Q5_K_M against experiment 1.** Change: CPU merge original FP16 base + adapter, quantize each once; keep prompts/sampler/grammar fixed. Add separate Q5+LoRA only if the Q4 merge loses. **Expected effect:** determine whether preserving float LoRA or rounding the merged update works better; possibly reduce latency. **3070 cost:** two or three E2 passes, roughly 10–35 minutes; Q5 at 8K starts from a 6.66 GiB weight/adapter/KV sum if separate, slightly lower merged, so 4K may be necessary. CPU merge requires tens of GB RAM/disk, no training VRAM. **Worked if:** a candidate retains meaning/chrF and improves measured time or headroom; choose on episode results, not bit-width alone.

3. **Run the current production path and the best tuned candidate on one identical full episode, then audit helper effects.** Change: isolated directories with identical cached upstream files; test tuned first pass versus base-only verify/shorten, preserving all rewrites. Add helper-error and CPS instrumentation before judging it. **Expected effect:** establish actual subtitle benefit and identify helpers undoing legitimate ASR repairs. **3070 cost:** approximately 30–60 minutes for baseline plus candidate/helper ablation, based on current ~20-minute production episodes; inference only, one model resident. **Worked if:** blind review prefers the tuned result, hallucinations/slips do not rise, CPS and line-fit failures are visible, and full-stage time/VRAM are acceptable. Repeat final selection on an independent held-out episode or human reference set.

4. **Train one controlled JSON-output counterpart only after line deployment is measured.** Change: start a fresh adapter from the same base/data, explicit `--epochs 1`, keep r/lr/split/8-cue/3-context fixed; serialize corrected Arabic then English in the drafted production schema and use the identical JSON prompt at serving. Supply empty summary targets until summaries are labeled; do not claim uncertainty calibration. **Expected effect:** remove custom translation syntax and test whether JSON costs quality; it need not improve idioms. **3070 cost:** one training run plus exported evaluation; existing round 2 training was 29 minutes, so budget roughly 30–90 minutes for the larger JSON experiment, with **OOM risk above the existing 7.2 GB peak**. Retain 64-token answer-loss slicing and reduce window/token length if required, documenting that confound. **Worked if:** independent quality and Arabic repair match/exceed the line profile, helpers remain useful, and measured token/time overhead is justified.

5. **Choose the fifth experiment from the measured failure, not a larger training sweep.** If a >1-point GGUF loss remains after template/scaling checks, change only serving to a bounded **vLLM BnB+LoRA** trial, explicitly matching NF4/double quant and 4K/one sequence; expected effect is to isolate quantizer transfer, at the cost of one setup session and ~one E2 run, with unverified fit/speed and no training. It works if it recovers HF quality with useful throughput/headroom. **Otherwise**, spend that slot on training-only, human-verified Damascene idiom/noise examples from `cue_edits`, preserving raw-ASR→corrected-Arabic→English and the chosen deployment prompt; one fresh one-epoch adapter costs roughly another 30–90-minute training/eval reservation at the current ~7.2 GB baseline. It works if a new independent idiom/noise set improves without increased invented content; another +0.2 on E2 is insufficient.
