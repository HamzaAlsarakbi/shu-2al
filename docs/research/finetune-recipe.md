# Fine-tuning Syrian drama translation within 8 GB

Research date: **2026-11-02**. Scope: training recipe, training data and base-model selection for Damascene Arabic ASR → corrected Arabic + English subtitles. Research only; no training, remote access, installations or production changes were performed.

**Recommendation:** keep Qwen3-8B as the controlled training baseline; test a lower learning rate, warmup/decay and accumulation before changing adapter algorithms. Spend the next annotation effort on recoverable ASR errors and idioms, and improve evaluation before interpreting 45.5 → 45.7 as a capacity ceiling. For alternatives, **ALLaM-7B and Command R7B Arabic deserve an inference bake-off**, with Jais-2-8B next if its training integration is practical. Qwen3.5-9B is already a useful production baseline, but its 4-bit training path is materially less attractive than its inference path.

Evidence labels used below:

- **Local:** inspected repository code or the task's supplied results; not independently rerun.
- **Published:** a paper's experiment, with its language, direction and scale identified.
- **Vendor/practitioner:** documentation or a first-person report; not an independent replication.
- **Proposal/estimate:** my recommendation or arithmetic for this project, not a measured gain. All proposed settings, budgets and acceptance thresholds have this status unless explicitly attributed.

## 1. What the current experiment actually measures

Read: [training implementation](../../worker/shu2al_worker/ft_llm.py), [design history](../DESIGN.md#annotate-then-fine-tune-round-1-2026-11-01-one-episode-tested-on-the-next), [earlier translation research](translation.md), [ASR research](asr-models.md), [production translation](../../crates/server/src/translate.rs) and [llama-server integration](../../crates/server/src/llm.rs).

| Observation from this checkout | Consequence for the plateau |
|---|---|
| Eight-line windows, stride four, three previous lines. | Most show lines are **supervised twice per epoch**. An epoch is not one exposure per unique subtitle. |
| Approximately `ceil(3671/4) + ceil(2000/8) = 1168` examples, before preserving episode boundaries. | Round 3 is roughly 918 show windows and 250 Levanti windows. Effective batch one means roughly 1,168 optimizer updates per epoch. |
| Round 2 was approximately 744 windows; round 3 uses the same LR per update. | Increasing data also increases the number of updates by about 57%. The experiment did not isolate data quantity from optimization dose. |
| Training previous-English context is reference English; inference context is the model's previous output. | There is a teacher-forcing mismatch across windows, beyond ordinary teacher forcing inside an answer. |
| `windows()` takes one flat list; neither training nor inference resets on episode IDs. | If the export concatenates episodes, some windows/context cross episodes. **Conditional finding:** inspect export boundaries before drawing conclusions; the supplied JSON was not available here. Skipped-cue gaps can also create misleading adjacency. |
| `corrected()` falls back to raw ASR when `fix` is absent. | The 2,813 lines without explicit corrections are not necessarily verified-clean Arabic. Treat them as copy targets of uncertain quality, not gold transcripts. |
| Answer loss is mean CE per example, including Arabic, English, numbering and EOS. | Arabic copy tokens can dominate some examples. Neither English loss nor Arabic correction quality is reported separately. |
| Torch shuffling and adapter initialization have no fixed seed. Only Levanti subsampling has seed 13. | Round-to-round variation includes uncontrolled training randomness. |
| CLI defaults to **three epochs**; documented runs used **one**. | Save the actual command/config; reproducing the default is not reproducing round 3. |
| AdamW has no explicit weight decay argument. | PyTorch's default is already 0.01; “add weight decay 0.01” would not be a new intervention. See [AdamW API](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html). |
| HF evaluation uses eight cues, text output, greedy decoding and repetition penalty 1.05. Production uses scene batches up to 25, JSON, temperature 0.3, DRY, a summary, verification and shortening. | **45.7 vs 39.9 is a system comparison, not an isolated base-model or adapter comparison.** Test the same first-pass format and decoder before attributing the difference. |

Round 2's roughly eight-point improvement over its own base is encouraging local evidence. The additional 0.2 in round 3 has **no supplied confidence interval or seed replication**. Neither “significant improvement” nor “statistically proven plateau” follows from those two numbers. The same annotator wrote both targets and references, so subtitle style matching can improve chrF++ without an equivalent improvement in faithfulness.

Before choosing an expensive training strategy, run an **oracle-source diagnostic**: translate the same held-out lines with raw ASR versus independently corrected Arabic, keeping context, format and decoder fixed. Report recoverable-noise and unrecoverable-noise subsets separately. This estimates the headroom that better source recovery could offer; it is not a deployable result. Do not feed a model's own corrections back as purported gold.

## 2. Training settings: evidence and a concrete starting recipe

### 2.1 What MT work and practitioners actually use

There is no established recipe specifically validated for 1,168 overlapping, noisy Syrian-drama windows on Qwen3-8B. The most relevant sources span substantially different data and hardware scales.

| Source | Actual setup / settings | What transfers here |
|---|---|---|
| **Arabic Dialect Translation with Small LLMs**, AbjadNLP 2026, Appendix A.1 ([paper](https://aclanthology.org/2026.abjadnlp-1.11.pdf)) | Qwen2.5; all seven attention/MLP projections; **r16, α32, dropout .05, LR 5e-5, effective batch 8, two epochs**, length 512, checkpointing. Paper reports two A100 80 GB GPUs for SFT. MADAR has 52k aligned dialect/MSA pairs, not 1.2k drama windows. | Closest dialect-MT support for our existing adapter shape and a lower LR / accumulated batch. Their sentence-length limit cannot accommodate our brief. |
| **ALMA** official LoRA SFT script ([script](https://github.com/fe1ixxu/ALMA/blob/master/runs/parallel_ft_lora.sh)) | r16; LR **2e-3**, inverse-square-root schedule, 1% warmup, one epoch, prompt loss masked; microbatch 4 × accumulation 4 × GPU count. | Answer-only MT loss has precedent. Its high LR is not a small-data QLoRA recommendation. |
| **ALMA-R / CPO** official script ([script](https://github.com/fe1ixxu/ALMA/blob/master/runs/cpo_ft.sh)) | LR **1e-4**, β .1, inverse-square-root schedule, 1% warmup, one epoch, microbatch 2 per GPU; 512 total tokens. | Preference tuning need not require online RL. Different objective and short examples; do not transfer LR blindly. |
| **SwiLTra-Bench**, ACL 2025 ([paper](https://aclanthology.org/2025.acl-long.725.pdf)) | Legal-domain adaptation: LR **1e-4**, linear schedule, 1,000 warmup steps, batch 128, weight decay .01, early stopping; lowest validation loss usually at one epoch. Explores ranks 16/128 and LR 1e-5–1e-3. | Validate stopping time; its absolute warmup and large batch would leave our tiny run barely trained. |
| **Data-Efficient Domain Adaptation using CPO**, Oct 2025 preprint, Appendix A ([paper](https://arxiv.org/html/2510.27556v1)) | Llama3-8B-Instruct NF4; 14.7k pairs; q/v only, r64/α16/dropout .1; LR **1e-3**, cosine, 200 warmup steps, batch 4 × accumulation 8, one epoch, A100 80 GB. No validation split in the reported configuration. | A post-editing data construction precedent, not evidence for its aggressive LR on Syrian drama. |
| **CUNI AlexandriaX 2026** ([paper](https://aclanthology.org/2026.arabicnlp-sharedtasks.101/)) | Conversation-level Gemma-4-31B QLoRA, metadata/speakers, **English → 13 Arabic varieties**. Reports dev chrF 45.2 → 50.4, then 51.3 with candidate reranking. | Supports conversation context and supervised adaptation; reverse direction and much larger model. Full hyperparameters were not independently extracted here. |
| **Unsloth practitioner guide** ([guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)) | Starting guidance: LR 2e-4, 1–3 epochs, rank 16/32, α r or 2r, all seven projections, 5–10% warmup, linear/cosine, effective batch around 16. | A search range, not an MT result. Our proposed batch 8 and LR 5e-5 are deliberately conservative. |

The published choices are not unanimous: constant LR is not automatically wrong, and cosine is not proven superior for this dataset. A 2026 comparison of nine LoRA variants found that careful hyperparameter tuning substantially narrowed method differences; its tasks were reasoning, coding and instruction following, **not Syrian MT**. This is evidence to prioritize a fair vanilla-LoRA baseline, not a prediction of equal MT scores. [Lee et al., 2026](https://arxiv.org/abs/2602.04998).

### 2.2 Proposed reference configuration

Keep the current model, data sample, answer format and rank fixed for the first recipe comparison. Start each SFT trial from the same original checkpoint; do not repeatedly continue whichever adapter happened to be trained last.

| Knob | Proposed setting | Reason / qualification |
|---|---|---|
| Model | Exact existing `Qwen/Qwen3-8B` revision and tokenizer; thinking disabled | Preserve the known fit and successful round-2 adaptation. “Base” below means no project adapter, not a switch to a separate `-Base` checkpoint. |
| Quantization | NF4, double quantization, bf16 compute; embeddings/output head frozen | Keep the measured path. GGUF Q4_K_M is an inference quantization, not the bitsandbytes training format. |
| LR | **5e-5** initially; compare **1e-4** under the same new schedule | Lower the adaptation rate without jumping orders of magnitude. Consider 2e-4 only if learning curves and generated examples indicate underfitting. |
| Scheduler | Linear warmup for **5% of optimizer updates**, then cosine decay to zero | With ≈1,168 windows, batch 8 and two epochs: ≈292 updates, about **15 warmup updates**. Do not use a copied 200/1,000-step warmup. |
| Batch | Microbatch **1**, accumulation **8**, effective **8 windows**, roughly 64 target cues before overlap | Does not require eight examples' activations simultaneously. Compare accumulation 1 at matched data exposures; do not keep max optimizer steps fixed across batch sizes. |
| Duration | At most **two epochs** initially; save **0.5, 1 and 2 epoch** checkpoints | Overlapping windows already repeat cues. Pick on dev translation/correction quality, not minimum training loss. Try epoch 3 only if dev still improves. |
| Adapter | **r16, α32, dropout .05**, no bias training, existing q/k/v/o/gate/up/down | This shape already works locally and matches the dialect-MT SFT precedent. Leave embeddings and `lm_head` out of trainable targets. |
| Optimizer | AdamW, β=(.9,.999), ε=1e-8, weight decay **.01**, gradient norm **1.0** | Match current defaults explicitly. An 8-bit optimizer is a memory option, not an expected quality improvement. |
| Loss | Answer-only, including corrected Arabic, English, format and EOS | Preserve the task that helped round 2. Log separate Arabic/English token losses and supervised-token counts. |
| Length | Measure full prompt+answer lengths; initial ceiling **2048 only if it retains every target** | If examples exceed it, shorten the brief/context or split at cue boundaries; never silently truncate the answer. Use 3072/4096 only after a memory probe. |
| Memory | Checkpointing; training `use_cache=False`; keep the sliced head or a verified fused loss | Explicitly disable KV-cache creation during training. Do not restore blanket fp32 upcasting through a trainer migration. |
| Packing | **Off** for the first comparison | Existing microbatch one has no inter-example padding waste. Packing can increase live sequence length without eliminating repeated briefs. |
| Randomness | Seed **42** for screening; finalists also **17 and 123** | Seed Python/NumPy/Torch, initialization, shuffling and sampling. Keep seed 13's Levanti subset identical across comparisons. |

**Accumulation must be mathematically defined.** Current loss weights every window equally, independent of answer length. To preserve that objective, average eight per-window mean losses, clip/step once after accumulation, and correctly scale the last partial group. If changing to global answer-token weighting, use `sum(token losses)/sum(unmasked tokens)` over the accumulation group and call that a separate objective change. Averaging per-microbatch token means is not equivalent when lengths differ. Hugging Face and Unsloth documented this accumulation-normalization issue: [HF explanation](https://huggingface.co/blog/gradient_accumulation), [Unsloth analysis](https://unsloth.ai/blog/gradient).

At batch 8, one epoch has about **146 updates**, versus 1,168 now. Accumulation reduces noisy updates but does **not** create new independent examples or guarantee improvement; neighboring windows share targets. Compare equal epochs/target-token exposure, report both update count and exposure, and do not apply a mechanical 8× LR multiplier. If batch 8 underfits, choose batch 4 or more exposure based on dev results.

### 2.3 Rank, modules and alternative adapters

| Method | Evidence and mechanism | Decision at this scale / serving implications |
|---|---|---|
| **Vanilla LoRA r8/16/32** | QLoRA-style adaptation commonly covers attention **and** MLP linear layers; PEFT supports `all-linear`. [PEFT](https://huggingface.co/docs/peft/en/developer_guides/lora) | Keep r16 first. Compare r8/α16 if overfitting or memory pressure; r32/α64 only after LR/exposure are tuned and memory permits. Do not assume more rank supplies missing idiom supervision. |
| **rsLoRA** | Changes scaling from α/r to **α/√r**, designed to prevent learning suppression at higher ranks. [paper](https://arxiv.org/abs/2312.03732) | Cheap optional ablation; no verified Syrian MT gain. At r16, α32 changes scale **2 → 8**. Use α8 to match initial scale 2, or explicitly tune LR/scaling; toggling the flag at α32 is confounded. Verify converted adapter scaling in llama.cpp. |
| **DoRA** | Separates magnitude from direction; paper reports improvements over LoRA. [paper](https://arxiv.org/abs/2402.09353) | PEFT notes extra overhead, especially relevant with only ~0.8 GB nominal headroom. No evidence here for 1.2k noisy MT windows. Prefer a merged ordinary model for deployment unless native adapter conversion is verified. [implementation constraints](https://huggingface.co/docs/peft/en/developer_guides/lora) |
| **PiSSA** | SVD-based initialization; reports faster convergence and reduced quantization error. [paper, revised 2025](https://arxiv.org/abs/2404.02948) | Initialization mutates the base decomposition. An adapter is not automatically interchangeable with an untouched Qwen GGUF. PEFT provides conversion by subtracting the initialized adapter; factorization/loading can need extra memory. Unproven benefit here; defer. [PEFT conversion](https://huggingface.co/docs/peft/en/developer_guides/lora) |
| **LoRA+** | Uses different learning rates for A/B, improving efficiency in its experiments. [paper](https://arxiv.org/abs/2402.12354) | Low additional state and ordinary LoRA weights at inference. If attempted, screen **LR_A=5e-6, LR_B=8e-5** (ratio 16), keeping r16/α32. This is a proposed starting pair, not a validated optimum. Do not multiply the current 1e-4 B rate by 16 without retuning. |
| **NEFTune** | Adds noise to input embeddings while training; its large AlpacaEval gains are conversational instruction-tuning results. [paper](https://arxiv.org/abs/2310.05914) | Optional **noise α=5 vs off** after the recipe works; little state overhead. No established gain for names, number fidelity, or our MT task. Embedding noise is **not ASR-like text noise**. Confirm hooks work through PEFT and remove them for evaluation. |

The current seven Qwen3 module names must not be copied indiscriminately to another architecture. In particular, Jais 2 uses an ungated MLP; Qwen3.5 mixes linear-attention blocks with full-attention blocks. Enumerate trainable modules and counts for each model. [Jais 2 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/jais2.cpp), [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json).

### 2.4 Answer-only loss versus full-sequence loss

**Keep answer-only as the reference.** It trains the conditional translation/correction objective directly; ALMA explicitly masks prompts. Here the repeated ~1k-token brief and source ASR are inputs, and supervising them would redirect much of the loss toward boilerplate and noisy Arabic. That last sentence is a project-specific hypothesis, not a universal finding.

There is counterevidence to blanket claims that input masking is always best: **Instruction Tuning With Loss Over Instructions** found benefits from instruction modeling, including small datasets with long instructions and short outputs. It did not establish that memorizing a repeated show brief improves translation. A fair optional comparison would add loss on **source dialogue only**, keep fixed system/brief tokens masked, and cap source-loss weight at .1; compare with answer-only at matched target exposure. [Shi et al., NeurIPS 2024](https://arxiv.org/abs/2405.14394).

Masking labels is also distinct from avoiding output-logit allocation: a conventional model forward can allocate `[batch, sequence, vocabulary]` logits even for `-100` prompt labels. Keep the current sliced head or use a fused linear+CE path. Axolotl's CCE documentation describes avoiding that full tensor; merely enabling completion-only labels does not provide this saving. [CCE integration notes](https://docs.axolotl.ai/docs/agents/new_model_support.html#cut-cross-entropy-cce).

Preserve supervision for both tracks. If Arabic copying overwhelms English, test **separately averaged Arabic and English losses, weighted .5/.5**, plus format/EOS supervision. This is an unvalidated proposal and changes the objective; report Arabic correction precision alongside English quality. Never suppress all Arabic loss just because the headline metric scores English.

### 2.5 Sequence packing

Packing improves utilization when examples would otherwise be padded; it is not extra training data or a quality intervention. The current dynamic microbatch-one loop already avoids batch padding. Packing two ~1.5k–2k-token windows can increase activation memory enough to defeat the 8 GB budget. It also does not automatically deduplicate the repeated system brief.

If shorter examples or a compact brief make it worthwhile, use packing with **attention isolation between examples**, preserved answer masks, EOS boundaries and no split cue/answer. Start at the same validated total sequence ceiling and recompute exposure-based schedules. Do not concatenate unrelated episodes with ordinary causal attention and assume EOS blocks information flow. Modern TRL documents packing and its truncation/splitting strategies; inspect packed outputs rather than relying on defaults. [TRL packing/memory guide](https://huggingface.co/docs/trl/en/reducing_memory_usage).

## 3. Tooling and the 8 GB constraint

### 3.1 The relevant memory arithmetic

Training fit is weights + unquantized tensors + adapters/gradients/optimizer + activations + loss workspace + allocator/CUDA overhead. It is not GGUF size or active-parameter count. These calculations are **estimates**, with the known 7.2 GB run as the local anchor.

- Qwen3-8B has 151,936 vocabulary entries and hidden size 4,096, with untied embeddings/head. Those two BF16 matrices alone occupy about **2.32 GiB**. Its r16 adapters on the seven named projections contain approximately **43.65M parameters**, calculated from the published 36-layer dimensions. FP32 Adam moments alone are about .325 GiB; adapter/gradient dtype determines the rest. [official config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json).
- Qwen3.5-9B has vocabulary 248,320 and hidden size 4,096, also untied: keeping both matrices in BF16 costs about **3.79 GiB** before the rest of the model. Quantization exclusions and loader behavior therefore matter greatly. Its hybrid attention can help inference context memory without making training weights/activations free. [official config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json).
- A hypothetical all-4-bit 14B model starts at **7 GB decimal / 6.52 GiB**, before scales, higher-precision embeddings/head, adapters or activations. Real loaders can exceed 8 GB with weights alone. Axolotl's Qwen3-14B example reports **9.264 GB CUDA usage after loading**, before training; Qwen's Unsloth guide targets a **16 GB T4**, not an 8 GB card. [Axolotl example](https://docs.axolotl.ai/examples/colab-notebooks/colab-axolotl-example.html), [Qwen guide](https://qwen.readthedocs.io/en/latest/training/unsloth.html).

Measure both framework **peak allocated/reserved** and whole-device usage. A 7.2 GiB allocated peak is not permission to run beside ASR or a game. Schedule training with the GPU exclusively available, use restartable checkpoints, and do not assume an optimizer-state offloader solves the much larger frozen-weight budget.

### 3.2 Comparison

| Tool | Answer-only support and useful features | Fit/speed judgment for this project |
|---|---|---|
| **Existing loop** | Already slices head logits and avoids the fp32 embedding/head upcast; known 7.2 GB peak. | **Strongest local evidence of fit.** Accumulation and a scheduler can be added without a framework migration. Lacks checkpoint selection, explicit seeds and richer logging today. |
| **Unsloth Core + TRL** | Qwen3 is supported; optimized kernels/checkpointing; `train_on_responses_only` masks non-answer content. Qwen-style role markers are `<\|im_start\|>user\n` / `<\|im_start\|>assistant\n`; inspect actual token masks, thinking prefix and EOS. [support](https://unsloth.ai/blog/qwen3), [masking docs](https://unsloth.ai/docs/basics/troubleshooting-and-faqs) | **Best optimization candidate, conditional on a smoke test.** Vendor claims ~2× speed and ~70% less VRAM against standard HF/FA2, not our sliced loop. Do not subtract 70% from 7.2 GB or promise a 3070 throughput number. |
| **TRL SFTTrainer** | Prompt-completion datasets support `completion_only_loss=True`; conversation training has `assistant_only_loss=True`. Current Qwen3 support patches the assistant generation mask. Built-in accumulation/schedules/checkpoints. [SFT docs](https://huggingface.co/docs/trl/en/sft_trainer) | A maintained training interface, **not by itself a memory reduction**. Verify the chosen version's loss path, adapter dtypes and preparation behavior before replacing the sliced head. A standard full-logit forward can OOM. |
| **Axolotl** | Assistant-role labeling, `train_on_inputs: false`; packing, CCE/Liger/chunked loss, checkpointing and explicit activation/layer offloading. [labeling](https://docs.axolotl.ai/docs/dataset-formats/conversation.html), [optimization guide](https://docs.axolotl.ai/docs/optimizations.html) | Good if a declarative configuration or **frozen-layer CPU streaming** is needed. More moving parts for one small dataset. Qwen3 integration is documented; don't infer that every kernel supports every Qwen3.5 path. |
| **torchtune** | PyTorch-native QLoRA, activation offloading, Qwen3 support added May 2025. Repository now states development wound down in 2025 and it is **no longer actively maintained**. [repository](https://github.com/meta-pytorch/torchtune), [QLoRA tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/qlora_finetune.html) | Do not start a new Qwen3.5 migration here. Older low-memory recipes are useful references, but no verified current 8 GB advantage for this workload. |

**3070 practitioner evidence is mixed and version-dependent.** A first-person Qwen/Unsloth repository says its RTX 3070 8 GB validation covers **Qwen3-1.7B/4B**, and that its **8B configuration OOMs with Unsloth 2026.3.4+**. This is one report, not proof that our 8B workload cannot fit; it is a concrete reason to retain the working loop and pin tested versions. I found no reproducible, like-for-like 3070 benchmark for our Qwen3-8B, long brief, answer-only workload. [practitioner README](https://github.com/techwithsergiu/qwen-qlora-train).

**Qwen3.5 support exists, but it is not the same recommendation.** Unsloth's current guide supports dense 0.8B/2B/4B/9B/27B and larger MoE variants, requires Transformers v5, and explicitly discourages **4-bit QLoRA even for dense Qwen3.5** because of larger quantization differences. Its BF16-LoRA figures are **22 GB for 9B**, 10 GB for 4B and 5 GB for 2B; they are not 9B QLoRA measurements on 8 GB. This conflicts with casual “Qwen3.5 trains in 5 GB” headlines. [official fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune).

For a prospective migration, compare one fixed batch's rendered prompt, answer-token IDs, EOS mask and loss before optimizing. Then exercise **forward + backward + optimizer step** on the longest example and an accumulation boundary, save/reload the adapter and generate a few held-out outputs. A load-only or first-forward memory figure is insufficient. Keep r16/.05 and the same original model revision for this parity check; switching to a dynamic quantization at the same time would add another variable.

### 3.3 Can software unlock 12–14B or longer context?

**Longer context on 8B: plausible; 12–14B entirely on the GPU: not an evidence-backed expectation.** Prioritize efficient loss and checkpointing, then test 2048 → 3072 → 4096 with measured sequence lengths. There is little justification for 8k context on eight subtitle lines before showing that more context helps the hard cases. A shorter, relevant brief may save more useful compute than packing.

**CPU offload during training is a real option**, not just inference `device_map="auto"`. Axolotl now documents `layer_offloading: true`, which streams frozen decoder parameters from pinned CPU memory and prefetches the next layer for forward/backward. `activation_offloading: hidden_states` moves checkpoint inputs instead. They solve different memory terms; the output head remains on the GPU. **Unverified here:** the complete QLoRA + model + fused-loss combination at the required versions. [offloading documentation](https://docs.axolotl.ai/docs/gradient_checkpointing.html).

With sufficient host RAM, this could make a 12–14B LoRA trial fit **in principle**, but PCIe transfers, adapter state and the head still matter. Budget at least **32 GB host RAM, preferably 64 GB**, as a planning allowance rather than a measured requirement; host RAM/CPU/PCIe details were not supplied. Do a 20-step feasibility benchmark before planning a sweep. No trustworthy minutes/epoch figure is available. Larger offline annotation/teacher inference is a more economical first use of offload than larger-model training.

For **inference**, llama.cpp supports partial GPU layer placement, and the existing `Lease::start` appends extra arguments after `-ngl 99`; a future configuration should request a lower layer count explicitly. Try **Qwen3-14B** or **Gemma-3-12B-it**, whose releases are verified, at Q4 with CPU-resident layers only after the small-model comparison. Record prompt processing, generation speed, peak RAM/VRAM and full-episode latency. Do not extrapolate all-GPU 8B token rates to offloaded dense models. [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Gemma 3 family card](https://huggingface.co/google/gemma-3-4b-it), [llama-server options](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

## 4. Base models: which Arabic evidence is relevant?

### 4.1 Benchmark map

| Benchmark | Actually measured on Levantine/Syrian? | What it can tell us |
|---|---|---|
| **AraBench**, COLING 2020 ([paper](https://aclanthology.org/2020.coling-main.447.pdf)) | **Yes:** Syrian MDC, MADAR **Damascus/Aleppo**, Beirut, Jordanian/Palestinian and other Levantine sets. Dialect → English. | Useful task/domain breakdown. Older NMT baselines do not rank 2026 LLMs. Table 4's Damascus result varies from 25.9 to 33.1 BLEU across training mixtures, illustrating sensitivity to data mix. |
| **MADAR / AraDiCE**, COLING 2025 ([paper](https://aclanthology.org/2025.coling-main.283.pdf)) | **Yes**, MADAR city varieties; AraDiCE reports aggregated **Levantine → English** BLEU. Travel-style clean text, not noisy drama. | Table 2: original Jais 36.1, AceGPT 35.3, Llama3 23.0, Mistral 12.2. These are the paper's older checkpoints; **not Jais 2** or Qwen3.5. |
| **AL-QASIDA**, ACL Findings 2025 ([paper](https://aclanthology.org/2025.findings-acl.1137/)) | **Yes:** Syrian MADAR and North Levantine FLORES among eight dialect varieties. | Direct dialect MT and dialect-generation analysis; supports evaluating SILMA and the distinction between understanding dialect and generating it. No controlled Qwen3.5-9B vs tuned Qwen3-8B result. |
| **TARJAMAT**, ArabicNLP 2023 ([paper](https://aclanthology.org/2023.arabicnlp-1.6/)) | **Jordanian and Palestinian TV dialogue**, not specifically Damascene. | Domain is closer than travel phrases. Useful independent supplemental test; the published model roster is old. |
| **NADI 2023/2024 MT** ([2024 organizers](https://arabicnlp2024.sigarab.org/shared-tasks), [2023 proceedings](https://aclanthology.org/events/arabicnlp-2023/)) | Levantine includes **Jordanian/Palestinian**, not a demonstrated Syrian-drama test. | MT direction is **dialect → MSA**. Good evidence about normalization; unsuitable as an English subtitle leaderboard. |
| **OSACT6 2024 MT** ([overview](https://aclanthology.org/2024.osact-1.11/)) | **Levantine** among five groups; no separately established Damascene slice here. | Also **dialect → MSA**; 200 dev items per dialect, 1,888 total blind test items. Converting dialect toward MSA is not our corrected-Arabic objective. |
| **IWSLT 2025 low-resource ST** ([findings](https://aclanthology.org/2025.iwslt-1.44/), [KIT system](https://aclanthology.org/2025.iwslt-1.20/)) | **North Levantine `apc` → English**. | Direct speech-translation evidence and relevant corpus choices; not a head-to-head ranking of the candidate chat models. |
| **NADI 2026 speech tasks** ([overview](https://arxiv.org/abs/2609.27086), [organizers](https://nadi.dlnlp.ai/2026/)) | SLT includes Levantine Casablanca speech, with Jordan/Palestine results. No isolated Damascene result verified here. | More realistic noise/domain than FLORES; end-to-end scores conflate ASR and translation. |
| **BALSAM**, ArabicNLP 2025 ([paper](https://aclanthology.org/2025.arabicnlp-main.21.pdf)) | Contains dialect translation: Phase 1 has **1,200 test / 600 dev** items. I could not verify a separately scored Syrian/Levantine → English subset. | Broad 78-task/14-category platform. The paper itself discusses misleading lexical-overlap rankings and examines outputs manually. Do not select a translator from its aggregate alone. |
| **Open Arabic LLM Leaderboard v2** ([live board](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard), [rendered tasks](https://oall-open-arabic-llm-leaderboard.hf.space/)) | No isolated Syrian → English MT score in the displayed suite: AlGhafa, ArabicMMLU, EXAMS, MadinahQA, AraTrust, ALRAGE, ArbMMLU-HT. | Arabic knowledge/understanding screening, not our task ranking. |
| **DialectalArabicMMLU**, 2025/2026 ([paper](https://arxiv.org/abs/2510.27543)) | **Explicit Syrian**, alongside four other dialects, with manually adapted questions. | Syrian academic multiple-choice understanding, **not MT or ASR robustness**. |
| **AlexandriaX 2026** ([overview](https://arxiv.org/abs/2609.22796)) | Multiple dialects and contextual dialogue; inspect variety labels for any proposed slice. | English → dialect dialogue, cross-dialect translation and error annotation. Valuable recipes, but direction/domain differ from this application. |

### 4.2 A direct comparison that earlier research did not include

The **Jais 2 technical report, Table 15, PDF page 37** reports these dialect → English BLEU scores. This is **vendor-authored evaluation**, not an independent replication. [Jais 2 report](https://arxiv.org/pdf/2608.13580).

| Checkpoint in that table | North Levantine → EN (`apc_n`) | South Levantine → EN (`apc_s`) |
|---|---:|---:|
| ALLaM-7B-Instruct-preview | **38.25** | 40.86 |
| Jais 2 8B | 36.68 | **40.98** |
| Command R7B Arabic 02-2025 | 36.59 | 39.39 |
| Gemma-3-4B-it | 36.52 | 39.79 |
| Aya Expanse 8B | 34.86 | 37.65 |
| Fanar-1-9B-Instruct | 33.10 | 35.55 |
| SILMA-9B-Instruct-v1.0 | 32.73 | 36.93 |
| Qwen3-8B | 30.63 | 34.51 |
| Llama-3.1-8B-Instruct | 27.20 | 29.83 |

These are clean North/South Levantine FLORES BLEU scores, not Damascene-drama chrF++. Qwen3.5-9B/Gemma 4 are absent. Benchmark families also appear in the authors' training inventory: scrutinize split hygiene. Use this table to **shortlist**, then measure our actual task.

### 4.3 Verified candidates and practical ranking

| Candidate | Evidence / integration | Recommendation for shu-2al |
|---|---|---|
| **`Qwen/Qwen3-8B`** | Existing locally successful QLoRA, Apache-2.0, thinking/non-thinking support. [card](https://huggingface.co/Qwen/Qwen3-8B) | **First training baseline.** Known fit and +8-ish local improvement outweigh generic model rankings until a matched comparison exists. |
| **`Qwen/Qwen3.5-9B`** | Real released model, Apache-2.0; current production model. Official WMT24++ scores are averages across languages, not a Syrian score. [card](https://huggingface.co/Qwen/Qwen3.5-9B) | **Keep as inference control.** Same exact first-pass prompts needed for comparison. Lower priority for 8 GB QLoRA because of the documented quantization warning and large embeddings/head. |
| **`humain-ai/ALLaM-7B-Instruct-preview`** | Former `ALLaM-AI` URL redirects here; Apache-2.0; Transformers `llama` architecture. [card](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview) | **First alternative to screen.** Direct North Levantine result above and a conventional architecture. “Preview” is the checkpoint name; do not substitute a hosted ALLaM service. |
| **`CohereLabs/c4ai-command-r7b-arabic-02-2025`** | Approximately **8B total**, despite “7B”: 7B transformer + 1B embeddings. Arabic/English, CC-BY-NC release; card emphasizes MSA and instruction/length control. [card](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025) | **Second alternative.** Direct Levantine MT evidence comes from the Jais report, not a claim that its own card targets Syrian. Private local noncommercial context makes it worth testing; verify load/adapter memory before SFT. |
| **`inception42/Jais-2-8B-Chat`** | Real 8B Arabic/English checkpoint, Apache-2.0, 8,192 context, custom `jais2` architecture; card recommends current Transformers integration. [card](https://huggingface.co/inception42/Jais-2-8B-Chat) | **Third alternative**, or earlier if prebuilt GGUF is easy to obtain. llama.cpp has native Jais 2 code; optimized QLoRA/loss integration is less established than Llama/Qwen. Do not assume Unsloth's fast Qwen kernels apply. [native implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/jais2.cpp) |
| **`QCRI/Fanar-1-9B-Instruct`** | Actually 8.7B, Gemma2-derived, 4,096 context; claims Levantine support; continually pretrained on Arabic/English. [card](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) | Secondary candidate. Its direct English MT numbers above are weaker than ALLaM/Command; dialect QA strength alone does not justify a migration. |
| **`silma-ai/SILMA-9B-Instruct-v1.0`** | Gemma2-derived, Gemma license; measured in AL-QASIDA and the Jais table. [card](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) | Worth a cheap inference check if other candidates fail; not the first new training base. |
| **`CohereLabs/aya-expanse-8b`** | Real 23-language model including Arabic; CC-BY-NC; direct North/South Levantine measurements above. [card](https://huggingface.co/CohereLabs/aya-expanse-8b) | Behind Command R7B Arabic for this Arabic/English specialization. General multilingual preference training is not Syrian-drama evidence. |
| **Gemma 3: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`** | Released 1B/4B/12B/27B family; 4B has direct Levantine MT evidence above. [family card](https://huggingface.co/google/gemma-3-4b-it) | **4B is a useful efficiency challenger** if rapid ablation matters; do not invent a “Gemma-3-9B.” 12B needs an inference/offload budget check and was not better than 4B in that particular vendor table. |
| **Gemma 4: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`** | Verified family. E2B is **5.1B total / 2.3B effective**; E4B **8B total / 4.5B effective**, with large per-layer embeddings; Apache-2.0. [card](https://huggingface.co/google/gemma-4-E4B-it) | Legitimate ≤9B candidates, but no verified Damascene → English result found. “E4B” does not mean only 4B weights in memory. Test before promoting. Larger 12B/31B/26B-A4B variants are not ≤9B by total weights. |
| **Llama: `meta-llama/Llama-3.1-8B-Instruct` and Arabic adaptations** | Existing direct Levantine evidence above is weaker. AceGPT's old strong 13B AraDiCE result does not establish AceGPT-v2-8B as best. [Llama card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [AraDiCE](https://aclanthology.org/2025.coling-main.283.pdf) | Mature tooling is useful, but no task-specific reason to prioritize vanilla Llama over Qwen/ALLaM. Do not treat Llama 4's active parameters as an 8 GB dense-weight budget. |
| **`inception42/Jais-3-7B`**, newer 2026 option | Verified new checkpoint; Arabic/English, Apache-2.0, `k2_horizon` architecture, linked GGUF release. [card](https://huggingface.co/inception42/Jais-3-7B) | Optional inference-only screen. General Arabic numbers do not establish Syrian MT superiority; new architecture and reasoning behavior add training/latency risk. Recheck its exact template and reasoning controls. |

There is no public apples-to-apples comparison here that proves Qwen3.5-9B, Gemma 4, or Jais 3 is the best **noisy Damascene drama → short English + corrected dialect** model. Select the new base using that task, including refusal/profanity behavior, names, number fidelity, Arabic overcorrection and latency. Broad Arabic cultural alignment can be helpful, but test whether it changes dramatic dialogue in undesirable ways rather than assuming it does.

## 5. MT-specific training strategies

### 5.1 ALMA-style monolingual adaptation, then SFT

ALMA's transferable idea is to improve source-language modeling before learning the translation interface. The original work used Llama2 with large monolingual adaptation and then high-quality parallel SFT; released 7B/13B pretraining stages consumed **20B/12B tokens**. It is not evidence that continued pretraining on 3,671 already-used subtitle lines will help. [ALMA paper](https://arxiv.org/abs/2309.11674), [training/model notes](https://github.com/fe1ixxu/ALMA#download-alma-models-and-dataset-).

**Feasible reduced experiment, lower priority:** collect **1–5M tokens of clean, distinct Syrian/Levantine dialogue**, deduplicate against all dev/test episodes and benchmarks, exclude raw repetition loops, and mix about **20% English subtitle prose** as retention data. QLoRA r16, LR **2e-5**, 3% warmup/cosine, microbatch 1 × accumulation 8, up to one token-budget pass, then the same supervised run. Keep the same adapter across the two stages so the first stage's update is retained. These amounts/mixture are proposals, not ALMA's published recipe.

At an illustrative observed throughput of 100–500 **training tokens/s**, 1–5M extra tokens would cost roughly **0.6–14 hours**; this is a scenario calculation, not a 3070 measurement. Measure after compilation and include backward/recompute. Given Qwen already understands much Arabic and the remaining errors are sparse idioms/noise, curated parallel examples are the higher-priority use of time. Do not train a language model on raw ASR and expect it to unlearn ASR errors.

### 5.2 Tower and X-ALMA: borrow the applicable pieces

**Tower** continued Llama2 training on 20B tokens with approximately two-thirds monolingual and one-third parallel text, then trained translation, post-editing and related instructions. The original ten-language mixture **does not include Arabic**. Its ablations support parallel data during continued pretraining and show that task transfer/interference needs measurement. Borrow the recipe idea, not an assumption that an original Tower checkpoint knows Syrian. [Tower paper](https://arxiv.org/html/2402.17733v1).

**Tower+ (2025)** extends specialization while preserving general instruction ability, using continued pretraining, SFT, preferences and RL; 2B/9B/72B models are described. `Unbabel/Tower-Plus-9B` exists, but I found no direct Syrian MT measurement that warrants putting it ahead of the shortlist. Reproducing its full training pipeline is out of budget. [paper](https://arxiv.org/abs/2506.17080), [9B card](https://huggingface.co/Unbabel/Tower-Plus-9B).

**X-ALMA (ICLR 2025)** uses language-group modules and five training stages for 50 languages. Its architectural grouping addresses many-language interference; one Syrian→English task does not need that machinery. Its **Adaptive-Rejection Preference Optimization (ARPO)** addresses excessive rejection when good and bad translations share many tokens. The authors show that some preference objectives improve semantic metrics while sharply changing surface style. Borrow that warning when using chrF and near-identical subtitle edits; do not equate ARPO with unrelated algorithms sharing the acronym. [paper](https://arxiv.org/html/2410.03115v2).

### 5.3 Preference optimization: feasible, but labels matter more than the acronym

| Method | Relevant MT evidence | Feasibility / suggested priority |
|---|---|---|
| **CPO** | ALMA-R used **22k sources** and small adapters, comparing candidate translations; reference-free contrastive loss with a positive NLL/behavior-cloning term. [paper](https://arxiv.org/html/2401.08417v3) | **Best first preference experiment.** One policy model, no frozen-reference copy. Still needs chosen/rejected completion computation; a concatenated pair can OOM where one SFT example fits. |
| **DPO** | Post-edits can define real preferences; best outcomes in *Post-edits Are Preferences Too* followed SFT. [WMT 2024 paper](https://aclanthology.org/2024.wmt-1.122/) | Feasible if reference log-probabilities are precomputed using the **frozen SFT checkpoint**, then released from VRAM. Current TRL documents this. If sharing the same quantized base, disabling the current adapter is not a valid reference for a nonzero frozen SFT adapter. [DPO docs](https://huggingface.co/docs/trl/en/dpo_trainer) |
| **KTO** | 2025 Indian-language work used QLoRA Llama3.1-8B and progressively perturbed negatives; selected language results improved by roughly 2.85–4.01 chrF over SFT. These are **not Arabic** or our sample size. [paper](https://aclanthology.org/2025.mtsummit-1.26/) | Useful when the editor supplies good/bad labels without pairs. Needs a reference/KL baseline and balanced label handling; not automatically cheaper than CPO. [KTO docs](https://huggingface.co/docs/trl/en/kto_trainer) |
| **ARPO** | X-ALMA's similarity-dependent rejection reduces pressure against tokens shared by chosen/rejected translations. | Potentially relevant to minimally edited subtitles, but extra implementation/evaluation work. Defer until CPO/DPO is measured; no 8 GB Syrian result found. |
| **Online RL / GRPO** | The 2026 MADAR study reports dialect→English BLEU **17.4 → 34.9** using reasoning-oriented RL with small Qwen2.5 models. It used **three A100 80 GB GPUs** for RL. | Real dialect evidence, but much larger rollout/reward compute and different templates/data. Not the next 8B/8 GB experiment. Its result is not proof that on-device RL will cheaply fix Damascene idioms. |

The 2025 CPO domain-adaptation preprint reports that 14.7k preference examples approached a 160k+ SFT baseline for software-localization language pairs. That supports leveraging edits; it does **not** establish a gain from 100–500 pairs or justify automatically labeling every base output “rejected.” [study](https://arxiv.org/html/2510.27556v1).

For this project, form a record from **the exact original prompt/raw ASR/context**, a demonstrably erroneous generated ar+en answer, and its reviewed replacement. Keep source/context identical between chosen and rejected. A user changing Arabic, timing or scene boundaries may change the problem itself; do not turn every `cue_edits` before/after row into a preference pair. Exclude equivalent stylistic alternatives, unchanged outputs and uncertain/garbled sources. Prefer substantive errors: invented facts, wrong negation, wrong name, literal idiom, unwanted MSA normalization.

Proposed pilot: **200–500 reviewed pairs**, beginning from the best SFT adapter; CPO β **.1**, positive NLL weight **1**, LR **5e-6**, 5% warmup/cosine, effective batch **4**, **one epoch**. Use 1–4 target cues per record with enough shared context, not two full 8-line answers concatenated on the GPU. Compute chosen/rejected forwards sequentially with checkpointed/fused loss if necessary, or use validated offload. The short-record SFT control must use the same data/context so a gain is not attributed solely to the objective. No defensible chrF gain forecast exists at 200–500 pairs.

### 5.4 Distillation and multi-task learning

The existing Claude annotations are already **teacher-generated sequence supervision**. Self-distilling the current model's unreviewed translations would reinforce its specific idiom/name mistakes. Use a stronger teacher to propose translations/Arabic corrections for **training-only** hard cases, verify against audio/context, and retain provenance and uncertainty. Sequence-level hard targets fit the current SFT memory budget; online teacher logits or simultaneous teacher/student loading are unnecessary.

**CycleDistill (2025)** reports strong low-resource MT gains from repeated generation and fine-tuning using monolingual data in three Indian languages, starting with only a few translation demonstrations. Its initial baselines and language-resource setting differ substantially from our already fine-tuned model; the large reported gains are not transferable expectations. [paper](https://arxiv.org/abs/2506.19952).

Keep **correction + translation**: Dou et al. found joint transcript/translation refinement better than translation-only refinement, including context-aware parameter-efficient fine-tuning, on MuST-C/CoVoST2 tasks. That supports our task shape but is not a Syrian-specific ablation. Our round-2 gain also changed data quantity and Levanti, so it did not isolate the Arabic auxiliary task. [Speech Translation Refinement, 2025](https://arxiv.org/abs/2501.15090).

Useful small additions are clean-Arabic → English/copy-Arabic examples and selected correction-focused examples. Avoid launching many unrelated Arabic tasks: they consume the same finite supervised-token budget. If mixing task types, label them explicitly and keep the deployment format represented.

## 6. ASR robustness: exploit the pairs we already have

### 6.1 Real corruptions first

The 858 explicit corrections are more relevant than random character typos. They can reveal dialect→MSA normalization, phonetic substitutions, names, missing words, splitting/merging, and repetitions. Audit a sample against audio before treating the model annotations as truth; the annotator may have inferred plausible dialogue from the same corrupted transcript.

Proposed dataset variants, each at a **fixed window/target-token budget**:

1. **Raw-source baseline:** current raw ASR → corrected Arabic + English, with a provenance flag for explicit correction versus copy fallback.
2. **Raw + clean views:** on **25% of eligible corrected-source windows**, replace raw lines with their reviewed corrections, retaining identical ar+en targets. Keep 75% of eligible windows raw; unchanged windows remain raw. This teaches the same translation under both source views without making eight-line answers longer. Count unique sources separately from views.
3. **Empirical noise augmentation:** replace, rather than append, **20% of training-window presentations** with a lightly corrupted version of reviewed clean text. Estimate corruption types/rates only from training raw/fix alignments. Corrupt one or two lines per window; preserve enough evidence to recover meaning and keep labels fixed. Avoid independently applying the same numeric rate to character noise, word deletion and repetition.

Do not introduce both 2 and 3 in the first data ablation. Use separate runs to learn whether actual raw/clean invariance is enough before building a corruption generator.

**Why empirical noise:** PiDA (Interspeech 2026) studies Vietnamese ASR substitutions and reports up to **+2.04 BLEU** from phonetically informed corruptions, with small clean-text improvements. This supports realistic confusions rather than random edits, but the language, ASR and MT models differ. [PiDA](https://arxiv.org/abs/2606.12911). A 2025 robustness study also shows modern MT/LLMs are already tolerant of several spelling/formatting noises; noise augmentation is not automatically beneficial. [Peters & Martins, ACL 2025](https://aclanthology.org/2025.acl-long.122/).

Concrete Syrian error templates should come from observed pairs: clitic boundaries, a character-name phonetic neighbor, a dialect expression replaced with an MSA-looking phrase, or a short duplicate span. Keyboard typos alone do not model acoustic confusion. Do not fabricate negation/name deletions that make two different intended utterances indistinguishable and then reward confident reconstruction.

### 6.2 Idioms, copy behavior and unrecoverable input

Build **100–200 reviewed training examples** around recurring idioms and name confusions, with multiple speakers/contexts, literal-use counterexamples and varied inflections. The three cited idioms are already known E2 failure cases: adding their answers to training makes E2 a development set. Reserve unseen paraphrases, contexts and idiom families for the new test; no claim of blind improvement can rely on memorizing E2.

For Levanti, keep the current sampled pairs for optimization controls, then separately compare a **quality-filtered Syrian/Levantine, conversational subset**. The dataset contains original and synthetic material and dialect metadata; inspect provenance instead of assuming 2,000 random pairs are representative of Damascus drama. Keep verified-clean copy targets so correction training does not encourage editing every line. [dataset card](https://huggingface.co/datasets/guymorlan/levanti).

The skipped ~250 unrecoverable lines are a distinct failure mode. A text model cannot reliably reconstruct information absent from all input/context. Do not reintroduce guessed fluent English as supervision. Where a user can recover the words from audio, add reviewed triples; otherwise preserve an explicit uncertainty label in research evaluation and test conservative behavior. If an abstention target is introduced later, it must agree with the subtitle pipeline's accepted output policy/schema; do not silently train a new `<unrecoverable>` token that production cannot interpret.

Track Arabic **unnecessary-edit rate on reviewed-clean inputs**, correction precision/recall on recoverable-error inputs, name/negation accuracy and hallucination rate. CER alone can reward MSA normalization or penalize valid dialect spelling; normalize orthography consistently while preserving dialect distinctions. The 2025 CoVoGER benchmark provides evidence that speech-recognition/translation correction and N-best inputs merit evaluation, but its 15-language/28-pair setting is not direct validation of a Syrian corruption recipe. [CoVoGER](https://aclanthology.org/2025.emnlp-main.320/).

## 7. Evaluation, cheap ablations and production transfer

### 7.1 A small but credible evaluation design

**No universal minimum test size guarantees detection of +1 chrF.** Variance depends on utterance length, error overlap, scene dependence and the system pair. Treat the present 165-reference-line E2 set as a **development/smoke set** after repeated inspection.

Proposed allocation:

- **Dev:** 150–250 reviewed lines drawn from whole held-out scenes/episodes, covering clean, recoverable-noisy and idiomatic inputs. Use for choosing LR/checkpoints; log every evaluated candidate.
- **New frozen test:** initially **600–1,000 scored lines from at least three untouched episodes**, aiming for 1,000–2,000 over time. Prefer at least one other Syrian drama for a separate transfer slice. These are practical starting targets, not paper-derived statistical minima.
- **Human audit:** 100–200 blinded, paired outputs, using audio and scene context for adjudication; independently reviewed references where possible. Include a random sample and a separately labeled hard-case sample. Score meaning errors, invention, idioms, tone/profanity and subtitle usability, not closeness to Claude's wording.
- **Correction set:** a balanced, reviewed subset of genuinely clean and recoverable-noisy Arabic, with explicit transcript references. Keep its construction independent of model predictions.

Split by **episode before creating windows**, retain cue IDs and reset rolling context at boundaries. Deduplicate train/dev/test against external corpora where practical. Do not split overlapping windows randomly. A benchmark derived from translations already used in model pretraining/SFT is supplemental evidence, not a clean test of our project adapter.

### 7.2 Metrics and significance

Use **corpus chrF++**, preserving current `word_order=2`, case/normalization settings and SacreBLEU version/signature. Do not replace corpus chrF with mean sentence chrF; they are different quantities. Store per-cue raw source, reviewed source when available, output, reference, episode/scene IDs and parse failures.

For every finalist against the baseline:

1. Generate outputs once in original episode order, with rolling model-generated context. Save exact outputs/configs; resampling does not require regeneration.
2. Perform **paired bootstrap**, preferably **10,000 replicates**: resample the same scene blocks for both systems, recompute corpus chrF++ on each sample and report the 95% CI of the **difference**. A simple line-level SacreBLEU test is a useful secondary check, but correlated subtitle lines make its independence assumption optimistic.
3. Report each episode's effect and the pooled effect. With only three episodes, episode-level bootstrap itself has little support; scene-block intervals quantify within-episode uncertainty and do not prove broad show-level generalization.
4. Report training-seed variation separately from test-sampling uncertainty: three seeds for both finalist and baseline when the margin is small. Do not count three seed outputs for the same test line as three independent test examples.

SacreBLEU provides paired bootstrap and paired approximate-randomization tests; its default bootstrap count is 1,000. A simple reproducible **line-level** comparison, to supplement the scene-block analysis, would be:

```sh
sacrebleu refs.en.txt -i baseline.en.txt candidate.en.txt \
  -m chrf --chrf-word-order 2 --paired-bs --paired-bs-n 10000
```

These commands are a future evaluation recipe; they were not run. [SacreBLEU statistical testing documentation](https://github.com/mjpost/sacrebleu#paired-bootstrap-resampling---paired-bs).

Add reference-based **`Unbabel/wmt22-comet-da`**, run **after unloading the translator**, in a small batch or on CPU. It takes source/translation/reference and covers Arabic, but its source encoder is not a validated Damascene-drama judge. Keep the same source for all systems; additionally score against a fixed reviewed-Arabic source where available, since a faithful translation of a corrected utterance can disagree with garbled raw ASR. Do not score each system against its own generated Arabic. [COMET model card](https://huggingface.co/Unbabel/wmt22-comet-da).

Bootstrap **paired sentence COMET score differences** by the same scene blocks. Report COMET's scale/model/version and chrF separately; COMET-22, xCOMET and BLEU values are not interchangeable. Avoid deploying XCOMET-XXL beside an 8B translator just for evaluation. Meaning and hallucination audits should arbitrate disagreements, especially condensed subtitles.

**Planning test size:** estimate scene-block sampling SE from a pilot. As a rough independence-based guide, reducing a 95% CI half-width from `h` to `δ` requires approximately `n_new = n_pilot × (h/δ)^2`. For 80% power at two-sided 5%, a paired-mean approximation is `n ≈ (1.96 + .84)^2 × σ_diff² / δ²`; this is only a planning analogy for nonlinear corpus chrF and correlated scenes. Do not present either formula as an exact chrF sample-size guarantee. Reassess with the actual block distribution.

Predeclare **+1.0 chrF++ as a worthwhile target**, require a positive paired CI plus consistent COMET/human evidence, and accept that a smaller real gain may need more data. This threshold is a project choice, not the test's resolution. For many screened settings, choose on dev then perform one frozen-test comparison; repeated p<.05 tests on the same “test” recreate the current selection bias.

### 7.3 A cheap sequence of comparisons

First compare the saved round-2/round-3 outputs, if available, with paired scoring at **zero new generation cost**. Then reproduce one baseline with an explicit seed and make a small, sequential search:

| Phase | Controlled comparison | Budget discipline |
|---|---|---|
| Runtime parity | Existing loop vs optional Unsloth/TRL on identical data/precision/masks/rank. | Longest-example backward probe first; this is an engineering fit/speed test, not evidence of translation gain. |
| Optimization | Constant-1e-4 baseline → warmup/cosine at same LR/batch/exposure → LR 5e-5 → accumulation 8. | One factor at a time initially. If interactions appear, fill the missing LR × batch cell rather than running a large blind grid. |
| Stopping | 0.5/1/2-epoch checkpoints on the selected schedule. | One trajectory supplies checkpoints. A checkpoint halfway through a two-epoch cosine run is **not** an independently scheduled one-epoch run; label it accordingly. |
| Data | Raw/clean views, then empirical noise, then targeted idioms or filtered Levanti. | Fix exposure and change one data treatment per run. Reuse the same dev examples and subset selection seed. |
| Confirmation | Best candidate vs matched baseline, seeds 17/42/123; frozen test once. | Small differences need this confirmation; eliminate obvious regressions early on dev. |

The local design log reports **744 steps in 29 minutes**, about **2.34 seconds/microbatch**. Scaling that workload to 1,168 windows gives roughly **46 minutes/epoch**, or **91 minutes/two epochs** before evaluation, assuming similar lengths/kernel behavior. This is arithmetic, not a measured round-3 runtime. Accumulation reduces optimizer calls, not forward/backward count, so it will not make two epochs eight times faster.

At the supplied **18 minutes per 434 lines**, a single 1,000-line HF generation pass extrapolates to about **41 minutes**. Full evaluation at every 20 optimizer steps could cost more than training. Use cheap held-out loss/format checks during training, generate on dev at selected checkpoints, then run expensive test/COMET/human checks only for finalists. Exact Unsloth savings against this loop remain unmeasured; budget the working-loop time until a hardware probe demonstrates otherwise.

### 7.4 Do not lose the gain when serving

An adapter belongs to **its exact base architecture/checkpoint/tokenizer**. A Qwen3-8B adapter cannot be attached to production's Qwen3.5-9B GGUF. llama.cpp provides a PEFT→GGUF LoRA conversion tool and server adapter loading; verify actual module mappings and scaling for the chosen architecture/adapter variant. [converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py), [server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

Two legitimate deployment routes are **base Q4 GGUF + converted adapter**, or **merge into the original higher-precision base on CPU, then quantize the merged model**. They are different quantization paths. Avoid assuming that merging directly into a dequantized 4-bit training model recreates the original-weight merge. CPU merging needs host RAM/disk headroom; do not require a 16-bit 8B merge on the GPU.

Verify in stages: (a) HF adapter vs llama.cpp adapter with identical text prompt/greedy decoding and extracted cue IDs; (b) the chosen first-pass production interface; (c) verify/shorten passes. The current training answer is numbered text, while production requires JSON plus `scene_summary`. Either retain a text-output translation path, or train/evaluate a compatible JSON target as a **separate format ablation**; grammar constraints alone do not teach that missing task. Then test larger cue batches—do not assume an adapter trained at eight works at 25.

Record parse completeness, Arabic preservation, English semantic quality and latency **before and after** verification/shortening. This can reveal that a model gain is being erased by a downstream pass. Freeze reasoning settings, sampler/DRY settings and GGUF quantization during model comparisons. Q4_K_M inference remains the release condition even when NF4 HF evaluation looks better.

## Recommended next experiments

Ranked by expected information/quality value per 3070 hour. Effects are hypotheses; no source supports a promised numerical gain on this dataset.

1. **Establish an honest baseline and locate the bottleneck.** Exact change: retain E2 as dev; create a new independently reviewed **600–1,000-line / ≥3-episode** test; compare saved round-2/3 outputs with paired chrF++; run raw-versus-reviewed-Arabic diagnostic on **100 recoverable-noise dev lines**, same model/prompt/decoder. **Expected effect:** no automatic score gain; distinguishes sampling/style effects, ASR headroom and true training failure. **3070 cost:** zero for saved-output chrF; about **8 minutes** for two 100-line HF diagnostic passes by linear extrapolation, plus sequential COMET; reference review is the dominant human cost. **Success:** quantified paired CI, an error breakdown, and a frozen test that has not supplied training idioms.

2. **Tune vanilla LoRA optimization, keeping the known 8B fit.** Exact changes, sequentially: seeded reproduction at **1e-4 constant / batch 1 / one epoch**; then **5% warmup + cosine** at the same settings; then **LR 5e-5**; then **accumulation 8**. Rerun the most promising setting from the original base with a **two-epoch schedule**, retaining .5/1/2 checkpoints; r16/α32/.05 and all seven modules unchanged. Optionally establish Unsloth/TRL parity before this sequence, with its longest-example backward/memory gate. **Expected effect:** more stable updates and a measurable stopping point; gain unknown, and accumulation may underfit. **3070 cost:** roughly **46 minutes per one-epoch trial**, 91 minutes for a two-epoch run; plan **3–5 GPU hours plus selective dev generation**, then three-seed confirmation only for finalists. **Success:** better dev meaning/correction metrics and a positive final paired effect without extra hallucinations; failed runtime parity leaves the working loop as the trainer.

3. **Use real ASR corrections, then targeted idioms, at fixed exposure.** Exact change A: replace **25% of eligible corrected-source window presentations** with reviewed clean-source views; keep the remainder raw and all targets unchanged. Exact follow-up B: separately replace **10% of window presentations** with **100–200 reviewed idiom/name examples** and literal counterexamples; preserve held-out idiom contexts. Do not add synthetic noise until A is measured; its next comparison would replace 20% of presentations with empirical corruptions. **Expected effect:** better recovery and fewer literal idiom translations, with less clean-input overcorrection; no guaranteed corpus gain. **3070 cost:** one selected SFT run per treatment, approximately **46–91 minutes** each, unchanged VRAM class; annotation/verification costs more than noise generation. **Success:** improvement on unseen recoverable-noise/idiom subsets, stable clean-input correction precision and no new invented content.

4. **Screen Arabic-centric bases before paying to train them.** Exact change: run **ALLaM-7B-Instruct-preview** and **Command R7B Arabic 02-2025** on the same **200-line dev subset** as Qwen3-8B with/without its best adapter and production Qwen3.5-9B, using the same numbered ar+en task and controlled decoding. Add **Jais-2-8B-Chat** only after a GGUF/context/memory smoke test; Gemma-3-4B-it is the cheap fallback candidate. Train only the strongest new base, using the selected recipe with architecture-appropriate targets. **Expected effect:** tests the direct Levantine benchmark signal; avoids an unsupported assumption that Arabic leaderboard rank predicts drama quality. **3070 cost:** two new-model HF passes extrapolate to **~17 minutes** at the current model's rate, but actual speeds/fit differ; GGUF screening should be timed, not forecast. Add roughly one **1–2-epoch training budget** for a winner after a memory probe. **Success:** blinded meaning/idiom wins and competitive latency, followed by adapter improvement on the frozen test; keep Qwen3.5 inference unchanged until that evidence exists.

5. **Pilot CPO on reviewed, substantive errors after SFT stabilizes.** Exact change: **200–500** same-prompt chosen/rejected pairs from training-only model errors/editor corrections; CPO **β=.1, NLL weight=1, LR 5e-6, warmup 5%/cosine, effective batch 4, one epoch**, 1–4 target cues per example. Compare against additional SFT on the same chosen examples with identical context/token budget. **Expected effect:** suppress specific hallucination/negation/name/idiom errors; data scale is below the strong published precedents, so improvement is unverified. **3070 cost:** allow **30–90 minutes training** for short examples as a planning budget, plus rejected-output generation and review; require a pairwise-backward memory probe or use sequential forwards. **Success:** fewer substantive errors than matched SFT, stable copy/correction behavior, and a positive independently tested paired effect rather than merely stronger preference for the annotator's wording.
