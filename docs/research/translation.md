# Translation research: Levantine (Syrian) Arabic → English subtitles

*Research date: 2026-10-27/28. Compiled from five research sub-reports: dedicated MT models, subtitle-LLM prior art, open-weight local LLMs, API pricing, and dialect-MT benchmarks. The final synthesis agent was cut off by rate limits, so this file was assembled from their returned findings.*

Markers: **[LEV]** = Levantine-specific, **[MSA/gen]** = MSA or general, **UNVERIFIED** = not confirmed from a primary source.

The decision and the prompt/schema design live in [../DESIGN.md §5.5–5.6](../DESIGN.md#55-translation-rust--llm).

---

## 1. Bottom line
- **Use a cascade:** ASR to an Arabic transcript, then an LLM that both corrects the Arabic and translates it to English.
  - The owner wants both tracks anyway.
  - On the one fine-tuned comparison, end-to-end speech translation was only about 0.7 BLEU better (IWSLT apc→en, KIT: 27.3 vs 26.6).
  - The best IWSLT 2025 apc→en systems were cascades.
- **Frontier LLMs are the best-evidenced Levantine→English translators.**
  - GPT-4 had the best COMET on Jordanian and Palestinian **TV-series** dialogue (TARJAMAT).
  - GPT-4o led on Syrian MADAR/FLORES (AL-QASIDA).
  - GPT-4o and Command R+ led on Lebanese subtitles and podcasts.
  - **No published eval covers 2025–26 frontier models (Claude 5.x, GPT-6, Gemini 4) on Levantine→English.** Our own edits will be the benchmark.
- **On authentic speech, quality drops a lot:** GPT-4o went from 0.93 xCOMET on translated-from-English Lebanese to 0.79 on authentic podcast speech. Benchmarks built from translated text likely overstate quality on drama.
- **LLM correction of Arabic ASR can make things worse.** In one Interspeech 2024 study, fine-tuned LLM correction raised Arabic CER for Whisper v2 (5.33 → 6.79). Correct conservatively: flag uncertainty and guard against over-correction.
- **Correcting the transcript and the translation together beats refining the translation alone** (Dou et al. 2025). That supports emitting the corrected Arabic and the English in one call per cue ID.

## 2. Benchmarks (dialect → English)

**TARJAMAT** ([ArabicNLP 2023](https://aclanthology.org/2023.arabicnlp-1.6), Table 3)
- Dialect sets are TV-series transcripts from YouTube; the MSA set is news. Scores are BLEU / COMET.

| System | MSA | JOR [LEV] | PAL [LEV] |
|---|---|---|---|
| GPT-4 0-shot | 43.18 / 86.32 | 24.57 / 76.37 | **31.97 / 76.62** |
| GPT-4 5-shot | 43.66 / 86.22 | 25.26 / **76.50** | 30.48 / 76.19 |
| ChatGPT 5-shot | 44.81 / 86.24 | 26.82 / 75.27 | 31.56 / 75.51 |
| Google Translate | **66.01 / 87.23** | **29.35** / 71.83 | 25.78 / 68.69 |
| NLLB-1.3B-distilled | 41.34 / 84.76 | 13.07 / 60.25 | 14.98 / 60.57 |

**AL-QASIDA** ([Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1137); numbers from the repo CSVs). spBLEU / chrF:

| Model | MADAR Syria→EN [LEV] | FLORES apc→EN [LEV] |
|---|---|---|
| GPT-4o | **49.6 / 64.6** | **42.5 / 62.9** |
| Command R+ | 46.3 / 62.3 | 41.9 / 62.0 |
| SILMA-9B | 33.9 / 51.4 | 34.2 / 58.2 |
| AceGPT-7B | 16.4 / 42.1 | 27.9 / 54.3 |

**Lebanese→English** ([AbjadNLP 2025](https://aclanthology.org/2025.abjadnlp-1.13), xCOMET)
- GPT-4o: 0.934 on OpenSubtitles-derived data, 0.793 on authentic podcast speech.
- Command R+: 0.927 / **0.810**.
- Google Translate: 0.929 / 0.665.

**AraDiCE** ([COLING 2025](https://aclanthology.org/2025.coling-main.283)), MADAR Levantine→EN BLEU [LEV]: Jais-13B 36.1, AceGPT-13B 35.3, Llama-3-8B 23.0, Mistral-7B 12.2.

**IWSLT 2025 apc speech→EN** ([Findings](https://aclanthology.org/2025.iwslt-1.44), Table 34) [LEV], Syrian/Levantine spontaneous speech:
- KIT: 23.34 BLEU / 0.704 COMET.
- LIA: 22.56 / **0.719**. Its cascade was a Levantine fine-tuned Whisper-large-v3, then NLLB-1.3B fine-tuned on UFAL+Levanti, then BLASER rerank.

**NADI 2026 spoken translation** ([arXiv 2609.27086](https://arxiv.org/abs/2609.27086)), Casablanca TV episodes:
- Best overall: 14.19 BLEU (Jordan 27.46, Palestine 20.32).
- Whisper-large-v3 baseline: Jordan 21.6, Palestine 0.35.

**Leaderboards:**
- No leaderboard isolates Levantine→English.
- SILMA ABB's "Translation (incl. dialects)" category is mixed-direction, only 36 items, and judged by an LLM (GPT-4.1). The top 20 models sit within about 0.7/10 of each other.
- Palm ([ACL 2025](https://aclanthology.org/2025.acl-long.1579)), human eval of Syrian-dialect responses: Claude-3.5-Sonnet scored highest (4.65). That is a generation task, not translation.

## 3. Dedicated MT models
| Model | Levantine? | License | Notes |
|---|---|---|---|
| NLLB-200 (600M–54B) | `apc_Arab`, `ajp_Arab` | **CC-BY-NC** | FLORES apc→en chrF++: 64.9 (MoE) to 59.7 (600M). FLORES is Wikipedia prose, so expect a large drop on drama. |
| SeamlessM4T v2 | arb/ary/arz only | CC-BY-NC | — |
| MADLAD-400 | no Levantine tag | Apache | — |
| `Helsinki-NLP/opus-mt-tc-big-ar-en` | general ar | CC-BY | FLORES ara-eng BLEU 42.6 [MSA] |
| `guymorlan/levanti_translate_ar_en` | Syrian/Leb/Pal/Jor | CC-BY-NC | Fine-tuned opus-mt on Levanti; no published scores |

Verdict: these can't correct the Arabic and can't follow a style guide, and most carry NC licenses. Not used.

**Parallel data, useful later for fine-tuning or evaluation:**
- [UFAL North Levantine](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5033): 120.6k OpenSubtitles lines translated into apc. **Subtitle domain.** CC BY-NC-SA.
- [Levanti](https://huggingface.co/datasets/guymorlan/levanti): 42k real + 466k synthetic pairs, with a Syrian label. CC-BY-NC.
- [MADAR](https://camel.abudhabi.nyu.edu/madar-parallel-corpus): Damascus, Aleppo and Beirut travel phrases. Research-only.

## 4. Local open-weight LLMs on an 8 GB RTX 3070
Weights are GGUF file sizes; the KV cache is computed from the configs. The budget assumes about 7.5 GB usable.

| Model | Q4_K_M weights | Arabic/dialect claim | Fits alone / next to 1.5 GB ASR | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` (Apache-2.0) | 5.68 GB | 201 languages; Qwen3 explicitly lists **Levantine** | ✓ / ✗ | **Chosen fallback.** Can do correction and translation. Pass `enable_thinking: false`. |
| `Qwen/Qwen3.5-4B` | 2.74 GB | same | ✓ / ✓ | Lighter fallback |
| `tencent/HY-MT1.5-7B` | 4.62 GB | Arabic among 33 | ✓ / tight | Translation-only. License excludes EU/UK/KR. |
| `ByteDance-Seed/Seed-X-PPO-7B` | 4.56 GB | Arabic among 28 | ✓ / tight | Translation-only |
| `QCRI/Fanar-1-9B-Instruct` (Apache) | 5.38 GB | **Gulf, Levantine, Egyptian** | ✓ (KV q8) / ✗ | 4k context; Levantine PIQA 59 vs MSA 68 |
| `inception42/Jais-3-7B` (Apache, Oct 2026) | 5.59 GB | MSA + dialects | ✓ / ✗ | Always reasons (more tokens); new llama.cpp architecture |
| `google/translategemma-4b-it` | 2.49 GB | ar-SY maps to plain "Arabic" | ✓ / ✓ | Translation-only; 2k input |

- **Measured speed:** Llama-3.1-8B Q4_K_M on a 3070 ran at 73.7 tok/s with a 5.4 GB peak (single run, Ollama). Other 7–9B Q4 models should be similar.
- At about 30k output tokens per episode, that works out to roughly 7 minutes per episode on the GPU lane.

## 5. API pricing (seen 2026-10-27; USD per 1M tokens in/out)
| Provider | Models (in / out) | Notes |
|---|---|---|
| Anthropic | Opus 5.5 4/20 · Sonnet 5.5 2/10 · Haiku 5.5 0.10/0.50 (≤100k prompt) · Fable 5.1 10/50 | Batch −50%; strict JSON; prompt caching. Tokenizer uses about 30% more tokens than GPT-style. |
| OpenAI | gpt-6-astra 10/50 · gpt-6.1-sol 2/10 · gpt-6.1-terra 1/5 · gpt-6-luna 0.10/0.50 | Batch/Flex −50% |
| Google | gemini-4-argon 2/10 (→4/20 on Dec 1) · gemini-3.8-flash 0.75/3.75 (→1.50/7.50 on Jan 1) | — |
| DeepSeek | deepseek-flash 0.30/1.20 (off-peak −50%) · v4-pro 1.32/3.96 | Data stored in China; only `json_object` mode |
| Qwen (Alibaba Intl) | qwen3.8-max 2/6 · qwen3.8-flash 0.15/0.47 | Strict schema not yet available in the Singapore region |

**Arabic tokens per word** (measured on 5,000 Shami-corpus lines): Gemma 1.89, o200k 1.96, Qwen3.5 1.97, Llama 2.18, DeepSeek 2.20. Claude's tokenizer is not public; use `count_tokens`.

**Per-episode estimate** (about 21k tokens in on Claude, about 30k out with corrected Arabic plus English, no thinking):

| Model | Cost per episode | Batch |
|---|---|---|
| Opus 5.5 | about $0.68 | about $0.34 |
| Sonnet 5.5 | about $0.34 | about $0.17 |
| Haiku 5.5 | about $0.02 | — |

Thinking tokens bill as output and can add 20–100%.

**Data handling:** Anthropic keeps no content by default (Fable has a mandatory 30-day retention). OpenAI keeps 30 days. DeepSeek trains on data.

## 6. Prior art: LLM subtitle translators
| Project | Technique worth copying |
|---|---|
| [llm-subtrans](https://github.com/machinewrapped/llm-subtrans) | Scenes by time gap. Batches split at the largest gap. `#N` IDs. Rolling scene summaries returned by the model. Auto-growing terminology map. "Whispered" instructions: *the input has ASR errors, infer from context*. Profanity: "use equivalent profanity". |
| [VideoLingo](https://github.com/Huanshere/VideoLingo) | Global theme + terms pre-pass. Only terms that occur in a chunk are injected. Translate → reflect ("too wordy?") → free passes. |
| [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) | Separate ASR post-edit pass ("Zero Structural Changes"). Ellipsis bridging across cues. **Anti-pattern:** pads missing lines with blanks. |
| [Lingarr](https://github.com/lingarr-translate/lingarr) | Earlier lines sent *with their translations* (consistency, cache-friendly). JSON `{position, line}` batches. |
| [chatgpt-subtitle-translator](https://github.com/Cerlancism/chatgpt-subtitle-translator) | Batch-size step-down down to single lines on a count mismatch. Repetition-loop guard. Planning pre-pass (characters, tone). |
| [gnehs/subtitle-translator-electron](https://github.com/gnehs/subtitle-translator-electron) | Scenes at 1.5 s / 4 s gaps. A "brief" (summary, tone, characters, glossary). Hard checks: uncovered cues are errors; CPS 17/22, 42 CPL, 500 ms min are warnings. |
| [polygluttony](https://github.com/blyat-uk/polygluttony) (Rust/Tauri) | `<0001:D>` markers inside JSON; keep the valid prefix on drift; glossary across episodes. |

Research: TEaR, DelTA (proper-noun memory plus a bilingual summary, which suits series), SubER metric, and length control ([2506.04855](https://arxiv.org/abs/2506.04855)).

## 7. Guidelines used
- **Netflix English:**
  - 42 CPL, 2 lines, ≤ 20 CPS (adult).
  - Profanity at equivalent severity, never censored.
  - Names untranslated; dialogue hyphen without a space; U+2026 ellipsis.
- **Netflix Arabic:**
  - 42 CPL, 2 lines, 20 CPS. "Never censor dialogue."
  - §25: in dialect files, words pronounced differently from MSA keep the MSA spelling (آل → قال). This is the basis of DESIGN §5.6.
- **Netflix general:** min 5/6 s, max 7 s. Key-names tables keep names consistent across episodes.
