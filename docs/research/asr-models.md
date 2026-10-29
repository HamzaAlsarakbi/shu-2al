# ASR model research: Levantine (Syrian) Arabic TV drama → Arabic text with timestamps

*Researched 2026-10-27. Sources were checked live against HF model cards, the HF leaderboard Space source, arXiv and GitHub. Every number below links to its source. Numbers I computed myself are labelled **[computed]**, and estimates are labelled **[estimate]**.*

**Hardware target (decided):** inference runs on an **Arch Linux box with an RTX 3070, 8 GB VRAM** (Ampere: BF16 yes, FP8 tensor cores no, 448 GB/s). Python ML workers are fine. The **Apple M2 (24 GB) is only the dev machine.** The ranking below is driven by what fits and runs well in 8 GB.

---

## 1. TL;DR

| Rank | Model | Why | Fits 8 GB? | Expected RTF on 3070 |
|---|---|---|---|---|
| **1** | **[CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)** (2.07B, Apache-2.0) | Best measured numbers on **Levantine TV-series speech** (Casablanca Jordan 23.3 / Palestine 30.8 WER **[computed]**). Fits in BF16 with no quantization. Runs in vLLM and transformers, and through the GGML **transcribe.cpp**, which has Rust bindings. | **Yes** (BF16 weights ≈4.1 GB) | ≈0.02–0.05 at batch 1 **[estimate]** |
| **2** | **[IFM/Jais-ASR](https://huggingface.co/IFM/Jais-ASR)** (5.7B, Apache-2.0, released 2026-10-06) | Top of the leaderboard by average (22.82). Has an explicit **Syrian dialect setting (`SY`)** and an explicit **"no speech" refusal on music/noise**. Needs 4-bit to fit. | **Only quantized**: GGUF Q4_K_M + Q6_K mmproj, 5.03 GiB peak, avg WER 22.99. BF16 needs ≈12.4 GiB. | ≈0.05–0.15 at batch 1 **[estimate]** |
| **3** | **Whisper large-v3-turbo / large-v3 Levantine fine-tune**, e.g. [oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2) (Apache-2.0) | The only top option with **built-in segment and word timestamps**. Runs natively in Rust (whisper-rs / whisper.cpp) or faster-whisper. Weaker zero-shot and notorious for hallucinating on silence and music. | **Yes** easily (int8 ≈1–3 GB) | large-v3 ≈0.09–0.11 sequential, ≈0.03 batched. Turbo ≈0.02–0.04 **[estimate, anchored on faster-whisper's measured RTX 3070 Ti numbers]** |

**My recommendation:** run **Cohere Transcribe Arabic** on the 3070 inside a Python worker. Put **Silero VAD** in front of it and an **Apache-licensed Arabic CTC forced aligner** behind it for timestamps (see §6). Then A/B test it against **Jais-ASR Q4_K_M** on 1–2 hours of hand-transcribed Syrian drama before committing. Keep a Whisper fine-tune as the fallback with native timestamps, and as the cheapest model to fine-tune on the 3070 itself.

Every leading model in the dialect race (Cohere, Jais, Audar) has the same gap: **none of them emits timestamps**. Timestamping has to come from a separate stage either way.

---

## 2. Comparison table: all candidates ranked for an 8 GB RTX 3070

"Casa" is the **Casablanca** test set: TV-series dialogue from YouTube in 8 dialects, including **Jordan and Palestine**. It is the closest public match to Syrian drama. Leaderboard numbers come from the [Open Universal Arabic ASR Leaderboard source](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard/blob/main/app.py), last modified 2026-10-20. RTF = processing time ÷ audio duration, so lower is better (0.05 means 20× faster than real time).

| Model | Params | License (commercial?) | LB avg WER | Casa WER (dialect TV) | Levantine-specific evidence | Timestamps | **Fits 8 GB?** | **Expected RTF on 3070** |
|---|---|---|---|---|---|---|---|---|
| Cohere Transcribe Arabic 07-2026 | 2.07B | Apache-2.0 ✅ (click-through gate) | 24.19 | 45.68 | JOR 23.32 / PAL 30.83 **[computed]**; lahgtna Syrian 43.0 | ❌ none | ✅ BF16 | ≈0.02–0.05 **[est.]** |
| Sygma-ASR-Arabic (Cohere + 8-beam decoding) | 2.07B | Apache-2.0 ✅ | 23.01 | **44.67** | none separate | ❌ | ✅ BF16 | "several times slower" than greedy, ≈0.1–0.3 **[est.]** |
| Jais-ASR | 5.72B | Apache-2.0 ✅ | **22.82** | 46.61 | none published | ❌ | ⚠️ Q4 GGUF only (5.03 GiB). W4A16 AWQ 5.85 GB is tight. Q8 needs 7.55 GiB (no) | ≈0.05–0.15 **[est.]** |
| Audar-ASR-V1-Turbo | 2.35B | **Community licence ⚠️** (thresholds; no hosting as an ASR service) | 23.17 | 47.02 | JOR 22.58 / PAL 36.2–38.0 **[computed]** | ❌ | ✅ BF16 4.7 GB, GGUF Q8 2.8 GB | ≈0.03–0.08 **[est.]** |
| omniASR-LLM-7B (Meta) | 7.8B | Apache-2.0 ✅ | 28.32 | 56.46 | vendor-internal Levantine 74.50 (Cohere's eval) | ❌ | ❌ (~17 GiB) | — |
| omniASR-LLM-1B | ~1B | Apache-2.0 ✅ | 29.96 | 60.68 | `apc_Arab` conditioning possible but untested | ❌ | ✅ (~6 GiB) | ≈0.15–0.3 **[est.]** |
| Qwen3-Omni-30B-A3B | 35B | Apache-2.0 ✅ | 30.71 | 62.55 | — | ❌ | ❌ | — |
| Gemma 4 E4B-it | 8.0B (with PLE embeddings) | Apache-2.0 ✅ | 32.98 | 58.63 | — | ❌ | ⚠️ only 4-bit | not measured |
| Qwen3-ASR-1.7B | 2.35B | Apache-2.0 ✅ | 33.36 | 64.47 | — | aligner has **no Arabic** | ✅ BF16 4.7 GB | ≈0.02–0.05 **[est.]** |
| Voxtral-Small-24B | 24B | Apache-2.0 ✅ | 34.47 | 66.30 | — | — | ❌ | — |
| Nemotron 3.5 ASR streaming 0.6B (NVIDIA) | 0.64B | OpenMDW-1.1 ✅ (permissive) | 35.86 | 69.37 | — | ✅ token-level | ✅ | <0.01 **[est.]** |
| Whisper large-v3 | 1.54B | Apache-2.0/MIT ✅ | 36.86 | 71.81 | JOR 48.4 / PAL 58.0 (Casablanca paper, zero-shot) | ✅ segment + word | ✅ int8 2.9 GB / fp16 4.5 GB | ≈0.09–0.11 seq, ≈0.03 batched **[est. from 3070 Ti measurement]** |
| Whisper large-v3-turbo | 0.81B | MIT ✅ | 40.05 | 73.79 | — | ✅ | ✅ | ≈0.02–0.04 **[est.]** |
| Whisper Levantine fine-tunes (oddadmix v2) | 0.81–1.54B | Apache-2.0 ✅ (training-data licence unclear) | not on LB | not on LB | lahgtna Syrian **27.2** (large-v3) / **29.1** (turbo), in-domain | ✅ | ✅ | same as base Whisper |
| SeamlessM4T-v2-large | 2.3B | **CC-BY-NC-4.0 ❌** | 38.16 | 66.25 | JOR 47.9 / PAL 56.8 (Casablanca paper) | — | ✅ | not measured |
| MMS-1b-all | 0.96B | **CC-BY-NC-4.0 ❌** | 54.54 | 87.95 | JOR 81.5 / PAL 83.1 | CTC alignable | ✅ | — |
| VibeVoice-ASR (Microsoft) | 8.7B | MIT ✅ | 52.99 | 93.37 | — | ✅ (diarized) | ❌ | — |
| Voxtral-Mini-3B-2507 | 4.7B | Apache-2.0 ✅ | 42.58 | 77.52 | — | — | ❌ in BF16 | — |

**How I estimated RTF.** No published benchmark covers these models on a plain RTX 3070. The Whisper figures scale [faster-whisper's measured RTX 3070 Ti 8 GB results](https://github.com/SYSTRAN/faster-whisper): large-v2, 13 min of audio, fp16 beam 5 took 63 s (RTF 0.081); batch 8 took 17 s (RTF 0.022); int8 took 59 s. large-v3 has the same compute profile. I applied the memory-bandwidth ratio (3070 at 448 GB/s vs 3070 Ti at 608 GB/s). For Cohere, the anchors are transcribe.cpp's **measured 77× real-time (RTF 0.013) on an M4 Max via Metal at Q8** ([doc](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/cohere-transcribe-arabic-07-2026.md)) and a **measured RTF of 0.18–0.22 for a Cohere-Arabic fine-tune on a T4 with 1 CPU core and short clips** ([Rlamas card](https://huggingface.co/Rlamas/Cohere-Jordanian-Dialect)). For the LLM-decoder models (Jais, Audar, Qwen3-ASR) I used bandwidth-bound decoding arithmetic, not measurements. **Benchmark on the real box before trusting any of these.**

---

## 3. Benchmarks

### 3.1 Open Universal Arabic ASR Leaderboard (snapshot 2026-10-20)

Source: [Space app.py](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard/blob/main/app.py), [GitHub](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard), [paper arXiv:2412.13788](https://arxiv.org/abs/2412.13788). WER % (CER % for Casablanca).

**What each column measures:**
- **CV** (Common Voice 18) is mostly MSA read speech.
- **MGB-2** (Al Jazeera broadcast) is predominantly MSA.
- **MASC** is YouTube speech of mixed genre and dialect. Its test sets lean MSA, and Jais's card labels them as MSA lectures.
- **SADA** is Saudi dialect.
- **Casablanca** is the only dialectal TV-series set.

**Only the Casablanca column is evidence about drama dialogue.**

| Model | Avg WER | SADA | CV (MSA) | MASC clean | MASC noisy | MGB-2 (MSA) | **Casablanca WER / CER** |
|---|---|---|---|---|---|---|---|
| IFM/Jais-ASR | **22.82** | 30.52 | 8.59 | 16.14 | 25.47 | **9.60** | 46.61 / 17.92 |
| abedbanna/Sygma-ASR-Arabic | 23.01 | 33.63 | **5.24** | 15.92 | 25.10 | 13.48 | **44.67 / 17.42** |
| audarai/Audar-ASR-V1-Turbo | 23.17 | **28.92** | 8.09 | 16.73 | 27.19 | 11.08 | 47.02 / 18.75 |
| CohereLabs/cohere-transcribe-arabic-07-2026 | 24.19 | 36.83 | 5.27 | 16.35 | 25.65 | 15.33 | 45.68 / 20.00 |
| omniASR_LLM_7B | 28.32 | 41.61 | 8.75 | 19.69 | 29.29 | 14.13 | 56.46 / 23.96 |
| omniASR_LLM_1B | 29.96 | 43.84 | 9.55 | 20.03 | 30.26 | 15.34 | 60.68 / 28.02 |
| cohere-transcribe-03-2026 (multilingual base) | 30.67 | 60.11 | 8.17 | **8.66** | **19.01** | 25.33 | 62.71 / 30.31 |
| Qwen3-Omni-30B-A3B | 30.71 | 44.82 | 11.46 | 21.47 | 30.85 | 13.09 | 62.55 / 28.53 |
| Audar-ASR-V1-Flash (0.78B) | 32.04 | 44.36 | 15.38 | 22.56 | 34.19 | 17.05 | 58.71 / 25.97 |
| NVIDIA conformer-ctc-large-arabic (+LM) | 32.91 | 44.52 | 8.80 | 23.74 | 34.29 | 17.20 | 68.90 / 32.97 |
| gemma-4-E4B-it | 32.98 | 43.40 | 19.65 | 24.86 | 33.59 | 17.72 | 58.63 / 25.11 |
| Qwen3-ASR-1.7B | 33.36 | 45.53 | 16.90 | 24.37 | 34.29 | 16.57 | 64.47 / 26.23 |
| Voxtral-Small-24B | 34.47 | 50.82 | 15.25 | 23.96 | 34.43 | 16.03 | 66.30 / 30.64 |
| nemotron-3.5-asr-streaming-0.6b | 35.86 | 53.06 | 15.05 | 23.62 | 35.24 | 18.84 | 69.37 / 32.18 |
| whisper-large-v3 | 36.86 | 55.96 | 17.83 | 24.66 | 34.63 | 16.26 | 71.81 / 35.04 |
| seamless-m4t-v2-large | 38.16 | 62.52 | 21.70 | 25.04 | 33.24 | 20.23 | 66.25 / 29.85 |
| whisper-large-v3-turbo | 40.05 | 60.36 | 25.73 | 25.51 | 37.16 | 17.75 | 73.79 / 34.83 |
| Qwen3-ASR-0.6B | 42.19 | 53.75 | 28.28 | 31.34 | 42.63 | 25.45 | 71.68 / 31.06 |
| Voxtral-Mini-3B | 42.58 | 63.65 | 22.12 | 28.37 | 41.27 | 22.56 | 77.52 / 39.43 |
| VibeVoice-ASR | 52.99 | 69.83 | 44.25 | 32.95 | 52.43 | 25.10 | 93.37 / 55.89 |
| mms-1b-all | 54.54 | 77.48 | 26.52 | 38.82 | 57.33 | 39.16 | 87.95 / 40.41 |

Notes:
- The leaderboard re-ran its normalization in 2026. Cohere's own card still shows its July snapshot (avg 25.87, Casablanca 49.71), while the current board shows 24.19 / 45.68 ([Cohere card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026), [Audar card note on "updated normalization"](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)).
- omniASR models were run with `lang=['arb_Arab']` (MSA) for everything ([leaderboard inference script](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/blob/main/models/omnilingual_asr.py)). Their **Levantine `apc_Arab` conditioning was never tested** there.
- **Closed APIs** (not on HF, quality reference only), measured by IFM with the leaderboard protocol ([Jais-ASR card](https://huggingface.co/IFM/Jais-ASR)):
  - Speechmatics **Melia 1**: avg 23.51, Casablanca 46.41 / 20.41.
  - ElevenLabs **Scribe v2**: avg 24.54, Casablanca 50.20 / 21.18.
  - Open models are now at parity with these commercial APIs on this board.

### 3.2 Levantine-specific numbers

Arabic WER is inflated by spelling variation in dialect (e.g. انتو/انتوا). Treat **CER as the more stable signal**.

**(a) Casablanca test, per country, under the leaderboard normalization [computed].** I took Sygma's published per-utterance outputs ([paper_artifacts](https://huggingface.co/abedbanna/Sygma-ASR-Arabic/tree/main/paper_artifacts)) and the leaderboard reference manifest ([casablanca_test.json](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/blob/main/datasets/casablanca_test.json)). I mapped each `seg_id` to its country using [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) and scored with the leaderboard's `normalize_arabic_text` plus corpus-level WER/CER. As a sanity check, my overall numbers reproduce Sygma's published totals exactly (Cohere greedy 45.70, Audar with loop fix 50.18).

| System (as run by Sygma authors) | **Jordan** WER / CER (n=848) | **Palestine** WER / CER (n=667) | Egypt | UAE | All 8 |
|---|---|---|---|---|---|
| Cohere Transcribe Arabic, greedy | 23.32 / 7.97 | **30.83 / 12.31** | 26.77 | 35.15 | 45.70 |
| Audar-ASR-V1-Turbo, transformers recipe, no loop guard | 22.58 / 7.02 | 38.01 / 15.85 | 32.97 | 39.33 | 56.05 |
| Audar-ASR-V1-Turbo + Sygma loop guard | **22.58 / 7.02** | 36.20 / 12.79 | 32.03 | 33.94 | 50.18 |

These are not official leaderboard numbers. Audar's official Casablanca score (47.02) is better than Sygma's re-run. I could not compute Jais-ASR per country because no per-utterance outputs are published.

**(b) Casablanca paper, zero-shot, per country** ([arXiv:2410.04527](https://arxiv.org/abs/2410.04527), Table 3, WER / CER). This uses the paper's own normalization. The "+pre" column is the closest to the leaderboard's normalization.

| Model | Jordan −pre | Jordan +pre | Palestine −pre | Palestine +pre |
|---|---|---|---|---|
| Whisper large-v3 | 48.44 / 16.18 | 39.68 / 13.47 | 58.02 / 21.05 | 50.20 / 18.38 |
| Whisper large-v2 | 49.47 / 16.34 | 41.13 / 13.64 | 57.06 / 20.02 | 48.64 / 17.24 |
| SeamlessM4T-v2 | 47.94 / 15.84 | 39.24 / 13.12 | 56.78 / 20.74 | 48.92 / 18.13 |
| MMS-1b-all | 81.46 / 33.02 | 78.54 / 31.03 | 83.14 / 33.07 | 80.18 / 30.82 |

The comparison is rough because the normalizations differ. Even so, Cohere roughly **halves Whisper-v3's WER on Jordan and Palestine TV dialogue** (≈23 vs ≈40 on Jordan, ≈31 vs ≈50 on Palestine).

**(c) NADI 2025 Subtask 2** (Casablanca-derived; JOR and PAL test sets) ([arXiv:2509.02038](https://arxiv.org/abs/2509.02038), Table 4, WER / CER):

| System | Jordanian | Palestinian | Overall (8 dialects) |
|---|---|---|---|
| Munsit (CNTXT, **closed**; 121M Conformer, 15k h weak labels + Casablanca FT) | 20.68 / 5.64 | 22.27 / 8.05 | 35.68 / 12.20 |
| ELYADATA-LIA (SeamlessM4T-v2 FT per dialect) | 28.03 / 9.36 | 27.36 / 10.20 | 38.53 / 14.52 |
| BYZÖ-Whisper (Whisper-large-v3 FT) | 28.84 / 9.47 | 27.65 / 10.59 | 39.78 / 14.75 |
| Baseline Whisper-large-v3 zero-shot | 46.09 / 19.28 | **100.76 / 77.92** | 93.89 / 72.79 |

The zero-shot Whisper Palestinian WER above 100% shows its failure modes (looping and translating instead of transcribing). It also shows that **in-domain fine-tuning brings Whisper to about 28 WER**.

**(d) NADI 2026** ([arXiv:2609.27086](https://arxiv.org/abs/2609.27086)). It ran Robust ASR (low-bandwidth Casablanca), Mixed-dialect and Code-switched tracks. The best results were 43.24 WER (robust) and 44.22 WER (mixed), against a Whisper-large-v3 baseline of 81.37 and 57.91. The runner-up on the robust track fully fine-tuned cohere-transcribe-arabic-07-2026. **I did not find a per-country Levantine breakdown** in the parts of the paper I read.

**(e) Syrian, Lebanese and Palestinian on the lahgtna-v3-small balanced test** (200 clips per dialect). These are model-card numbers from oddadmix. The fine-tunes are **in-domain** (trained on the same corpus and transcription conventions); Cohere base is zero-shot.

| Model | Syrian | Palestinian | Lebanese | Overall (13 dialects) |
|---|---|---|---|---|
| whisper-large-v3-arabic-dialectal-v2 (FT) | **27.2** / 9.6 | 30.1 / 13.7 | 31.7 / 11.3 | 32.0 / 12.0 |
| whisper-large-v3-turbo-arabic-dialectal-v2 (FT) | 29.1 / 10.5 | 30.8 / 13.5 | 34.8 / 13.2 | 33.2 / 12.5 |
| cohere-transcribe-arabic-07-2026, **base, zero-shot** | 43.0 | 41.4 | 51.8 | 47.0 / 17.9 |
| cohere …-dialectal-v2 (FT; regressed) | 56.3 | 37.6 | 50.1 | 51.4 / 29.4 |

Sources: [whisper-large-v3 v2](https://huggingface.co/oddadmix/whisper-large-v3-arabic-dialectal-v2), [turbo v2](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2), [cohere v2](https://huggingface.co/oddadmix/cohere-transcribe-arabic-07-2026-dialectal-v2). The takeaway: a Whisper fine-tuned on matching Levantine data can beat the best zero-shot model **on that data's distribution**. Whether that carries over to Syrian drama is unknown.

**(f) Other Levantine figures (weaker evidence):**
- Cohere's internal dialect eval, Levantine group ([HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)): Cohere Arabic 39.78, Cohere Transcribe base 50.98, Whisper large-v3 58.64, omniASR-LLM-7B 74.50. Vendor eval on an undisclosed test set.
- [Rlamas/Cohere-Jordanian-Dialect](https://huggingface.co/Rlamas/Cohere-Jordanian-Dialect): full fine-tune of Cohere Arabic took Jordanian WER from 35.0 to **18.99** (CER 15.6 → 7.83) on an out-of-domain held-out set. Training ran in **1 h 28 min on one L40S**.
- [HebArabNlpProject/WhisperLevantine](https://huggingface.co/HebArabNlpProject/WhisperLevantine): Whisper-large-v3 fine-tuned on about 1,200 h, mostly Israeli Levantine (8 kHz phone-like audio). Reports **33% WER on an undescribed test set**.

**(g) Not found or not applicable:**
- **No public Syrian TV-drama test set exists.** Casablanca has no Syria subset.
- I found no 2025–26 results for the new models on MGB-3 (Egyptian anyway) or MGB-5.
- I found no published Levantine numbers for Gemini, gpt-4o-transcribe, Google Chirp or Azure. My web-search budget ran out before a dedicated search, so treat this as not checked rather than "no data exists".

### 3.3 MSA-only reference points (FLEURS ar, read speech; not dialect)

- Cohere Arabic: 11.0 WER (transcribe.cpp BF16 vs reference 11.00).
- Nemotron 3.5: 12.03 WER at 1.12 s chunks.
- Voxtral Mini Transcribe 2.0: 13.54 WER.
- Voxtral Realtime: 22.53 WER at 480 ms delay.

Sources: [transcribe.cpp doc](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/cohere-transcribe-arabic-07-2026.md), [Nemotron card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), [Voxtral Realtime card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

---

## 4. Candidate notes by family

### General multilingual: Whisper
- **large-v3** (1.54B) and **large-v3-turbo** (0.81B, 4 decoder layers). Both have a single `ar` language token; there are no dialect tokens.
- **Accuracy:** dialect TV dialogue is clearly behind the Arabic-specialized models (Casablanca 71.8 / 73.8).
- **Strengths:**
  - The most mature ecosystem: faster-whisper/CTranslate2, whisper.cpp/GGML (→ **whisper-rs** in Rust), MLX, candle, ONNX, transformers.
  - Native segment timestamps, plus word timestamps (cross-attention DTW).
- **Weaknesses:**
  - Severe **hallucination on silence and music**. In Arabic it famously outputs subtitle credits such as «ترجمة نانسي قنقر» ([openai/whisper#2608](https://github.com/openai/whisper/discussions/2608); analysis in [arXiv:2501.11378](https://arxiv.org/abs/2501.11378)).
  - It also loops, and sometimes translates to MSA or English.
  - Mitigations: VAD gating, `condition_on_previous_text=False`, no_speech and compression-ratio thresholds. The README's planned "Word filter" module is exactly what this needs.
- **Output:** Arabic script, inconsistent punctuation.
- **Levantine fine-tunes:**
  - [oddadmix dialectal v2 family](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2): Apache-2.0, trained on lahgtna-v3-small, whose own licence is unstated.
  - [HebArabNlpProject/WhisperLevantine](https://huggingface.co/HebArabNlpProject/WhisperLevantine): Apache-2.0, Israeli Levantine, ships a faster-whisper conversion with word timestamps.
  - [oddadmix/whisper-large-v3-turbo-arabic-dialectal (v1)](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal): 34.4 WER on lahgtna v2 test vs 59.0 zero-shot. Training data includes Lebanese and Syrian.
  - Fine-tunes keep Whisper's architecture, so whisper.cpp and faster-whisper conversions work.

### Meta: Omnilingual ASR, MMS, Seamless
- **[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)** (Nov 2025; v2 checkpoints and "LLM Unlimited" long-audio variants added Dec 2025). Apache-2.0. CTC and LLM variants at 300M/1B/3B/7B.
  - **Language codes include `apc_Arab` (Levantine), but not `ajp_Arab`** (I checked [`lang_ids.py`](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)). Other Arabic codes: arb, arz, ary, arq, aeb, acm, afb, apd, ars, ayl, ayp, acw, aec, ssi.
  - Non-Unlimited models accept **audio under 40 s only**. Inference runs through the fairseq2 pipeline, not transformers. No timestamps.
  - VRAM (A100, BF16): CTC-1B ≈3 GiB, LLM-1B ≈6 GiB, LLM-3B ≈10 GiB, LLM-7B ≈17 GiB. **Only ≤1B LLM and ≤1B CTC fit comfortably in 8 GB.**
  - Most useful here as an **Apache-licensed CTC model for forced alignment**, or as a fine-tuning base.
- **[MMS-1b-all](https://huggingface.co/facebook/mms-1b-all)**: CC-BY-NC-4.0. Its only Arabic adapter is generic **`ara`** (I checked the repo files); no apc or ajp. Weak on dialect (Casablanca 87.95).
- **[SeamlessM4T-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large)**: CC-BY-NC-4.0 (non-commercial). Casablanca 66.25. I did not verify its exact Arabic variety codes this session.

### NVIDIA NeMo
- **[nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)** (May 2026, OpenMDW-1.1): Arabic is "transcription-ready", with punctuation, token timestamps, GGUF and transformers support. Weak on dialect (Casablanca 69.37).
- **[stt_ar_fastconformer_hybrid_large_pcd_v1.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)**: 115M, CC-BY-4.0. Outputs **diacritized** Arabic with `. ، ؟`. MSA-oriented, but it is a commercial-OK **CTC model for the NeMo Forced Aligner**.
- Canary and Parakeet-v3 are European-language only. Parakeet-CTC-1.1B universal and concat score Casablanca 81.3 / 80.8.

### Qwen
- **[Qwen3-ASR-1.7B / 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** (Jan 2026, Apache-2.0): Arabic supported, no Arabic dialect handling, Casablanca 64.47.
- **[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) does not support Arabic** (11 languages only).
- Qwen3-Omni-30B-A3B is better (Casablanca 62.55) but far too big for 8 GB.

### Mistral Voxtral
- Voxtral Mini 3B (4.7B params) and Small 24B, both Apache-2.0. Casablanca 77.52 and 66.30.
- **[Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)** lists Arabic (FLEURS 22.5 WER) and needs ≥16 GB in vLLM.
- Nothing here fits 8 GB without quantization, and nothing is competitive on dialect.

### Google Gemma 4 / Microsoft VibeVoice
- **[gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)** (Apache-2.0) is a general any-to-any model with Casablanca 58.63. It is surprisingly good on dialect, but you prompt it rather than run it as a dedicated ASR model, and it needs 4-bit to fit.
- **[VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)** (MIT, 8.7B) has built-in diarization and timestamps, but is poor on Arabic (Casablanca 93.37).

### Arabic-specialized (the real contenders)
- **[Cohere Transcribe Arabic 07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)** (July 2026). A 2B FastConformer encoder with a light Transformer decoder.
  - Licence: Apache-2.0, behind a click-through gate. Language must be set (`ar`/`en`); there is no auto language ID.
  - **No timestamps, no diarization.**
  - The card warns it "tends to transcribe non-speech sounds" and recommends a noise gate or VAD.
  - Runtimes:
    - transformers ≥5.4 (`CohereAsrForConditionalGeneration`); vLLM 0.19 (`/v1/audio/transcriptions`)
    - **[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)**: GGML, MIT, CUDA/Metal/Vulkan, **official Rust bindings**, Silero VAD, 400 s max input per call. GGUF builds: [handy-computer GGUF](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf), Q8_0 2.41 GB, Q4_K_M 1.56 GB, no measurable WER loss on FLEURS.
    - Community MLX (4/6/8-bit) and ONNX ports.
  - Output: Arabic script **with punctuation** (e.g. «وش بيع؟» in the published outputs).
- **[Sygma-ASR-Arabic](https://huggingface.co/abedbanna/Sygma-ASR-Arabic)**: not a new model. It is 8-best beam search with a word-count bonus, a loop detector, and chunking for clips over 35 s, all on top of Cohere. It gives −1.1 avg WER and is best on Casablanca (44.67), at several times the cost. Its card notes rare inherited subtitle-style hallucinations such as «اشتركوا بالقناة». Worth reusing: the **loop-breakdown detector**, which is cheap.
- **[Jais-ASR](https://huggingface.co/IFM/Jais-ASR)** (IFM/MBZUAI, 2026-10-06). A Qwen3-Omni-style audio encoder plus a Jais-3-4B decoder, 5.7B total, Apache-2.0.
  - 19 dialect codes, **including SY, LB, JO, PS**, with automatic language/dialect ID. Optional diacritics and code-switch spans.
  - **`[IFM-NoTranscript]` refusal for music and noise.**
  - Long audio is chunked internally.
  - Quantized builds:

    | Build | Peak VRAM or size | Avg WER |
    |---|---|---|
    | BF16 (original) | 12.42 GiB | 22.85 |
    | GGUF Q8_0 + F16 mmproj | 7.55 GiB | 22.83 |
    | **GGUF Q4_K_M + Q6_K mmproj** | **5.03 GiB** | **22.99** |
    | W4A16 AWQ | 5.85 GB | 24.02 |
    | FP8 | 7.68 GB (no FP8 hardware on Ampere anyway) | 22.91 |
    | MLX 4-bit | 4.5 GB | 22.94 |
    | MLX 8-bit | 7.0 GB | 22.76 |

    Sources: [GGUF](https://huggingface.co/IFM/Jais-ASR-GGUF), [AWQ](https://huggingface.co/IFM/Jais-ASR-W4A16-AWQ), [FP8](https://huggingface.co/IFM/Jais-ASR-FP8), [MLX 4-bit](https://huggingface.co/IFM/Jais-ASR-MLX-4bit), [MLX 8-bit](https://huggingface.co/IFM/Jais-ASR-MLX-8bit).
  - GGUF needs the **MBZUAI llama.cpp fork plus a patch**; MLX needs a patched mlx-vlm. vLLM serving uses a shipped plugin.
  - No timestamps. Only three weeks old: custom code, little community testing so far.
- **[Audar-ASR-V1-Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)** (2.35B). Its parameter count and architecture are identical to Qwen3-ASR-1.7B, so it appears to be a fine-tune of it.
  - Very strong on Gulf; good on Jordan, weaker on Palestine in my per-country computation.
  - Runtimes: GGUF/llama.cpp, vLLM, transformers.
  - **Licence:** the [AudarAI Community License](https://www.audarai.com/license/audarai-community-license-v1.0/) is commercial only under <50 staff / <$2M revenue / <100k MAU thresholds. It also **forbids hosting it as a general ASR service** and generally bans redistributing the original weights. That conflicts with the README's "host it online as a service" idea.
  - Without its no-speech protocol it loops on music and silence.
- **Munsit (CNTXT)**: closed API and on-prem only. Won NADI 2025 (see §3.2c).

### Closed APIs (non-HF, reference only)
- ElevenLabs Scribe v2 and Speechmatics Melia 1: numbers in §3.1.
- Cohere offers a free rate-limited API for the same Arabic model.
- I found no Levantine-specific figures for Google Chirp, Azure, Gemini or OpenAI gpt-4o-transcribe (search budget ran out; unverified either way).

---

## 5. Practical details for the top candidates

| | Cohere Transcribe Arabic | Jais-ASR | Whisper large-v3(-turbo) / Levantine FT |
|---|---|---|---|
| Licence | Apache-2.0 (accept the gate once). Commercial OK. | Apache-2.0. Commercial OK. | MIT / Apache-2.0. Fine-tune licence follows its card; training-data licences vary. |
| Size | 2.07B, 4.1 GB BF16; GGUF Q8 2.4 GB | 5.72B, 11.4 GB BF16; Q4_K_M+Q6_K 3.7 GB download | 1.54B / 0.81B; ~3.1 / 1.6 GB fp16 |
| VRAM on 3070 | BF16 fits (est. 5–6 GB peak) | Q4_K_M 5.03 GiB peak (measured by IFM); Q8 7.55 GiB: no | int8 2.9 GB, fp16 4.5 GB (large-v2 measured on 3070 Ti) |
| Speed, 3070 | est. RTF 0.02–0.05 (M4 Max Metal measured 0.013) | est. RTF 0.05–0.15 (no measurements published) | large-v3 est. 0.09–0.11 seq / ≈0.03 batched; turbo est. 0.02–0.04 |
| Speed, M2 (dev) | transcribe.cpp Metal or MLX ports; no M2 number found | MLX 4-bit (4.5 GB) / 8-bit (7 GB); no speed published | whisper.cpp / mlx-whisper; mlx ≈2× whisper.cpp ([billmill](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)); no M2 number found |
| Timestamps | **None** → VAD + forced alignment | **None** → VAD + forced alignment | **Segment + word** built in |
| Runtimes | transformers, vLLM, **transcribe.cpp (C API + Rust bindings, CUDA/Metal)**, MLX, ONNX, CoreML (base model) | transformers (custom code, pinned version), vLLM plugin, llama.cpp **fork**, mlx-vlm (patched) | everything: faster-whisper, whisper.cpp/**whisper-rs**, transformers, MLX, candle, ONNX, transcribe.cpp |
| Music/silence behaviour | Transcribes non-speech; rare «اشتركوا بالقناة»; loops on long or noisy clips → needs VAD + loop guard | Explicit refusal token on non-speech (vendor claim; untested by me) | Worst of the three: subtitle-credit hallucinations, loops → VAD essential |
| Output | Arabic script, punctuation, no diacritics | Arabic script; optional diacritics; optional Latin code-switch spans; punctuation not specified | Arabic script, uneven punctuation; may drift to MSA or English |

**Rust integration paths:**
1. **Python worker (recommended to start).** Run vLLM or transformers behind an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Cohere and Jais both document this, and Rust calls it over HTTP.
2. **In-process GGML.** transcribe.cpp's Rust bindings cover Cohere, Whisper, Qwen3-ASR and Nemotron on CUDA. Use whisper-rs for Whisper only.
3. Jais is only available via its patched llama.cpp server (HTTP) or Python.

---

## 6. Getting timestamps when the ASR has none

Subtitles need segment-level timing, and word-level timing helps the planned "Synchronizer". The pipeline I'd use with Cohere or Jais:

1. **Optional:** vocal or music separation (e.g. Demucs) on drama audio, to suppress score and SFX. This is not benchmarked here.
2. **Silero VAD** (MIT) → speech regions. Split at pauses into chunks of ≤ ~15–25 s. The chunk boundaries **are** your segment timestamps. This also stops the non-speech hallucinations.
3. Run ASR per chunk. Drop outputs that hit loop detection or a refusal token.
4. **Word-level forced alignment** of the ASR text to the chunk audio with an Arabic CTC model. Commercially OK options:
   - [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) (Apache-2.0; WhisperX's default `ar` aligner, [alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)), via [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) (BSD-2).
   - NVIDIA's Arabic FastConformer (CC-BY-4.0) with the NeMo Forced Aligner.
   - omniASR-CTC (Apache-2.0) emissions with torchaudio's `forced_align`.
   - **Avoid** ctc-forced-aligner's default MMS-300M aligner if you need commercial use: it is CC-BY-NC-4.0.
   - Qwen3-ForcedAligner does not support Arabic.
5. Alternative: run Whisper-turbo only for timing and Cohere or Jais for text, then align the two. This is more complex; only worth it if step 4 is poor on noisy dialogue.

---

## 7. Fine-tuning feasibility

**Compute:**
- **Feasible on the 3070:** LoRA or full fine-tune of Whisper turbo or small, and LoRA on Whisper large-v3 with 8-bit optimizers.
- **Rent a GPU (24–48 GB)** for Cohere (2B) or Jais full fine-tunes. The Jordanian Cohere fine-tune took 1.5 h on one L40S. Note that a careless Cohere fine-tune *regressed* Syrian (43.0 → 56.3), so hold out a test set.

**Levantine and Syrian speech data on HF (and nearby):**

| Dataset | Content | Size | Licence / caveats |
|---|---|---|---|
| [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) | **TV series** from YouTube; Jordan and Palestine configs (no Syria) | Paper: 6.0 h JOR, 6.05 h PAL (48 h total). HF release has validation and test only (~850 utts per country) | **CC-BY-NC-ND-4.0**, research only. Ideal as an *eval* set, not commercial training. |
| NADI 2025 / 2026 ASR data | Casablanca-derived; 1,600 utts per dialect per split (2025) | small | Shared-task terms; not verified |
| MASC ([IEEE DataPort](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus)) | ~1,000 h, 700+ YouTube channels, dialect-labelled. **Levantine ≈148 h** per [arXiv:2506.02627](https://arxiv.org/abs/2506.02627) | 148 h Levantine | Mirrors disagree: CC-BY-4.0 vs CC-BY-NC-4.0. Check DataPort. Preprocessed Levantine split: [otozz/levantine_train_set](https://huggingface.co/datasets/otozz/levantine_train_set) (14,247 examples, tagged CC-BY-4.0) and [test](https://huggingface.co/datasets/otozz/levantine_test_set) |
| [oddadmix/dialectal-arabic-lahgtna-v2](https://huggingface.co/datasets/oddadmix/dialectal-arabic-lahgtna-v2) | 3,000+ h, 13 dialects incl. **sy / lb / ps**; dialect spelling preserved | ~611k utts; per-dialect hours not stated | **Licence not stated**; provenance not documented |
| [oddadmix/lahgtna-v3-small](https://huggingface.co/datasets/oddadmix/lahgtna-v3-small) | Balanced 13-dialect subset (52k train / 2.6k test, 200 test clips per dialect) | — | Licence not stated |
| [oddadmix/arabic-audio-collection-syrian-podcast](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-syrian-podcast) | Syrian podcast; transcripts are **AI-assisted** with non-verbal tags | ~116 h, 24,406 clips, few speakers | "other"; original creators keep rights to the audio |
| [Overtech/syrian-arabic-speech-dataset](https://huggingface.co/datasets/Overtech/syrian-arabic-speech-dataset) (SYRA) | Syrian dialect | 116 h 31 m, 24,653 clips | Custom "SYRA Research Dataset License". The size matches the podcast set above, so it may be the same audio; verify. |
| [SpeechDataAI/levantine-arabic-conversational-speech](https://huggingface.co/datasets/SpeechDataAI/levantine-arabic-conversational-speech) | Spontaneous two-person conversations recorded in Lebanon, time-aligned | 500 h | **Commercial, paid** ($65/h); card only, not hosted |
| [mohammedaly22/lahgtna-levantine-tts](https://huggingface.co/datasets/mohammedaly22/lahgtna-levantine-tts) | **Synthetic** TTS Levantine + code-switching | 50k utts, 10 voices | CC-BY-4.0; augmentation only |
| [RanaGaber/Syrian_Rewayat](https://huggingface.co/datasets/RanaGaber/Syrian_Rewayat) and [_Segmented](https://huggingface.co/datasets/RanaGaber/Syrian_Rewayat_Segmented) | No dataset card | 10.6k / 15.0k rows | Unknown content and licence; inspect before use |
| Omnilingual ASR corpus, `apc_Arab` subset ([viewer](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)) | Levantine | 517 rows | CC-BY-4.0; tiny |
| WhisperLevantine's 1,200 h | Israeli Levantine | — | **Not released** (model only) |

**Practical route:**
1. Pseudo-label your own Syrian drama audio with Cohere or Jais.
2. Hand-correct a few hours.
3. Fine-tune.

Copyright on broadcaster audio is your call. Weak labels from strong teachers is exactly how Munsit won NADI 2025.

---

## 8. Suggested next step: a small bake-off on the 3070

1. Hand-transcribe **60–120 min** of real Syrian drama (several series, music-heavy scenes included). Keep it as a held-out test set.
2. Run five configurations with an identical Silero VAD front end:
   - Cohere (greedy)
   - Cohere + Sygma beam
   - Jais Q4_K_M (with `language=SY` and with auto)
   - Whisper-large-v3-turbo dialectal-v2 FT
   - Whisper-large-v3 base
3. Score WER and CER with the leaderboard normalizer. Also count **hallucinated lines on music-only segments** and measure wall-clock RTF.
4. Spot-check whether the downstream EN translation quality differs between them. Spelling differences that inflate WER often don't matter for translation.

---

## 9. Caveats and what I could not verify

- My **web-search budget ran out** near the end. I did not run dedicated searches for:
  - closed-API Levantine benchmarks (Gemini, gpt-4o-transcribe, Chirp, Azure)
  - LDC's paid Levantine telephone corpora
  - Seamless's Arabic variety codes

  A follow-up session can cover these.
- **All 3070 RTF figures are estimates.** Only the faster-whisper numbers come from an 8 GB Ampere card (3070 Ti).
- Jais-ASR and Audar results are largely vendor-reported. Only the leaderboard rows were reproduced by the leaderboard maintainers; Jais's row on the board appears to be the vendor's submission run with the leaderboard script.
- The per-country Casablanca numbers in §3.2(a) are my computation from third-party (Sygma) inference outputs, not official numbers.

## 10. Sources (main)

- Leaderboards and shared tasks:
  - [Open Universal Arabic ASR Leaderboard: Space](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard), [GitHub](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard), [paper](https://arxiv.org/abs/2412.13788)
  - [Casablanca paper](https://arxiv.org/abs/2410.04527) and [dataset](https://huggingface.co/datasets/UBC-NLP/Casablanca)
  - [NADI 2025](https://arxiv.org/abs/2509.02038) / [ACL](https://aclanthology.org/2025.arabicnlp-sharedtasks.99), [NADI 2026](https://arxiv.org/abs/2609.27086), [Munsit NADI paper](https://arxiv.org/abs/2508.08912)
- Models:
  - Cohere: [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026), [release blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release), [Sygma](https://huggingface.co/abedbanna/Sygma-ASR-Arabic)
  - Jais: [Jais-ASR](https://huggingface.co/IFM/Jais-ASR) and its [GGUF](https://huggingface.co/IFM/Jais-ASR-GGUF) build
  - Audar: [Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo), [Flash](https://huggingface.co/audarai/Audar-ASR-V1-Flash), [licence](https://www.audarai.com/license/audarai-community-license-v1.0/)
  - Meta: [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), [omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B), [paper](https://arxiv.org/abs/2511.09690), [MMS-1b-all](https://huggingface.co/facebook/mms-1b-all), [SeamlessM4T-v2](https://huggingface.co/facebook/seamless-m4t-v2-large)
  - Whisper: [large-v3](https://huggingface.co/openai/whisper-large-v3), [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  - Qwen: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
  - NVIDIA: [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), [Arabic FastConformer](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
  - Mistral: [Voxtral Small](https://huggingface.co/mistralai/Voxtral-Small-24B-2507), [Voxtral Mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Voxtral Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
  - Others: [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it), [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)
- Fine-tunes:
  - [oddadmix whisper-large-v3 v2](https://huggingface.co/oddadmix/whisper-large-v3-arabic-dialectal-v2), [turbo v2](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2), [cohere dialectal v2](https://huggingface.co/oddadmix/cohere-transcribe-arabic-07-2026-dialectal-v2)
  - [Cohere-Jordanian-Dialect](https://huggingface.co/Rlamas/Cohere-Jordanian-Dialect), [WhisperLevantine](https://huggingface.co/HebArabNlpProject/WhisperLevantine)
- Runtimes and speed:
  - [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) and its [Cohere Arabic doc](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/cohere-transcribe-arabic-07-2026.md)
  - [faster-whisper (RTX 3070 Ti benchmark)](https://github.com/SYSTRAN/faster-whisper), [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest), [mlx vs whisper.cpp](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)
  - [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner), [WhisperX aligner map](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)
- Hallucination: [whisper#2608](https://github.com/openai/whisper/discussions/2608), [arXiv:2501.11378](https://arxiv.org/abs/2501.11378)
- Data: [Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning (arXiv:2506.02627)](https://arxiv.org/abs/2506.02627), [MASC](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus), plus the dataset links in §7.
