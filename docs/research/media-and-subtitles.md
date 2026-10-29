# Media, audio and subtitle-formatting stages for shu-2al

*Research date: 2026-10-27/28. Scope: everything around ASR and translation (both are researched separately in [asr-models.md](asr-models.md) and [architecture.md](architecture.md)).*

**How to read the markers**

- **[measured]**: run on the dev Mac during this research.
- **[estimate]**: scaled from published numbers. I could not benchmark on the RTX 3070, so measure these before relying on them.
- **UNVERIFIED**: the claim could not be confirmed online. The shared web-search budget ran out near the end of this research.

**Owner decisions this report follows**

- **Hardware and deployment:** inference runs on an Arch Linux box with an RTX 3070 (8 GB), under Docker Compose with the NVIDIA toolkit. Python ML workers are acceptable; Rust is preferred everywhere else.
- **Sources:** YouTube channels and playlists, other sites yt-dlp supports, local files (upload or drop), and the owner's Plex server on another LAN machine.
- **Requests:** paste a URL or playlist, upload or drop a file, or search by show name (yt-dlp `ytsearch`).
- **Outputs:** separate `.ar.srt` and `.en.srt` files (no stacked dual-language track).
  - The Arabic track uses MSA-normalized orthography and follows Netflix's Arabic rules.
  - The English track follows Netflix's English rules.
- **Delivery:** download from the dashboard, and push to the owner's live Plex library.
- **Retention:** keep the 16 kHz audio (as training data) and a 480p proxy (for the in-browser editor). Delete the full download.
- **Editor:** v1 has full waveform timing using wavesurfer.js regions.

---

## 0. TL;DR: recommended pipeline

Stage names match [architecture.md §4](architecture.md).

| # | Stage | Tool (runs where) | Status | Time per 45-min episode on the 3070 box |
|---|---|---|---|---|
| 0 | **Request intake** | Dashboard: URL/playlist paste, file drop (tus or streamed), `ytsearch`, Plex browse | must | – |
| 1 | **fetch** | `yt-dlp` subprocess (Rust); Plex part download; local path | must | Network-bound. Add 10–20 s of polite sleep with `-t sleep` |
| 2 | **extract** | **One** `ffmpeg` pass producing: `audio.16k.flac` (training data + ASR input), `proxy.mp4` (480p H.264/AAC, faststart), `shots.txt` (scdet). Then Rust computes `peaks.json` for wavesurfer | must | ~1–3 min on CPU **[estimate]**. Peaks take <5 s |
| 3 | **separate** | `audio-separator` (Mel-/BS-RoFormer) or htdemucs, in a Python worker | **optional, A/B-gated** | htdemucs ~1–2 min; RoFormer ~5–10 min at overlap 2 **[estimate]** |
| 4 | **vad** | Silero VAD v6.2.x (Python worker on CPU, or Rust `silero-vad-rust`/`sherpa-onnx`) | must | ~10–60 s on CPU **[estimate]** |
| 5 | asr | *out of scope* ([asr-models.md](asr-models.md)) | – | Reference: faster-whisper large-v2 on an RTX **3070 Ti** took 63 s for 13 min of audio (fp16, beam 5, 4.5 GB VRAM) and 17 s batched (6.1 GB) **[measured by the faster-whisper authors]** |
| 6 | **align** | WhisperX ≥3.8.4 `whisperx.align()` with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (Python worker, CUDA). A/B against Omnilingual CTC (has `apc_Arab`) | must (the top ASR candidates emit no timestamps) | ~0.5–1.5 min, ~2–3 GB VRAM **[estimate]** |
| 7 | **cues: filter** | Rust: normalized phrase lists + signal checks (VAD overlap, music, alignment score, loops, CPS outliers) | must | <1 s |
| 8 | **cues: segment (ar)** | Rust: Netflix Arabic rules (42 CPL, 2 lines, 20 CPS, 5/6–7 s, 2-frame gaps) | must | <1 s |
| 9 | translate | *out of scope*, then a Rust **English fit pass** (Netflix English rules) onto the Arabic timing template | must (fit pass) | <1 s |
| 10 | **snap** | Rust: shot-change snapping using `shots.txt` from stage 2 | optional (nearly free) | <1 s |
| 11 | **diarize** | pyannote `speaker-diarization-community-1` (Python worker) | optional, phase 2 | ~1.5–3 min **[estimate]** |
| 12 | **publish** | Rust SRT writer (UTF-8, Arabic bidi wrapping), VTT for the editor; dashboard zip; Plex upload API (sidecar fallback) | must | <1 s |

Excluding ASR and translation, the **must-have path costs about 3–5 minutes per episode** after download.

Separation adds ~1–10 minutes when enabled. The 8 GB card means GPU stages must run **one at a time, in separate processes**, which matches [architecture.md §5](architecture.md).

**What is optional:**
- Separation: adopt only if it wins an A/B on a hand-corrected dev set.
- Linear loudness normalization.
- Music/AudioSet tagging.
- Shot-change snapping.
- Diarization (dialogue dashes).
- MKV muxing.
- Burn-in.

**What is not needed:**
- Stacked dual-language output (owner decision).
- Classical denoisers (`afftdn`, `arnndn`, `dialoguenhance`).
- `dynaudnorm`.
- ASS output.

### Implication for the existing code

The prototype's `Module` trait is `SRT in → SRT out` (`src/modules/module.rs`). Filtering, alignment-aware segmentation and shot snapping all need **word-level data**: word start/end, word/segment confidence, VAD overlap and music probability.

So the cue stages should consume the worker's `transcript.json` (words + scores) and only *emit* SRT at publish time.

The current `FilterModule` drops any subtitle that **contains** a listed phrase. That would delete real dialogue such as "صوت الموسيقى عالي". See §5 for a dominance-based rule.

---

## 1. Acquisition (stage 1)

### 1.1 Where Syrian dramas are legitimately available

**YouTube channels verified to post full episodes** (35–55 min each; checked with `yt-dlp 2026.08.19` queries from the dev Mac):

| Channel | Handle / channel ID | Series seen | Example title | Arabic auto-captions |
|---|---|---|---|---|
| Clacket Media – كلاكيت ميديا | `@Clacketproduction` / `UC3LGwTaBCu9EE4wwxqtLVXQ` | كسر عضم، غداً نلتقي، ولادة من الخاصرة، الإخوة | `مسلسل غداً نلتقي ـ الحلقة 1 الأولى كاملة HD \| Ghadan Naltaqi` | none |
| Golden Line | `@GoldenLineProduction` / `UCyaYjMo5HeeTygGrKvLZssw` | العربجي، على صفيح ساخن | `مسلسل العربجي ـ الحلقة 1 الأولى كاملة ـ Al Arbagi HD` | none |
| Sama Art International | `@samaartinternational8819` / `UCmGzgW9ZQfIJu7qA_IVHKgA` | ضيعة ضايعة | `مسلسل ضيعة ضايعة - الجزء الأول ـ الحلقة 2 الثانية كاملة HD \| الدكان` | yes (`ar-orig`) |
| Cedars Drama (Sabbah) | `@Cedarsdrama` / `UCXEOrz7UURyW5wGAEUt0Xnw` | الهيبة (all seasons) | `مسلسل الهيبة ( العودة ) \| الحلقة 1 \| بطولة ... - FULL HD` | yes |
| Kaband Art Production (قبنض) | `@KabandArtProduction` / `UCET-WX1ZZDKrXY4_LSUp6RA` | عطر الشام، شرف | `مسلسل عطر الشام 4 - الحلقة 1 الأولى - كاملة \| Etr Al Shaam` | – |
| EBLA International | `@ebla` | عرب لندن | `مسلسل عرب لندن ـ الحلقة 1 الأولى كاملة HD \| Arab London` | yes |
| Al Ward Pictures | `@alwardpictures` / `UCV1R_SsQUqDhnJEXTMfxMQQ` | مقابلة مع السيد آدم، دومينو | `مسلسل مقابلة مع السيد آدم الحلقة 1 الأولى بطولة غسان مسعود` | none |
| Aj Art Production (عاج) | `@3AJ` / `UCeuEvvfHa4BFK21xB6RXJXA` | باب الحارة، ليالي الصالحية، الوصية | `مسلسل باب الحارة الحلقة 1 الأولى بطولة عباس النوري` | – |
| Tareq Zuaiter | `@tareqzuaiter` / `UCREZ2f4IPZ4WpztKocSdnOQ` | باب الحارة S1–2، زهرة النرجس | `مسلسل باب الحارة الجزء الاول الحلقة 5 الخامسة \| Bab Al Harra Season 1 HD` | – |
| FLARE Arts & Media | `@FlareArtsMedia` / `UCM3ZDdbvu5Hjw4czms6V8cQ` | الندم، الخربة، التغريبة الفلسطينية | `مسلسل الخربة - الحلقة التاسعة (تشكيلات جديدة) - Al Kherbeh Ep 9` | – |

Notes on these channels:
- **Rights status:** inferred from channel names and production credits, not confirmed. FLARE, Tareq Zuaiter and `@3AJ` look like distributors.
- **Other sources of full episodes:**
  - Al-Alamiya (`@alalamya`).
  - Emarat TV (`@EmaratTV`) and Saudi Channel One (`@SaudiChannelOne`).
  - Shoof Drama aggregator (`@shoofdramaone`, `@shoofdrama2`). Licensing is UNVERIFIED.
  - Bassam al-Malla (`@Bassamalmalla1`), which posts ~2 h videos.
- **Clips and promos only:** `@MBCDrama`, `@MBCShahid`, `@MBC1`, `@CedarsArtProduction`, `@LANA-TV`, `@SyriaTelevision`.
- **Not found:** a SAPI channel, Al Ola, an official Syrian-TV drama channel, LBC.
- **Re-uploaders** repost other companies' series ("El Haram Production Plus", "Fox Media Production Plus", "Dee Drama", "NOOR TV"). **Allow-list by `channel_id`.**
- **Playlists:**
  - Full-series playlists exist on `@3AJ` (playlist index = episode), `@alwardpictures`, `@Cedarsdrama` and Sama.
  - Many other playlist tabs are clip compilations. **Always parse titles too.**
  - yt-dlp 2026.08.19 failed on `@tareqzuaiter`'s tab with `Unsupported lockup view model content type "LOCKUP_CONTENT_TYPE_SHOW"`. Re-test after updating.
- **Duplicates:** Clacket and Sama upload HD and 4K copies under different IDs.
- **Captions:** 0 of 7 sampled episodes had human Arabic subtitles; 3 of 7 had YouTube auto-captions (`ar-orig`). They are useful at most as a reference (`--write-auto-subs --sub-langs ar-orig --skip-download`).
- **Geo-blocking:** none seen from the dev network. Other regions are UNVERIFIED. The Data API field `contentDetails.regionRestriction` shows per-video restrictions ([videos resource](https://developers.google.com/youtube/v3/docs/videos)).

**Legal note.** YouTube's Terms forbid downloading "except (a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders", and forbid automated access ([YouTube Terms](https://www.youtube.com/static?template=terms)). Using a rights-holder's channel doesn't change that. This is a risk decision for the owner (see open questions).

**Subscription services (Shahid, OSN+, Netflix, Yango Play, TOD): out of scope.**
- They stream DRM-protected video.
- Shahid's terms grant "personal, non-commercial use" and forbid bypassing "security, territorial, device, concurrency or content protections" and scraping. Offline viewing exists only inside the app ([Shahid T&C](https://shahid.mbc.net/en/terms-and-conditions)).
- Netflix forbids reproducing content, circumventing protections, and automated access ([Netflix ToU](https://help.netflix.com/legal/termsofuse)).
- yt-dlp states "We also cannot support any service that serves only DRM protected content" ([CONTRIBUTING](https://github.com/yt-dlp/yt-dlp/blob/master/CONTRIBUTING.md)). Its `Shahid` extractor needs login and calls `report_drm()` on protected streams ([shahid.py](https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/extractor/shahid.py)).
- OSN+, Yango and TOD terms and DRM schemes are UNVERIFIED.
- **The dashboard should show yt-dlp's DRM error as "not supported" and stop.** It must not retry.

### 1.2 yt-dlp: version and upkeep

**Versions**
- Latest is **2026.10.26** (Python ≥3.11; bundles `yt-dlp-ejs`) ([release](https://github.com/yt-dlp/yt-dlp/releases/tag/2026.10.26)).
  - Arch `extra/yt-dlp` is **2026.10.26-1** (2026-10-27), depending on `yt-dlp-ejs` ([Arch](https://archlinux.org/packages/extra/any/yt-dlp/)).
  - Homebrew had 2026.10.21 ([formula](https://formulae.brew.sh/formula/yt-dlp)).
  - The dev Mac has **2026.08.19**, installed via pip, and `yt-dlp -U` refuses to self-update it.
- 2026 releases came out roughly monthly, in bursts. yt-dlp calls the nightly channel "the recommended channel for regular users" ([README#update](https://github.com/yt-dlp/yt-dlp#update)).

**JS runtime**
- YouTube needs `yt-dlp-ejs` plus a JS runtime.
- Only **deno** is enabled by default (minimum 2.3.0). node/quickjs are opt-in via `--js-runtimes`.
- `--remote-components ejs:github` fetches the solver if the bundled one is missing ([EJS wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS)).
- **In the container, install deno explicitly.** The Arch package page does not list it as a dependency.

**Player clients and PO tokens**
- Default clients in 2026.10.26 are `tv_samsung, visionos, web`.
- PO tokens are needed for `web`/`mweb`/`android`/`ios`. Test downloads worked without cookies or PO tokens.
- If that changes, use the `bgutil-ytdlp-pot-provider` plugin ([PO-Token guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)).

**Rate limits and cookies**
- About 300 videos/hour as a guest and about 2,000 with an account. The project recommends 5–10 s between downloads (`-t sleep`).
- "By using your account with yt-dlp, you run the risk of it being banned." If cookies are used, export them from a private window rather than using `--cookies-from-browser` ([Extractors wiki](https://github.com/yt-dlp/yt-dlp/wiki/Extractors)).

**Recommendation**
- Pin and own yt-dlp inside the fetch container: `pip install -U --pre "yt-dlp[default]"`, plus deno.
- Expose an **"update yt-dlp" button / weekly job** in the dashboard and log the version with each job.
- Most YouTube breakages are fixed in nightly within days.

### 1.3 yt-dlp invocations (flags checked against local `--help`)

```sh
# Preview a URL or playlist without downloading (dashboard "resolve" step)
yt-dlp --flat-playlist -J "https://www.youtube.com/playlist?list=PL…"
yt-dlp --flat-playlist --print '%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s' URL
yt-dlp -J --no-playlist "https://www.youtube.com/watch?v=ID"   # full per-video metadata (Arabic original title)

# Download one episode for a job. The full download is deleted after extract, so don't fetch 1080p/4K.
yt-dlp -f "bv*+ba/b" -S "res:480,vcodec:h264,acodec:aac" --merge-output-format mkv \
  -o "%(extractor_key)s-%(id)s.%(ext)s" -P "$JOB/source" \
  --download-archive /data/archive.txt --write-info-json --no-playlist \
  --match-filters "duration>1200 & duration<4500" \
  -t sleep --newline --progress --progress-template "download:%(progress)j" \
  --print after_move:filepath  URL
```

Notes on these flags:
- **Episode ranges:** `-I 5:10` (or `-I 1:3,7`) on a playlist URL.
- **Lazy playlists:** `--lazy-playlist` processes entries as they arrive but disables `n_entries` and reverse/random order.
- **Audio-only mode** (batch jobs without the editor): `-f ba`. That picks webm/opus 251 (~106 kbps). `-x` with the default `--audio-format best` stream-copies without re-encoding.
- **Archive:**
  - `--download-archive` lines are `extractor id`.
  - `--break-on-existing` stops at the first already-archived item (good for polling a channel's `/videos`).
  - `--force-download-archive` records IDs even when simulating.
  - The DB is the real source of truth; the archive file is a cheap second guard.
- **Filenames:** name files by ID, not title. Arabic titles get mangled in paths. Keep titles in the DB.
- **Metadata:** YouTube **does not populate** `series`/`season_number`/`episode_number` for these channels (all 7 samples returned `NA`).
  - `--parse-metadata 'title:(?P<series>.+?)\s*[-ـ|]\s*الحلقة\s*(?P<episode_number>\d+)'` works for simple cases, but parsing in Rust (§1.6) is more robust.
  - Other extractors (non-YouTube sites) sometimes fill these fields. Prefer them when present.
- **Progress:** `--progress-template 'download:%(progress)j'` emits one JSON object per line (`status`, `downloaded_bytes`, `total_bytes`, `eta`, `speed`) **[measured by the research agent]**. `--print` implies `--quiet`, so add `--progress`.
- **Other yt-dlp sites:**
  - Same wrapper.
  - Some sites need `--impersonate` (requires `curl_cffi`) or a per-site cookies file. Store that as a secret.
  - Treat a DRM error as terminal.
- **Rust crates:**
  - `yt-dlp` (boul2gom) 2.9.1 (2026-10-26) is active but GPL-3.0 and auto-downloads binaries.
  - `rusty_ytdl` (2024) and `ytextract` (dead) can't cope with YouTube's JS challenges.
  - `rustypipe` 0.11.4 is pure Rust but GPL-3 and needs botguard for PO tokens.
  - **Use a plain `tokio::process::Command` wrapper.**

### 1.4 Plex as a source, and local files

**Plex (on another LAN host)**

Pull the episode's media part. The HTTP side is covered in [architecture.md §9](architecture.md). Plex-sourced items need no title parsing, because show, season, episode, `ratingKey` and the part path come from Plex metadata. They also have a natural delivery target: the same `ratingKey`.

How to pull it:
- `GET {part.key}?download=1` with the token, saved to the job's `source/`. Then extract and delete, exactly as with yt-dlp downloads.
- Or read via an SMB/NFS mount with a path map.
- ffmpeg could read the HTTP URL directly. That depends on Plex honouring range requests: UNVERIFIED.

Pick the audio stream by checking ffprobe for `tags.language == "ara"`, then `disposition.default`.

**Local upload or drop** (files the owner already has)

1. Stream the upload to disk:
   - axum's default body limit is 2 MB, so disable it for the upload route and use a `tower_http` limit ([DefaultBodyLimit](https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html)).
   - For resumable uploads use the tus 1.0 protocol ([tus](https://tus.io/protocols/resumable-upload)). `tus-axum` is brand new; the alternatives are `rustus` or a `tusd` sidecar.
   - For a single-user LAN app, a plain streaming upload is enough.
   - Files already on the box should be referenced in place, not copied.
2. Validate with `ffprobe -v error -of json -show_format -show_streams -show_chapters FILE`. Check:
   - At least one audio stream.
   - Duration roughly 15–120 min.
   - `channels`/`channel_layout` (stereo vs 5.1, which decides FC extraction in §3.1).
   - Language tags.
   - Video codec (the proxy is always re-encoded anyway).
3. Ask the user for show, season and episode, pre-filled by parsing the filename with the same parser as titles (§1.6).

### 1.5 "Request an episode" in the dashboard

| Flow | Mechanism | Notes |
|---|---|---|
| **Paste URL or playlist** | `yt-dlp --flat-playlist -J` → preview table (title, duration, channel, parsed S/E) → user ticks episodes → jobs | For playlists, offer an episode range that maps to `-I a:b`. Show the parsed episode number next to the playlist index, since they disagree on many channels. |
| **Upload / drop file** | tus or streamed upload → ffprobe → S/E form | Accept mkv/mp4/ts/webm/m4a/opus/wav. |
| **Browse Plex** | Plex library → pick episode | See the Plex research in architecture.md. |
| **Search by show name** | `yt-dlp --flat-playlist -J "ytsearch20:<query>"` → filter → preview | See the Arabic quirks below. |

**Dedupe** on two keys:
- `(extractor, video_id)`.
- `(series, season, episode)`, so the HD and 4K uploads of the same episode aren't both fetched.

**`ytsearch` quirks with Arabic queries**

1. **The titles in search results may be YouTube's English *translated* titles** (e.g. `Bab Al-Hara Season 1 Episode 2 | …`) instead of the Arabic original.
   - A full per-video extraction (`-J --no-playlist`) returns the Arabic title.
   - `--extractor-args "youtube:lang=ar"` did **not** change search-result titles (tested with 2026.08.19).
   - So parse both languages, and re-extract the top few candidates before showing them.
2. **Results are noisy:** clips, promos, re-uploads and 4K duplicates. Filter on:
   - `duration > 1200`.
   - An allow-list of `channel_id`s. Show non-allow-listed results greyed out.
   - Dedupe on the parsed (series, season, episode).
3. **Orthographic variants matter:**
   - الحلقة vs الحلقه, أ/إ/ا, ى/ي.
   - Digit vs ordinal word (الحلقة 5 / الحلقة الخامسة).
   - Arabic-Indic digits.

   Run 2–3 normalized query variants and merge the results. How tolerant YouTube's search is of these variants is **UNTESTED**.
4. **Search for the series, not the episode.** For "search by show name", the best result is usually the channel's **playlist** for that series.
   - Search for an episode, take its `channel_id`, list `https://www.youtube.com/channel/<id>/playlists` with `--flat-playlist`, and match the series name.
   - Alternatively, use a YouTube search URL with a playlist-type `sp` filter. yt-dlp accepts search URLs, but the exact `sp` value is UNVERIFIED.
5. **Sanitize the query:**
   - Pass it as **one argv element**; never build a shell string.
   - Strip bidi control characters from pasted text (U+200E/F, U+061C, U+202A–E, U+2066–9). Copy-paste from web pages often carries them.
   - Use `dir="auto"` on the input.
6. **Quota and rate limits:**
   - `ytsearch` scrapes, so it costs no quota but is subject to the same rate limits and bot checks.
   - The YouTube Data API alternative has had a separate `search.list` bucket of 100 calls/day since 2026-06-01 ([quota costs](https://developers.google.com/youtube/v3/determine_quota_cost), [revision history](https://developers.google.com/youtube/v3/revision_history)). `playlistItems.list` still costs 1 unit from the 10,000/day pool.

### 1.6 Parsing series, season and episode from titles

**Variants seen in real titles**
- Separators: `الحلقة`/`حلقة`/`الحلقه`, tatweel `ـ`, `|`, `-`.
- Number and ordinal in either order (`الحلقة 2 الثانية`, `الحلقة الأولى 1`), or the ordinal only (`الحلقة التاسعة`, Egyptian spelling `الاولي`).
- Finale marker: `والأخيرة`.
- Season forms:
  - `الجزء الاول` / `الجزء 2` / `الجزء الثاني`.
  - A bare number after the name (`عطر الشام 4`, `كسر عضم 2 السراديب`).
  - English `Season 1 Episode 2` / `Ep 9`.
- Noise to strip:
  - `HD`, `FULL HD`, `4K`, `كاملة`, `بجودة`, `بطولة <actor>`.
  - AI-upscale notes, `رمضان 2026`.
  - RLM characters, stray apostrophes (`الحلقة '4`).

**Algorithm (Rust `regex`)**
1. **Normalize:**
   - Strip tatweel U+0640 and bidi marks.
   - Fold `أإآ→ا`, `ى→ي`, final `ة→ه`.
   - Map U+0660–0669 and U+06F0–06F9 to ASCII. Rust `regex`'s Unicode `\d` matches them, but `char::to_digit` doesn't ([regex docs](https://docs.rs/regex/latest/regex/)).
2. **Episode:** `(?:ال)?حلق[هة]\s*[^\p{L}\d]{0,3}(\d{1,3})`. Fall back to an ordinal-word table (1–40, masculine and feminine, compounds like `الحادية والعشرون`), then to `(?i)\b(?:episode|ep)\.?\s*(\d+)`.
3. **Season:** `(?:ال)?(?:جزء|موسم)\s*(\d{1,2}|<ordinal>)`. Otherwise use a trailing number after the series name, or `Season|Part (\d+)`.
4. **Series:** the text after `مسلسل` up to the first of `الحلقة`/`الجزء`/`ـ`/`-`/`|`.
5. **Confidence:** higher when digit and ordinal agree, or when the result matches the playlist index. Low-confidence parses go to the user in the preview.

---

## 2. Media artifacts and retention (stage 2: extract)

### 2.1 What is kept

| Artifact | Kept? | Purpose | Size per 45 min |
|---|---|---|---|
| `source/media.*` (full download / Plex part) | **deleted** after extract (and after separation input is written) | – | 0.2–2 GB |
| `audio/audio.16k.flac` (mono, 16 kHz, s16) | **kept** | ASR/VAD/alignment input and training data | ~40–55 MB **[estimate: 86.4 MB raw PCM, FLAC ~50–60%]** |
| `audio/proxy.mp4` (480p H.264 High + AAC, faststart, 2 s GOP) | **kept** (until the episode is closed) | Editor playback | ~150–300 MB at CRF 26 **[estimate]** |
| `audio/shots.txt` → `shots.json` | kept | Shot-change markers and snapping | <50 KB |
| `audio/peaks.*` | kept | wavesurfer.js waveform (§8) | ~0.1–2 MB |
| `work/sep_in.flac` (44.1 kHz stereo) | temporary | Separation input, only if separation is enabled | ~250 MB |
| `audio/vocals.16k.flac` | optional | Separated stem (VAD signal; could be extra training data) | ~40–55 MB |

**Ordering matters.** Separation needs 44.1 kHz stereo, which the 16 kHz file can't provide. If separation is enabled, write `sep_in.flac` in the same ffmpeg pass, and only delete it after the separate stage finishes.

### 2.2 The single-pass ffmpeg command

This reads the source once and writes the training/ASR audio, the proxy and the shot list together. **[measured]** on ffmpeg 9.0.2 with a synthetic 1080p25 5.1 file, it produced:
- 854×480 H.264 High + AAC-LC 48 kHz stereo.
- `moov` atom before `mdat` (faststart).
- Keyframes at 0/2/4 s.
- A 16 kHz s16 mono FLAC.
- The hard cut detected at exactly 3.0 s.

```sh
SRC=source/media.mkv; A=0   # audio stream index chosen from ffprobe

# Stereo source: plain mono downmix.  5.1 source: replace -ac 1 with -af "pan=mono|c0=FC"
ffmpeg -nostdin -hide_banner -y -i "$SRC" \
  -map 0:a:$A -vn -sn -dn -ac 1 -ar 16000 -c:a flac -sample_fmt s16 audio/audio.16k.flac \
  -map 0:v:0 -map 0:a:$A -sn -dn \
  -vf "scale=-2:480,scdet=threshold=10,metadata=mode=print:key=lavfi.scd.time:file=audio/shots.txt,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 26 -profile:v high -level:v 4.0 \
  -force_key_frames "expr:gte(t,n_forced*2)" \
  -c:a aac -b:a 96k -ac 2 -ar 48000 -map_metadata -1 -map_chapters -1 \
  -movflags +faststart audio/proxy.mp4 \
  -progress pipe:1 -nostats
# If separation is enabled, add a third output:
#   -map 0:a:$A -vn -ac 2 -ar 44100 -c:a flac work/sep_in.flac
#   (for 5.1, e.g. -af "pan=stereo|c0=FC+0.3*FL|c1=FC+0.3*FR")
```

Notes on the command:
- **Scaling:**
  - `scale=-2:480` keeps an even width.
  - For anamorphic sources (SAR ≠ 1, e.g. DVD rips) use `scale='trunc(oh*dar/2)*2':480,setsar=1`. That is standard ffmpeg expression usage, not tested here.
- **`format=yuv420p`** makes 10-bit/HEVC sources browser-playable.
- **`-force_key_frames` every 2 s** makes scrubbing in the editor snappy, independent of fps. x264 still adds keyframes at cuts.
- **`scdet` on the 480p frames is nearly free.** The default threshold is 10, and ffmpeg's docs say "good values are in the [8.0, 14.0] range" ([scdet](https://ffmpeg.org/ffmpeg-filters.html#scdet)). Rust converts `shots.txt` (`lavfi.scd.time=…` lines) to a JSON array of seconds.
- **Hardware variant on the 3070:**
  - `-hwaccel cuda -hwaccel_output_format cuda … -vf scale_cuda=-2:480 … -c:v h264_nvenc -preset p5 -cq 28`.
  - The container needs `NVIDIA_DRIVER_CAPABILITIES` to include `video` (§9).
  - It can't run `scdet` on GPU frames without `hwdownload`, so CPU libx264 is the simpler default. 480p encoding is cheap. On the M2, x264 `veryfast` at **1080p** ran at ~305 fps **[measured, synthetic]**.
- **Remux shortcut for YouTube:** if yt-dlp delivered H.264 ≤480p plus AAC (the `-S "res:480,vcodec:h264,acodec:aac"` sort), `-c copy -movflags +faststart` would skip re-encoding. YouTube's keyframe spacing is then outside our control (UNVERIFIED). Keep the uniform transcode as the default.
- **ffmpeg builds:** Arch's `ffmpeg 2:9.0.2-2` depends on `libass`, `fribidi`, `harfbuzz`, `freetype2` and `libsoxr` ([Arch package](https://archlinux.org/packages/extra/x86_64/ffmpeg/)). Burn-in and soxr resampling therefore work there.
  - The Mac's Homebrew core ffmpeg has **none of these** (`--enable-libass` is absent; there are no `subtitles`/`drawtext` filters).
  - On the Mac, use `brew install ffmpeg-full` (keg-only, includes libass/harfbuzz) if burn-in is ever needed ([formula](https://formulae.brew.sh/formula/ffmpeg-full)).
