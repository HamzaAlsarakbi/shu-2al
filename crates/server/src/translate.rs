//! Translate stage: corrects the Arabic and translates it to English, one
//! output line per cue, in scene-sized batches with rolling context.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;

use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use subs::Cue;

use crate::llm::{Lease, Usage};

pub const SYSTEM: &str = r#"You are subtitling a Syrian TV drama. You receive numbered cues of Arabic dialogue that an automatic speech recognizer transcribed. Expect recognition errors, missing words and inconsistent dialect spelling.

Return exactly one object per input cue, with the same id, in the same order. Never merge, split, drop or reorder cues; their timings depend on the ids.

"ar": the cue's Arabic, corrected and written in this orthography:
- Restore Standard Arabic spelling of words shared with Standard Arabic even when pronounced differently: آل→قال، ألب→قلب، تلاتة→ثلاثة، كتير→كثير، دهب→ذهب، عالبيت→على البيت.
- Keep every dialect-only word as it is commonly written: شو، هلق، بدي، منيح، هيك، لسا، مبارح، ليش، وين، هيدا، هاد.
- Standard hamza, ة and ى. No diacritics. Arabic punctuation (، ؟).
- Fix a misrecognized word only when the context makes the intended word clear. If unsure, keep the recognizer's word and set "uncertain": true.
- Never paraphrase and never convert dialect vocabulary to Standard Arabic.
- If a cue is only a non-speech tag or noise, return "" for both "ar" and "en".

"en": natural, idiomatic English a viewer can read at subtitle speed.
- When a sentence runs across several cues, understand the whole sentence first, then split the English across those cues in the same order, so each cue carries the part of the meaning it speaks. End a cue that breaks mid-sentence with "…" and start the continuation with "…".
- Each cue has "max_chars": how many characters a viewer can read while it is on screen. Keep "en" within it. Condense: drop filler, repetition and words the picture makes obvious, but keep the meaning and tone.
- Translate meaning and tone, not words. Contractions are fine. Short Arabic gets short English.
- Keep a little flavor where the meaning is obvious: habibi, yalla, wallah, inshallah, Abu/Umm + name. Otherwise use the natural English (تقبرني → "I adore you", يا خالتي → "Auntie", الله يرحم والديك → "Bless you").
- Never translate names. Match profanity and insults at equal severity; do not soften.

"previous" holds the cues just before this batch with their final English, for continuity only; do not return them. "summary" is the story so far.

Also return "scene_summary": at most two sentences describing who is talking and what is happening, for the next batch."#;

/// Reply schema for a batch of `n` cues. Exact array length and capped string
/// lengths stop runaway generations at the grammar level.
fn schema(n: usize) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["lines", "scene_summary"],
        "properties": {
            "lines": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "ar", "en", "uncertain"],
                    "properties": {
                        "id": { "type": "integer" },
                        "ar": { "type": "string", "maxLength": 300 },
                        "en": { "type": "string", "maxLength": 300 },
                        "uncertain": { "type": "boolean" }
                    }
                }
            },
            "scene_summary": { "type": "string", "maxLength": 400 }
        }
    })
}

#[derive(Debug, Clone, Default)]
pub struct Translated {
    pub ar: String,
    pub en: String,
    pub uncertain: bool,
    /// The corrected Arabic differs from the ASR by more than the guard allows.
    pub overcorrected: bool,
}

#[derive(Debug, Deserialize)]
struct Reply {
    lines: Vec<ReplyLine>,
    scene_summary: String,
}

#[derive(Debug, Deserialize)]
struct ReplyLine {
    id: usize,
    ar: String,
    en: String,
    uncertain: bool,
}

pub struct Settings {
    pub scene_gap_ms: u64,
    pub batch_min: usize,
    pub batch_max: usize,
    pub context_cues: usize,
    /// Flag a cue when CER(asr → corrected) exceeds this.
    pub overcorrection_cer: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            scene_gap_ms: 4000,
            batch_min: 10,
            batch_max: 25,
            context_cues: 8,
            overcorrection_cer: 0.4,
        }
    }
}

/// Translates every cue. Batches whose reply doesn't match the cue ids are
/// retried once, then split in half; a single cue that still fails keeps its
/// ASR Arabic, gets empty English and is marked uncertain.
pub async fn translate(
    lease: &Lease,
    cues: &[Cue],
    settings: &Settings,
    on_progress: impl Fn(usize, usize),
) -> Result<(Vec<Translated>, Usage)> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;
    let mut out: Vec<Option<Translated>> = vec![None; cues.len()];
    let mut usage = Usage::default();
    let mut summary = String::new();
    let mut pending: VecDeque<Range<usize>> = batches(cues, settings).into();

    while let Some(range) = pending.pop_front() {
        let previous: Vec<Value> = (range.start.saturating_sub(settings.context_cues)..range.start)
            .filter_map(|i| {
                let t = out[i].as_ref()?;
                Some(json!({ "id": i + 1, "ar": t.ar, "en": t.en }))
            })
            .collect();
        let lines: Vec<Value> = range
            .clone()
            .map(|i| {
                json!({
                    "id": i + 1,
                    "ar": cues[i].text.replace('\n', " "),
                    "max_chars": reading_budget(&cues[i]),
                })
            })
            .collect();
        let user = json!({ "summary": summary, "previous": previous, "lines": lines }).to_string();

        let mut reply = None;
        for _ in 0..2 {
            match request(lease, &http, &user, &range).await {
                Ok((r, u)) => {
                    usage += u;
                    reply = Some(r);
                    break;
                }
                Err(e) => eprintln!("  translate {range:?}: {e:#}"),
            }
        }
        match reply {
            Some((by_id, scene_summary)) => {
                for i in range.clone() {
                    let line = &by_id[&(i + 1)];
                    let overcorrected = !line.ar.is_empty()
                        && subs::metrics::cer(&cues[i].text, &line.ar)
                            > settings.overcorrection_cer;
                    out[i] = Some(Translated {
                        ar: line.ar.clone(),
                        en: line.en.clone(),
                        uncertain: line.uncertain || overcorrected,
                        overcorrected,
                    });
                }
                summary = scene_summary;
                on_progress(out.iter().filter(|t| t.is_some()).count(), cues.len());
            }
            None if range.len() > 1 => {
                let mid = range.start + range.len() / 2;
                pending.push_front(mid..range.end);
                pending.push_front(range.start..mid);
            }
            None => {
                let i = range.start;
                out[i] = Some(Translated {
                    ar: cues[i].text.clone(),
                    uncertain: true,
                    ..Default::default()
                });
            }
        }
    }
    Ok((
        out.into_iter().map(Option::unwrap_or_default).collect(),
        usage,
    ))
}

async fn request(
    lease: &Lease,
    http: &reqwest::Client,
    user: &str,
    range: &Range<usize>,
) -> Result<((HashMap<usize, ReplyLine>, String), Usage)> {
    // About 3× the tokens a cue normally needs, so a loop is cut off early.
    let max_tokens = 300 + 150 * range.len() as u32;
    let (value, usage) = lease
        .chat_json(http, SYSTEM, user, &schema(range.len()), max_tokens)
        .await?;
    let reply: Reply = serde_json::from_value(value)?;
    let by_id: HashMap<usize, ReplyLine> = reply.lines.into_iter().map(|l| (l.id, l)).collect();
    let missing: Vec<usize> = range
        .clone()
        .map(|i| i + 1)
        .filter(|id| !by_id.contains_key(id))
        .collect();
    if !missing.is_empty() {
        bail!("reply is missing cue ids {missing:?}");
    }
    Ok(((by_id, reply.scene_summary), usage))
}

/// Characters a viewer can read during the cue: 17 per second, between 16 and
/// two full lines (84).
fn reading_budget(cue: &Cue) -> u64 {
    (cue.duration_ms() * 17 / 1000).clamp(16, 84)
}

/// Splits cues into batches of `batch_min..=batch_max`, preferring to break
/// at scene gaps (long silences).
fn batches(cues: &[Cue], settings: &Settings) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=cues.len() {
        let len = i - start;
        let at_end = i == cues.len();
        let scene_break =
            !at_end && cues[i].start.0.saturating_sub(cues[i - 1].end.0) >= settings.scene_gap_ms;
        if at_end || len == settings.batch_max || (len >= settings.batch_min && scene_break) {
            out.push(start..i);
            start = i;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use subs::Timestamp;

    fn cue_at(start_s: u64) -> Cue {
        Cue::new(
            Timestamp(start_s * 1000),
            Timestamp(start_s * 1000 + 900),
            "x",
        )
    }

    #[test]
    fn batches_break_at_scene_gaps_within_bounds() {
        // 12 cues 1 s apart, a 10 s gap, then 50 cues 1 s apart.
        let mut cues: Vec<Cue> = (0..12).map(cue_at).collect();
        cues.extend((22..72).map(cue_at));
        let settings = Settings::default();
        let b = batches(&cues, &settings);
        assert_eq!(b, vec![0..12, 12..37, 37..62]);
        assert!(b.iter().all(|r| r.len() <= settings.batch_max));
    }

    #[test]
    fn small_inputs_make_one_batch() {
        let cues: Vec<Cue> = (0..3).map(|i| cue_at(i * 10)).collect();
        assert_eq!(batches(&cues, &Settings::default()), vec![0..3]);
        assert!(batches(&[], &Settings::default()).is_empty());
    }
}
