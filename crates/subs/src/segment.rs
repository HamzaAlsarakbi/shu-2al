//! Turns an aligned transcript into subtitle cues that respect reading limits
//! (Netflix-style: 42 characters per line, 2 lines, 20 characters per second,
//! 5/6 s to 7 s on screen, a 2-frame gap between cues).

use crate::transcript::{Segment, Transcript};
use crate::{Cue, Timestamp};

#[derive(Debug, Clone, PartialEq)]
pub struct Rules {
    pub max_cpl: usize,
    pub max_lines: usize,
    pub max_cps: f64,
    pub min_dur_ms: u64,
    pub max_dur_ms: u64,
    pub min_gap_ms: u64,
    /// A silence this long ends a cue once the cue is substantial
    /// (over `min_split_chars` or `min_split_ms`), as do sentence and segment ends.
    pub pause_ms: u64,
    /// A silence this long always ends a cue.
    pub hard_pause_ms: u64,
    pub min_split_chars: usize,
    pub min_split_ms: u64,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            max_cpl: 42,
            max_lines: 2,
            max_cps: 20.0,
            min_dur_ms: 833,
            max_dur_ms: 7000,
            min_gap_ms: 83,
            pause_ms: 500,
            hard_pause_ms: 1500,
            min_split_chars: 15,
            min_split_ms: 1200,
        }
    }
}

#[derive(Debug, Clone)]
struct TimedWord {
    text: String,
    start_ms: u64,
    end_ms: u64,
    /// Last word of its ASR segment: a soft cue break.
    segment_end: bool,
}

/// Builds cues from a transcript. Segments without word timings (alignment
/// failed or was skipped) get words spread over the segment by length.
/// Non-speech tags such as `[موسيقى]` are dropped.
pub fn cues_from_transcript(transcript: &Transcript, rules: &Rules) -> Vec<Cue> {
    let words: Vec<TimedWord> = transcript.segments.iter().flat_map(segment_words).collect();
    let mut cues = group(&words, rules);
    cues.sort_by_key(|c| c.start);
    fix_timing(&mut cues, rules);
    for cue in &mut cues {
        cue.text = break_lines(&cue.text, rules.max_cpl);
    }
    cues
}

fn segment_words(seg: &Segment) -> Vec<TimedWord> {
    let ms = |s: f64| (s.max(0.0) * 1000.0).round() as u64;
    let mut words: Vec<TimedWord> = match &seg.words {
        Some(words) if !words.is_empty() => words
            .iter()
            .map(|w| TimedWord {
                text: w.w.clone(),
                start_ms: ms(w.start),
                end_ms: ms(w.end),
                segment_end: false,
            })
            .collect(),
        _ => spread(&seg.text, ms(seg.start), ms(seg.end)),
    };
    words.retain(|w| !is_tag(&w.text));
    if let Some(last) = words.last_mut() {
        last.segment_end = true;
    }
    words
}

/// Gives each word a slice of `[start, end]` proportional to its length.
fn spread(text: &str, start: u64, end: u64) -> Vec<TimedWord> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let total: usize = tokens.iter().map(|t| t.chars().count() + 1).sum();
    let span = end.saturating_sub(start) as f64;
    let mut at = 0usize;
    tokens
        .iter()
        .map(|t| {
            let from = start + (span * at as f64 / total as f64) as u64;
            at += t.chars().count() + 1;
            let to = start + (span * at as f64 / total as f64) as u64;
            TimedWord {
                text: (*t).to_owned(),
                start_ms: from,
                end_ms: to,
                segment_end: false,
            }
        })
        .collect()
}

fn is_tag(word: &str) -> bool {
    (word.starts_with('[') && word.ends_with(']')) || (word.starts_with('(') && word.ends_with(')'))
}

/// True when the words wrap into at most `max_lines` lines of `max_cpl` characters.
fn fits(words: &[&str], max_cpl: usize, max_lines: usize) -> bool {
    let mut lines = 1;
    let mut len = 0;
    for w in words {
        let n = w.chars().count();
        if n > max_cpl {
            return words.len() == 1;
        }
        if len == 0 {
            len = n;
        } else if len + 1 + n <= max_cpl {
            len += 1 + n;
        } else {
            lines += 1;
            len = n;
        }
    }
    lines <= max_lines
}

fn ends_sentence(word: &str) -> bool {
    word.ends_with(['.', '!', '?', '؟', '…'])
}

fn group(words: &[TimedWord], rules: &Rules) -> Vec<Cue> {
    let mut cues = Vec::new();
    let mut current: Vec<&TimedWord> = Vec::new();
    let flush = |current: &mut Vec<&TimedWord>, cues: &mut Vec<Cue>| {
        if let (Some(first), Some(last)) = (current.first(), current.last()) {
            let text: Vec<&str> = current.iter().map(|w| w.text.as_str()).collect();
            cues.push(Cue::new(
                Timestamp(first.start_ms),
                Timestamp(last.end_ms.max(first.start_ms)),
                text.join(" "),
            ));
        }
        current.clear();
    };

    for (i, word) in words.iter().enumerate() {
        if let Some(first) = current.first() {
            let mut texts: Vec<&str> = current.iter().map(|w| w.text.as_str()).collect();
            texts.push(&word.text);
            let too_long = word.end_ms.saturating_sub(first.start_ms) > rules.max_dur_ms;
            if too_long || !fits(&texts, rules.max_cpl, rules.max_lines) {
                flush(&mut current, &mut cues);
            }
        }
        current.push(word);

        let gap = words
            .get(i + 1)
            .map(|next| next.start_ms.saturating_sub(word.end_ms));
        let chars: usize = current.iter().map(|w| w.text.chars().count() + 1).sum();
        let substantial = chars > rules.min_split_chars
            || word.end_ms.saturating_sub(current[0].start_ms) >= rules.min_split_ms;
        let hard = gap.is_none_or(|g| g >= rules.hard_pause_ms);
        let soft = word.segment_end
            || gap.is_some_and(|g| g >= rules.pause_ms)
            || ends_sentence(&word.text);
        if hard || (soft && substantial) {
            flush(&mut current, &mut cues);
        }
    }
    flush(&mut current, &mut cues);
    cues
}

/// Extends short cues toward the minimum duration and the reading-speed limit,
/// without running into the next cue, and removes overlaps.
fn fix_timing(cues: &mut [Cue], rules: &Rules) {
    for i in 0..cues.len() {
        let limit = cues
            .get(i + 1)
            .map(|next| next.start.0.saturating_sub(rules.min_gap_ms))
            .unwrap_or(u64::MAX);
        let cue = &mut cues[i];
        let chars = cue.text.chars().count() as f64;
        let wanted = (rules.min_dur_ms as f64)
            .max(chars / rules.max_cps * 1000.0)
            .ceil() as u64;
        let wanted_end = (cue.start.0 + wanted).min(cue.start.0 + rules.max_dur_ms);
        let end = cue.end.0.max(wanted_end).min(limit);
        cue.end = Timestamp(end.max(cue.start.0 + 1));
    }
}

/// Splits text longer than `max_cpl` into two lines at the space that best
/// balances them, preferring a shorter top line and a break after punctuation.
pub fn break_lines(text: &str, max_cpl: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let total = flat.chars().count();
    if total <= max_cpl {
        return flat;
    }
    let best = flat
        .char_indices()
        .filter(|&(_, c)| c == ' ')
        .min_by_key(|&(byte, _)| {
            let top = flat[..byte].chars().count() as i64;
            let bottom = total as i64 - top - 1;
            // Cost in "characters of imbalance"; a break after punctuation is
            // worth up to 25 characters of imbalance.
            let mut cost = (top - bottom).abs();
            if top > bottom {
                cost += 3; // bottom-heavy reads better
            }
            if top > max_cpl as i64 || bottom > max_cpl as i64 {
                cost += 1000;
            }
            if flat[..byte].ends_with([',', '،', '.', '؟', '?', '!', '…', ';', '؛']) {
                cost -= 25;
            }
            cost
        });
    match best {
        Some((byte, _)) => format!("{}\n{}", &flat[..byte], &flat[byte + 1..]),
        None => flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Word;

    fn word(w: &str, start: f64, end: f64) -> Word {
        Word {
            w: w.into(),
            start,
            end,
            score: None,
        }
    }

    fn transcript(segments: Vec<Segment>) -> Transcript {
        Transcript {
            version: 1,
            audio_s: 60.0,
            model: "test".into(),
            speech: vec![],
            segments,
        }
    }

    fn seg(start: f64, end: f64, text: &str, words: Option<Vec<Word>>) -> Segment {
        Segment {
            start,
            end,
            text: text.into(),
            words,
            align_error: None,
        }
    }

    fn spans(cues: &[Cue]) -> Vec<(u64, u64, &str)> {
        cues.iter()
            .map(|c| (c.start.0, c.end.0, c.text.as_str()))
            .collect()
    }

    #[test]
    fn short_pauses_split_only_substantial_cues() {
        let t = transcript(vec![
            seg(
                1.0,
                4.0,
                "شو بدك هلق",
                Some(vec![
                    word("شو", 1.0, 1.3),
                    word("بدك", 1.35, 1.7),
                    word("هلق", 2.6, 3.0),
                ]),
            ),
            seg(
                5.0,
                6.5,
                "ولا شي",
                Some(vec![word("ولا", 5.0, 5.4), word("شي", 5.45, 6.5)]),
            ),
        ]);
        let cues = cues_from_transcript(&t, &Rules::default());
        assert_eq!(
            spans(&cues),
            vec![(1000, 3000, "شو بدك هلق"), (5000, 6500, "ولا شي")]
        );
    }

    #[test]
    fn long_pauses_always_split() {
        let t = transcript(vec![seg(
            0.0,
            5.0,
            "لا اي",
            Some(vec![word("لا", 0.0, 0.3), word("اي", 2.0, 2.4)]),
        )]);
        let cues = cues_from_transcript(&t, &Rules::default());
        assert_eq!(spans(&cues), vec![(0, 833, "لا"), (2000, 2833, "اي")]);
    }

    #[test]
    fn spreads_words_when_alignment_is_missing_and_drops_tags() {
        let t = transcript(vec![seg(10.0, 12.0, "[موسيقى] اي والله", None)]);
        let cues = cues_from_transcript(&t, &Rules::default());
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "اي والله");
        assert!(cues[0].start.0 > 10_000 && cues[0].end.0 <= 12_000);
    }

    #[test]
    fn caps_characters_and_duration() {
        let words: Vec<Word> = (0..30)
            .map(|i| word("كلمة", i as f64 * 0.4, i as f64 * 0.4 + 0.35))
            .collect();
        let t = transcript(vec![seg(0.0, 12.0, "", Some(words))]);
        let rules = Rules::default();
        let cues = cues_from_transcript(&t, &rules);
        assert!(cues.len() >= 2);
        for cue in &cues {
            assert!(
                cue.text.replace('\n', " ").chars().count() <= 84,
                "{:?}",
                cue.text
            );
            assert!(cue.duration_ms() <= rules.max_dur_ms);
            assert!(cue.text.lines().all(|l| l.chars().count() <= 42));
        }
    }

    #[test]
    fn extends_short_cues_but_keeps_a_gap() {
        // A fast, substantial line followed closely by the next segment: it
        // grows toward its reading-speed time but stops 83 ms before the next.
        let fast: Vec<Word> = ["ما", "بعرف", "شو", "صار", "معك", "يا", "زلمة"]
            .iter()
            .enumerate()
            .map(|(i, w)| word(w, 1.0 + i as f64 * 0.05, 1.04 + i as f64 * 0.05))
            .collect();
        let t = transcript(vec![
            seg(1.0, 1.4, "ما بعرف شو صار معك يا زلمة", Some(fast)),
            seg(1.6, 2.0, "اي", Some(vec![word("اي", 1.6, 2.0)])),
        ]);
        let cues = cues_from_transcript(&t, &Rules::default());
        assert_eq!(
            spans(&cues),
            vec![
                (1000, 1517, "ما بعرف شو صار معك يا زلمة"),
                (1600, 2433, "اي")
            ]
        );
    }

    #[test]
    fn line_breaks_prefer_balance_bottom_heavy_and_punctuation() {
        assert_eq!(break_lines("قصير", 42), "قصير");
        let text = "ما بعرف والله، خليني فكر شوي وبرجع بقلك شو صار معي";
        let broken = break_lines(text, 42);
        assert_eq!(
            broken,
            "ما بعرف والله،\nخليني فكر شوي وبرجع بقلك شو صار معي"
        );
        let even = break_lines("one two three four five six seven eight nine ten", 30);
        let (top, bottom) = even.split_once('\n').unwrap();
        assert!(top.len() <= bottom.len());
    }
}
