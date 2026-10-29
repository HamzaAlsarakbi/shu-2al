//! Drops cues that are almost certainly ASR hallucinations: empty or
//! punctuation-only text, known credit/subscribe phrases, and decoding loops.
//!
//! Phrase matching works on folded whole words and only drops a cue when the
//! phrase *dominates* it, so real dialogue that merely mentions «الموسيقى» or
//! "jungle" survives.

use crate::Cue;
use crate::normalize::fold;

/// Phrases ASR models emit over music, silence or credits, learned from YouTube
/// fan subtitles.
pub const DEFAULT_PHRASES: &[&str] = &[
    "اشتركوا في القناة",
    "اشتركوا بالقناة",
    "لا تنسوا الاشتراك في القناة",
    "لا تنسوا الاشتراك",
    "المترجم للقناة",
    "ترجمة نانسي قنقر",
    "موسيقى",
    "jungle",
    "patch",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Rules {
    pub phrases: Vec<String>,
    /// Drop a cue when known phrases cover at least this share (0–1) of its letters.
    pub dominance: f32,
    /// Drop a cue when a 1–3 word pattern repeats this many times in a row.
    pub max_repeats: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            phrases: DEFAULT_PHRASES.iter().map(|p| (*p).to_owned()).collect(),
            dominance: 0.7,
            max_repeats: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    Empty,
    Phrase,
    Repetition,
}

/// Why `text` should be dropped, or `None` to keep it.
pub fn check(text: &str, rules: &Rules) -> Option<Reason> {
    let folded = fold(text);
    let words: Vec<&str> = folded.split(' ').filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return Some(Reason::Empty);
    }
    if phrase_coverage(&words, &rules.phrases) >= rules.dominance {
        return Some(Reason::Phrase);
    }
    if has_loop(&words, rules.max_repeats) {
        return Some(Reason::Repetition);
    }
    None
}

/// Splits cues into kept and dropped (with reasons), preserving order.
pub fn apply(cues: Vec<Cue>, rules: &Rules) -> (Vec<Cue>, Vec<(Cue, Reason)>) {
    let mut kept = Vec::with_capacity(cues.len());
    let mut dropped = Vec::new();
    for cue in cues {
        match check(&cue.text, rules) {
            Some(reason) => dropped.push((cue, reason)),
            None => kept.push(cue),
        }
    }
    (kept, dropped)
}

/// Share of the cue's letters covered by whole-word phrase matches.
fn phrase_coverage(words: &[&str], phrases: &[String]) -> f32 {
    let mut covered = vec![false; words.len()];
    for phrase in phrases {
        let folded = fold(phrase);
        let phrase: Vec<&str> = folded.split(' ').filter(|w| !w.is_empty()).collect();
        if phrase.is_empty() || phrase.len() > words.len() {
            continue;
        }
        for start in 0..=words.len() - phrase.len() {
            if words[start..start + phrase.len()] == phrase[..] {
                covered[start..start + phrase.len()].fill(true);
            }
        }
    }
    let len = |w: &&str| w.chars().count();
    let total: usize = words.iter().map(len).sum();
    let hit: usize = words
        .iter()
        .zip(&covered)
        .filter(|(_, c)| **c)
        .map(|(w, _)| len(w))
        .sum();
    hit as f32 / total as f32
}

/// True when some 1–3 word n-gram repeats `max_repeats` times back to back.
fn has_loop(words: &[&str], max_repeats: usize) -> bool {
    if max_repeats < 2 {
        return false;
    }
    (1..=3).any(|n| {
        (0..words.len()).any(|start| {
            let gram = &words[start..(start + n).min(words.len())];
            let mut count = 1;
            while words.get(start + count * n..start + (count + 1) * n) == Some(gram) {
                count += 1;
            }
            gram.len() == n && count >= max_repeats
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_default(text: &str) -> Option<Reason> {
        check(text, &Rules::default())
    }

    #[test]
    fn drops_empty_and_punctuation_only() {
        assert_eq!(check_default(""), Some(Reason::Empty));
        assert_eq!(check_default("  ...  "), Some(Reason::Empty));
        assert_eq!(check_default("♪ ♪"), Some(Reason::Empty));
    }

    #[test]
    fn drops_dominant_phrases_in_any_spelling() {
        assert_eq!(check_default("موسيقى"), Some(Reason::Phrase));
        assert_eq!(check_default("[موسيقى]"), Some(Reason::Phrase));
        assert_eq!(check_default("ترجمة نانسي قنقر"), Some(Reason::Phrase));
        assert_eq!(check_default("اشتركوا في القناةْ!"), Some(Reason::Phrase));
        assert_eq!(check_default("Jungle"), Some(Reason::Phrase));
    }

    #[test]
    fn keeps_dialogue_that_mentions_a_phrase() {
        assert_eq!(check_default("صوت الموسيقى عالي كتير"), None);
        assert_eq!(check_default("موسيقى هالمسلسل حلوة كتير والله"), None);
        assert_eq!(check_default("jungle gym"), None);
        assert_eq!(check_default("we need a patch for the roof"), None);
    }

    #[test]
    fn drops_decoding_loops_but_not_emphasis() {
        assert_eq!(check_default(&"patch ".repeat(50)), Some(Reason::Phrase));
        assert_eq!(check_default(&"يا ".repeat(12)), Some(Reason::Repetition));
        assert_eq!(
            check_default(&"شو بدك ".repeat(8)),
            Some(Reason::Repetition)
        );
        assert_eq!(check_default("لا لا لا لا"), None);
        assert_eq!(check_default("قلبي قلبي علينا"), None);
    }

    #[test]
    fn apply_preserves_order() {
        use crate::Timestamp;
        let cue = |t: u64, text: &str| Cue::new(Timestamp(t), Timestamp(t + 1), text);
        let (kept, dropped) = apply(
            vec![
                cue(0, "أهلين"),
                cue(1, ""),
                cue(2, "موسيقى"),
                cue(3, "كيفك"),
            ],
            &Rules::default(),
        );
        assert_eq!(kept, vec![cue(0, "أهلين"), cue(3, "كيفك")]);
        assert_eq!(
            dropped
                .iter()
                .map(|(c, r)| (c.start.0, *r))
                .collect::<Vec<_>>(),
            vec![(1, Reason::Empty), (2, Reason::Phrase)]
        );
    }
}
