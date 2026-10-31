//! Series / season / episode from YouTube titles like
//! «مسلسل باب الحارة الجزء الثاني الحلقة 5 الخامسة كاملة HD» or
//! "Bab Al Harra Season 1 Episode 5".

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Parsed {
    pub series: Option<String>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
}

static EP_DIGITS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:ال)?حلق[هة]\s*[^\p{L}\d]{0,3}(\d{1,3})").unwrap());
static EP_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:ال)?حلق[هة]\s+(\S+(?:\s+\S+)?)").unwrap());
static EP_EN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:episode|ep)\.?\s*(\d{1,3})").unwrap());
static SEASON_DIGITS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:ال)?(?:جزء|موسم)\s*(\d{1,2})").unwrap());
static SEASON_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:ال)?(?:جزء|موسم)\s+(\S+)").unwrap());
static SEASON_EN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:season|part)\s*(\d{1,2})").unwrap());
static SERIES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"مسلسل\s+(.+?)\s*(?:(?:ال)?حلق[هة]|(?:ال)?جزء|(?:ال)?موسم|ـ|-|\||$)").unwrap()
});

pub fn parse(title: &str) -> Parsed {
    let t = normalize(title);
    let episode = EP_DIGITS
        .captures(&t)
        .and_then(|c| c[1].parse().ok())
        .or_else(|| EP_WORD.captures(&t).and_then(|c| ordinal(&c[1])))
        .or_else(|| EP_EN.captures(&t).and_then(|c| c[1].parse().ok()));
    let season = SEASON_DIGITS
        .captures(&t)
        .and_then(|c| c[1].parse().ok())
        .or_else(|| SEASON_WORD.captures(&t).and_then(|c| ordinal(&c[1])))
        .or_else(|| SEASON_EN.captures(&t).and_then(|c| c[1].parse().ok()));
    let series = SERIES
        .captures(&t)
        .map(|c| c[1].trim().to_owned())
        .filter(|s| !s.is_empty());
    Parsed {
        series,
        season,
        episode,
    }
}

/// Folds spelling variants and digits so the patterns above stay simple.
fn normalize(title: &str) -> String {
    title
        .chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{0640}' | '\u{200E}' | '\u{200F}' | '\u{061C}' | '\u{202A}'..='\u{202E}'
            )
        })
        .map(|c| match c {
            'أ' | 'إ' | 'آ' => 'ا',
            'ى' => 'ي',
            '\u{0660}'..='\u{0669}' => char::from(b'0' + (c as u32 - 0x0660) as u8),
            '\u{06F0}'..='\u{06F9}' => char::from(b'0' + (c as u32 - 0x06F0) as u8),
            _ => c,
        })
        .collect()
}

/// Arabic ordinal words (folded spelling) up to 30, e.g. «الخامسة», «الحادية عشرة».
fn ordinal(words: &str) -> Option<u32> {
    const UNITS: [(&str, u32); 10] = [
        ("الاول", 1),
        ("الحادي", 1),
        ("الثاني", 2),
        ("الثالث", 3),
        ("الرابع", 4),
        ("الخامس", 5),
        ("السادس", 6),
        ("السابع", 7),
        ("الثامن", 8),
        ("التاسع", 9),
    ];
    let mut it = words.split_whitespace();
    let first = it.next()?.trim_end_matches(['ه', 'ة']);
    let second = it.next().unwrap_or("");
    if first.starts_with("العاشر") {
        return Some(10);
    }
    if first.starts_with("العشر") {
        return Some(20);
    }
    if first.starts_with("الثلاث") && first.ends_with("ون") {
        return Some(30);
    }
    let unit = UNITS
        .iter()
        .find(|(w, _)| first == *w || first.starts_with(w))?
        .1;
    if second.starts_with("عشر") {
        Some(10 + unit)
    } else if second.starts_with("والعشر") {
        Some(20 + unit)
    } else {
        Some(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(series: Option<&str>, season: Option<u32>, episode: Option<u32>) -> Parsed {
        Parsed {
            series: series.map(str::to_owned),
            season,
            episode,
        }
    }

    #[test]
    fn real_titles() {
        assert_eq!(
            parse("مسلسل غداً نلتقي ـ الحلقة 1 الأولى كاملة HD | Ghadan Naltaqi"),
            p(Some("غداً نلتقي"), None, Some(1))
        );
        assert_eq!(
            parse("مسلسل ضيعة ضايعة - الجزء الأول ـ الحلقة 2 الثانية كاملة HD | الدكان"),
            p(Some("ضيعة ضايعة"), Some(1), Some(2))
        );
        assert_eq!(
            parse("مسلسل باب الحارة الجزء الاول الحلقة 5 الخامسة | Bab Al Harra Season 1 HD"),
            p(Some("باب الحارة"), Some(1), Some(5))
        );
        assert_eq!(
            parse("مسلسل الخربة - الحلقة التاسعة (تشكيلات جديدة) - Al Kherbeh Ep 9"),
            p(Some("الخربة"), None, Some(9))
        );
        assert_eq!(
            parse("Day'a Day'a Series - Season 2 - Episode 1 Full HD | Failed Strategies"),
            p(None, Some(2), Some(1))
        );
    }

    #[test]
    fn ordinals_and_arabic_digits() {
        assert_eq!(parse("الحلقة الحادية عشرة").episode, Some(11));
        assert_eq!(parse("الحلقه الثالثة والعشرون").episode, Some(23));
        assert_eq!(parse("الحلقة العشرون").episode, Some(20));
        assert_eq!(parse("الحلقة ١٢").episode, Some(12));
        assert_eq!(parse("الجزء الثاني").season, Some(2));
        assert_eq!(parse("random video").episode, None);
    }
}
