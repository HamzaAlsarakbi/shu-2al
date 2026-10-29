//! Arabic-aware text folding for matching and error-rate scoring. The output is
//! for comparison only, never for display.

/// Folds text so spelling variants compare equal:
/// - drops harakat, superscript alef, Quranic marks, tatweel, bidi and zero-width
///   controls;
/// - unifies alef forms (أ إ آ ٱ → ا), alef maqsura (ى → ي) and ta marbuta (ة → ه);
/// - maps Arabic-Indic and Persian digits to ASCII and lowercases Latin;
/// - turns punctuation into spaces and collapses whitespace.
pub fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars().filter(|&c| !is_ignorable(c)) {
        let c = match c {
            'أ' | 'إ' | 'آ' | 'ٱ' => 'ا',
            'ى' => 'ي',
            'ة' => 'ه',
            '\u{0660}'..='\u{0669}' => char::from(b'0' + (c as u32 - 0x0660) as u8),
            '\u{06F0}'..='\u{06F9}' => char::from(b'0' + (c as u32 - 0x06F0) as u8),
            _ => c,
        };
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Characters removed outright, so they never split a word.
fn is_ignorable(c: char) -> bool {
    matches!(c,
        '\u{064B}'..='\u{065F}'   // harakat, tanwin, shadda, sukun
        | '\u{0670}'              // superscript alef
        | '\u{06D6}'..='\u{06ED}' // Quranic annotation marks
        | '\u{0640}'              // tatweel
        | '\u{061C}'              // Arabic letter mark
        | '\u{200B}'..='\u{200F}' // zero-width chars, LRM, RLM
        | '\u{202A}'..='\u{202E}' // bidi embeddings and overrides
        | '\u{2066}'..='\u{2069}' // bidi isolates
        | '\u{FEFF}'              // BOM / zero-width no-break space
    )
}

#[cfg(test)]
mod tests {
    use super::fold;

    #[test]
    fn strips_marks_and_tatweel() {
        assert_eq!(fold("مَرْحَبًا"), "مرحبا");
        assert_eq!(fold("يـــا"), "يا");
        assert_eq!(fold("\u{202B}شو\u{202C}"), "شو");
    }

    #[test]
    fn unifies_letter_variants() {
        assert_eq!(fold("أإآٱ"), "اااا");
        assert_eq!(fold("موسيقى"), "موسيقي");
        assert_eq!(fold("حلوة"), "حلوه");
    }

    #[test]
    fn digits_case_and_punctuation() {
        assert_eq!(fold("الحلقة ١٢"), "الحلقه 12");
        assert_eq!(fold("۳"), "3");
        assert_eq!(fold("  Yalla,  HABIBI!! ؟ "), "yalla habibi");
        assert_eq!(fold("شو؟ ليش،  هيك"), "شو ليش هيك");
        assert_eq!(fold("..."), "");
    }
}
