//! Error rates for comparing transcripts. Inputs are folded first (see
//! [`crate::normalize::fold`]) so spelling variants and punctuation don't count.

use crate::normalize::fold;

/// Character error rate of `hypothesis` against `reference`: edit distance over
/// the reference length, after folding. 0 is identical; can exceed 1.
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = fold(reference).chars().collect();
    let h: Vec<char> = fold(hypothesis).chars().collect();
    rate(&r, &h)
}

/// Word error rate, after folding.
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let (r, h) = (fold(reference), fold(hypothesis));
    let r: Vec<&str> = r.split_whitespace().collect();
    let h: Vec<&str> = h.split_whitespace().collect();
    rate(&r, &h)
}

fn rate<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(reference, hypothesis) as f64 / reference.len() as f64
}

fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(x != y);
            cur[j + 1] = substitution.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_after_folding_is_zero() {
        assert_eq!(cer("شو بدك تعمل هلق؟", "شو بدّك تعمل هلق"), 0.0);
        assert_eq!(wer("كثيرة", "كثيره"), 0.0);
    }

    #[test]
    fn counts_edits() {
        assert_eq!(levenshtein(&['a', 'b', 'c'], &['a', 'x', 'c', 'd']), 2);
        assert!((cer("كتير", "كثير") - 0.25).abs() < 1e-9);
        assert!((wer("ما بعرف والله", "ما بعرف") - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(cer("", ""), 0.0);
        assert_eq!(cer("", "x"), 1.0);
    }
}
