//! WebVTT (`.vtt`) parsing and writing, for the browser editor's `<track>`.
//!
//! Supports cue identifiers, cue settings (ignored), and skips NOTE, STYLE and
//! REGION blocks. Cue text keeps its tags; only `&amp;`, `&lt;` and `&gt;` are
//! unescaped.

use std::fmt::Write as _;

use crate::srt::{parse_timing, strip_bom};
use crate::{Cue, ParseError};

pub fn parse(input: &str) -> Result<Vec<Cue>, ParseError> {
    let input = strip_bom(input);
    let first = input.lines().next().unwrap_or("");
    if first != "WEBVTT" && !first.starts_with("WEBVTT ") && !first.starts_with("WEBVTT\t") {
        return Err(ParseError::MissingHeader);
    }

    let mut cues = Vec::new();
    let mut block: Vec<(usize, &str)> = Vec::new();
    // Skip the header block, then handle one blank-line-separated block at a time.
    for (lineno, line) in input
        .lines()
        .enumerate()
        .skip_while(|(_, l)| !l.trim().is_empty())
    {
        if line.trim().is_empty() {
            push_block(&block, &mut cues)?;
            block.clear();
        } else {
            block.push((lineno + 1, line));
        }
    }
    push_block(&block, &mut cues)?;
    Ok(cues)
}

fn push_block(block: &[(usize, &str)], cues: &mut Vec<Cue>) -> Result<(), ParseError> {
    let Some(&(first_no, first)) = block.first() else {
        return Ok(());
    };
    if ["NOTE", "STYLE", "REGION"].iter().any(|kw| {
        first == *kw
            || first.starts_with(&format!("{kw} "))
            || first.starts_with(&format!("{kw}\t"))
    }) {
        return Ok(());
    }
    // An identifier line may precede the timing line.
    let timing_at = usize::from(!first.contains("-->"));
    let Some(&(lineno, timing)) = block.get(timing_at).filter(|(_, l)| l.contains("-->")) else {
        return Err(ParseError::MissingTiming {
            line: first_no + timing_at,
        });
    };
    let (start, end) = parse_timing(timing, lineno)?;
    let text: Vec<String> = block[timing_at + 1..]
        .iter()
        .map(|(_, l)| unescape(l.trim()))
        .collect();
    cues.push(Cue::new(start, end, text.join("\n")));
    Ok(())
}

pub fn write(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for cue in cues {
        let _ = writeln!(out, "{} --> {}", cue.start.to_vtt(), cue.end.to_vtt());
        for line in cue.text.lines().filter(|l| !l.trim().is_empty()) {
            let _ = writeln!(out, "{}", escape(line));
        }
        out.push('\n');
    }
    out
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Timestamp;

    fn cue(start: u64, end: u64, text: &str) -> Cue {
        Cue::new(Timestamp(start), Timestamp(end), text)
    }

    #[test]
    fn parses_ids_settings_notes_and_short_timestamps() {
        let vtt = "WEBVTT - shu2al\n\nNOTE written by hand\nspanning lines\n\nintro\n00:01.000 --> 00:02.500 align:start\nمرحبا\nيا جماعة\n\n00:00:03.000 --> 00:00:04.000\nTom &amp; Jerry\n";
        assert_eq!(
            parse(vtt).unwrap(),
            vec![
                cue(1000, 2500, "مرحبا\nيا جماعة"),
                cue(3000, 4000, "Tom & Jerry")
            ]
        );
    }

    #[test]
    fn requires_header() {
        assert_eq!(
            parse("00:01.000 --> 00:02.000\nhi\n"),
            Err(ParseError::MissingHeader)
        );
        assert_eq!(parse("WEBVTTX\n"), Err(ParseError::MissingHeader));
    }

    #[test]
    fn reports_missing_timing_line() {
        assert_eq!(
            parse("WEBVTT\n\nid\nno timing here\n"),
            Err(ParseError::MissingTiming { line: 4 })
        );
    }

    #[test]
    fn writes_escaped_and_round_trips() {
        let cues = vec![
            cue(1000, 2000, "a < b & c\nسطر ثاني"),
            cue(3_723_456, 3_724_000, "x"),
        ];
        let out = write(&cues);
        assert!(out.starts_with("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\na &lt; b &amp; c\n"));
        assert_eq!(parse(&out).unwrap(), cues);
    }
}
