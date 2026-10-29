//! SubRip (`.srt`) parsing and writing.
//!
//! The parser is strict about timings but tolerant about layout: it accepts a
//! UTF-8 BOM, CRLF line endings, missing or wrong cue numbers, multi-line text,
//! WebVTT-style `.` millisecond separators, and cues that are not separated by a
//! blank line.

use std::fmt::Write as _;

use crate::{Cue, ParseError, Timestamp};

pub fn parse(input: &str) -> Result<Vec<Cue>, ParseError> {
    let lines: Vec<&str> = strip_bom(input).lines().collect();
    let mut cues = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        // An optional numeric index line comes before the timing line.
        let timing = if is_index(lines[i]) { i + 1 } else { i };
        let Some(timing_line) = lines.get(timing).filter(|l| l.contains("-->")) else {
            return Err(ParseError::MissingTiming { line: timing + 1 });
        };
        let (start, end) = parse_timing(timing_line, timing + 1)?;

        i = timing + 1;
        let mut text = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() && !starts_cue(&lines, i) {
            text.push(lines[i].trim());
            i += 1;
        }
        cues.push(Cue::new(start, end, text.join("\n")));
    }
    Ok(cues)
}

/// Writes cues as SRT, numbering them from 1. Blank lines inside cue text are
/// dropped because they would end the cue early.
pub fn write(cues: &[Cue]) -> String {
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        let _ = writeln!(out, "{}\n{} --> {}", i + 1, cue.start, cue.end);
        for line in cue.text.lines().filter(|l| !l.trim().is_empty()) {
            let _ = writeln!(out, "{line}");
        }
        out.push('\n');
    }
    out
}

pub(crate) fn strip_bom(input: &str) -> &str {
    input.strip_prefix('\u{feff}').unwrap_or(input)
}

/// Parses `start --> end [settings]`, ignoring WebVTT cue settings after `end`.
pub(crate) fn parse_timing(
    line: &str,
    lineno: usize,
) -> Result<(Timestamp, Timestamp), ParseError> {
    let (a, b) = line
        .split_once("-->")
        .ok_or(ParseError::MissingTiming { line: lineno })?;
    let parse = |s: &str| {
        s.parse::<Timestamp>().map_err(|source| ParseError::Time {
            line: lineno,
            source,
        })
    };
    let start = parse(a)?;
    let end = parse(b.split_whitespace().next().unwrap_or(""))?;
    if end < start {
        return Err(ParseError::EndBeforeStart { line: lineno });
    }
    Ok((start, end))
}

pub(crate) fn is_timing(line: &str) -> bool {
    parse_timing(line, 0).is_ok()
}

fn is_index(line: &str) -> bool {
    let line = line.trim();
    !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit())
}

/// True when line `i` begins a new cue: a valid timing line, or an index line
/// directly followed by one. Lets us split cues that lack a blank separator.
fn starts_cue(lines: &[&str], i: usize) -> bool {
    is_timing(lines[i]) || (is_index(lines[i]) && lines.get(i + 1).is_some_and(|l| is_timing(l)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: u64, end: u64, text: &str) -> Cue {
        Cue::new(Timestamp(start), Timestamp(end), text)
    }

    #[test]
    fn keeps_multi_line_text() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n\n2\n00:00:03,000 --> 00:00:04,000\nthree\n";
        assert_eq!(
            parse(srt).unwrap(),
            vec![
                cue(1000, 2000, "line one\nline two"),
                cue(3000, 4000, "three")
            ]
        );
    }

    #[test]
    fn handles_bom_crlf_and_missing_trailing_newline() {
        let srt = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,000\r\nمرحبا\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nأهلين";
        assert_eq!(
            parse(srt).unwrap(),
            vec![cue(1000, 2000, "مرحبا"), cue(3000, 4000, "أهلين")]
        );
    }

    #[test]
    fn splits_cues_without_blank_separator() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nHello, world!\n2\n00:00:05,000 --> 00:00:08,000\nThis is a test.";
        assert_eq!(
            parse(srt).unwrap(),
            vec![
                cue(1000, 4000, "Hello, world!"),
                cue(5000, 8000, "This is a test.")
            ]
        );
    }

    #[test]
    fn numbers_inside_text_are_text() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nHow many?\n42\n";
        assert_eq!(parse(srt).unwrap(), vec![cue(1000, 2000, "How many?\n42")]);
    }

    #[test]
    fn index_line_is_optional_and_ignored() {
        let srt = "00:00:01,000 --> 00:00:02,000\na\n\n7\n00:00:03.500 --> 00:00:04,000\nb\n";
        assert_eq!(
            parse(srt).unwrap(),
            vec![cue(1000, 2000, "a"), cue(3500, 4000, "b")]
        );
    }

    #[test]
    fn keeps_empty_cues() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nb\n";
        assert_eq!(
            parse(srt).unwrap(),
            vec![cue(1000, 2000, ""), cue(3000, 4000, "b")]
        );
    }

    #[test]
    fn reports_errors_with_line_numbers() {
        assert_eq!(
            parse("1\n00:00:01,000 --> 00:00:02,000\na\n\nstray text\n"),
            Err(ParseError::MissingTiming { line: 5 })
        );
        assert_eq!(
            parse("1\nnot a timing\n"),
            Err(ParseError::MissingTiming { line: 2 })
        );
        assert!(matches!(
            parse("1\n00:00:01,000 --> 00:00:0x,000\na\n"),
            Err(ParseError::Time { line: 2, .. })
        ));
        assert_eq!(
            parse("1\n00:00:05,000 --> 00:00:02,000\na\n"),
            Err(ParseError::EndBeforeStart { line: 2 })
        );
    }

    #[test]
    fn writes_and_round_trips() {
        let cues = vec![cue(1000, 2000, "line one\nline two"), cue(3000, 4500, "")];
        let out = write(&cues);
        assert_eq!(
            out,
            "1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n\n2\n00:00:03,000 --> 00:00:04,500\n\n"
        );
        assert_eq!(parse(&out).unwrap(), cues);
    }

    #[test]
    fn writer_drops_blank_lines_inside_text() {
        let out = write(&[cue(0, 1000, "a\n\nb")]);
        assert_eq!(parse(&out).unwrap(), vec![cue(0, 1000, "a\nb")]);
    }
}
