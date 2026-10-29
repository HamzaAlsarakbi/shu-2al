//! Regression tests against real subtitle files.

use subs::filter::{self, Reason, Rules};
use subs::{Timestamp, srt, vtt};

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Whisper large-v3 output for an Arabic song: 390 cues including empty cues,
/// a "jungle" music hallucination, a credit line and a decoding loop.
#[test]
fn whisper_output_parses_and_cleans() {
    let cues = srt::parse(&fixture("whisper_arabic_song.srt")).unwrap();
    assert_eq!(cues.len(), 390);
    assert_eq!(cues[5].text, "jungle");

    let (kept, dropped) = filter::apply(cues, &Rules::default());
    let count = |r: Reason| dropped.iter().filter(|(_, reason)| *reason == r).count();
    assert_eq!(count(Reason::Empty), 50, "40 empty + 10 punctuation-only");
    assert_eq!(
        count(Reason::Phrase),
        4,
        "jungle, 2× credit line, patch loop"
    );
    assert_eq!(kept.len(), 336);
    assert!(
        kept.iter()
            .all(|c| !c.text.contains("نانسي") && !c.text.contains("patch"))
    );
    assert!(kept.iter().any(|c| c.text == "قلبي قلبي علينا"));
}

#[test]
fn whisper_output_round_trips_through_srt_and_vtt() {
    let cues = srt::parse(&fixture("whisper_arabic_song.srt")).unwrap();
    assert_eq!(srt::parse(&srt::write(&cues)).unwrap(), cues);
    assert_eq!(vtt::parse(&vtt::write(&cues)).unwrap(), cues);
}

#[test]
fn cues_without_blank_separators() {
    let cues = srt::parse(&fixture("no_blank_lines.srt")).unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[1].start, Timestamp(5000));
    assert_eq!(cues[1].text, "This is a test.");
}

#[test]
fn simple_file() {
    let cues = srt::parse(&fixture("simple.srt")).unwrap();
    assert_eq!(
        cues.iter()
            .map(|c| (c.end.0, c.text.as_str()))
            .collect::<Vec<_>>(),
        vec![(30_000, "Hello"), (42_600, "World")]
    );
}
