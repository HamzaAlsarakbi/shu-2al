use crate::ParseTimeError;

/// A subtitle file that could not be parsed. Line numbers are 1-based.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("missing WEBVTT header")]
    MissingHeader,
    #[error("line {line}: expected a timing line like `00:00:01,000 --> 00:00:02,000`")]
    MissingTiming { line: usize },
    #[error("line {line}: {source}")]
    Time {
        line: usize,
        #[source]
        source: ParseTimeError,
    },
    #[error("line {line}: cue ends before it starts")]
    EndBeforeStart { line: usize },
}
