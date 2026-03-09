use crate::source::source::SourceInput;

#[derive(Debug)]
pub enum SRTError {
    InvalidSourceType(SourceInput, SourceInput),
    SubtitleParseError(String),
    FileError(String),
    InvalidInput(String),
    Unknown,
}

impl std::fmt::Display for SRTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SRTError::InvalidSourceType(expected, actual) => write!(
                f,
                "Invalid source type: expected {:?}, but got {:?}",
                expected, actual
            ),
            SRTError::FileError(msg) => write!(f, "File error: {}", msg),
            SRTError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            SRTError::SubtitleParseError(msg) => write!(f, "Subtitle parse error: {}", msg),
            SRTError::Unknown => write!(f, "An unknown error occurred"),
        }
    }
}

impl std::error::Error for SRTError {}
