use crate::core::{error::SRTError, srt::SRT};

/// Enum representing the source type for loading subtitles.
///
/// This enum can represent different types of sources such as:
/// - Media files (e.g., video files)
/// - Subtitle files (e.g., .srt files)
/// - YouTube URLs
#[derive(Debug)]
pub enum SourceInput {
    SubtitleFile(String),
    Youtube(String),
}

pub trait Source {
    /// Loads a file or URL  and returns its content as a vector of strings (lines).
    ///
    /// # Arguments
    ///
    /// * `input` - A string representing the path to the file or URL.
    fn process(input: SourceInput) -> Result<SRT, SRTError>;
}
