use crate::core::error::SRTError;
use crate::core::srt::SRT;

use super::source::{Source, SourceInput};

pub struct SRTLoader;

impl Source for SRTLoader {
    /// Loads a subtitle file and returns its content as a vector of strings (lines).
    fn process(input: SourceInput) -> Result<SRT, SRTError> {
        match input {
            SourceInput::SubtitleFile(path) => {
                let mut content = SRT::new(&path);
                content.read_file()?;

                Ok(content)
            }
            _ => Err(SRTError::InvalidSourceType(
                SourceInput::SubtitleFile("".to_string()),
                input,
            )),
        }
    }
}

// TODO: Tests
mod tests {
    use crate::core::srt::SRT;

    fn test_srt_loader() {
        let subtitles = vec![
            
        ]
        let loaded = SRTLoader::process(SourceInput::SubtitleFile("examples/sample.srt".to_string()));
    }
}