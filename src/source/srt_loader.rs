use crate::core::error::SRTError;
use crate::core::srt::SRT;

use super::source::{Source, SourceInput};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    use crate::{
        core::{subtitle::Subtitle, timestamp::Timestamp},
        source::{
            source::{Source, SourceInput},
            srt_loader::SRTLoader,
        },
    };

    fn test_srt_loader() {
        let subtitles = vec![
            Subtitle {
                index: 1,
                start_time: Timestamp::from_string("00:00:01,000").unwrap(),
                end_time: Timestamp::from_string("00:00:04,000").unwrap(),
                text: "Hello, world!".to_string(),
            },
            Subtitle {
                index: 2,
                start_time: Timestamp::from_string("00:00:05,000").unwrap(),
                end_time: Timestamp::from_string("00:00:08,000").unwrap(),
                text: "This is a test.".to_string(),
            },
        ];

        let result = SRTLoader::process(SourceInput::SubtitleFile(
            "test_files/srt_loader/test1.srt".to_string(),
        ))
        .unwrap();

        assert_eq!(result.subtitles, subtitles);
    }
}
