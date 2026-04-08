use std::sync::{Arc, Mutex};

use crate::{core::srt::SRT, modules::module::Module};

/// Module for filtering subtitles based on specific criteria.
///
/// This module can be used to remove empty lines or filter out subtitles containing specific words.
///
/// # Fields
/// * `enabled` - A boolean indicating whether the filter module is enabled.
/// * `remove_empty_lines` - A boolean indicating whether to remove empty lines from the subtitles.
/// * `words_list` - A list of words to filter out from the subtitles. If a subtitle contains any of these words, it will be removed.
///
/// # Example
/// ```
/// let filter_module = FilterModule {
///     enabled: true,
///     remove_empty_lines: true,
///     words_list: vec!["test".to_string()],
/// };
/// ```
pub struct FilterModule {
    pub enabled: bool,
    /// Whether to remove empty lines from the subtitles.
    pub remove_empty_lines: bool,
    /// A list of words to filter out from the subtitles. If a subtitle contains any of these words, it will be removed.
    pub words_list: Vec<String>,
}

impl Module for FilterModule {
    fn process(
        &self,
        input: Arc<Mutex<SRT>>,
    ) -> Result<Arc<Mutex<SRT>>, crate::core::error::SRTError> {
        if !self.enabled {
            return Ok(input);
        }

        let mut lock = input.lock().unwrap();
        lock.subtitles.retain(|subtitle| {
            if self.remove_empty_lines && subtitle.text.trim().is_empty() {
                return false;
            }

            for word in &self.words_list {
                if subtitle.text.contains(word) {
                    return false;
                }
            }

            true
        });

        // Re-index subtitles after filtering
        for (index, subtitle) in lock.subtitles.iter_mut().enumerate() {
            subtitle.index = index + 1;
        }

        drop(lock);
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::{
        core::{srt::SRT, subtitle::Subtitle, timestamp::Timestamp},
        modules::{filter::FilterModule, module::Module},
    };

    #[test]
    fn test_filter_module() {
        let subtitles = vec![
            Subtitle {
                index: 1,
                start_time: Timestamp::from_string("00:00:01,000").unwrap(),
                end_time: Timestamp::from_string("00:00:05,000").unwrap(),
                text: "Hello, World!".to_string(),
            },
            Subtitle {
                index: 2,
                start_time: Timestamp::from_string("00:00:06,000").unwrap(),
                end_time: Timestamp::from_string("00:00:10,000").unwrap(),
                text: "This is a test.".to_string(),
            },
            Subtitle {
                index: 3,
                start_time: Timestamp::from_string("00:00:11,000").unwrap(),
                end_time: Timestamp::from_string("00:00:15,000").unwrap(),
                text: "Another line.".to_string(),
            },
        ];

        let srt = SRT {
            subtitles: subtitles.clone(),
            file_path: "test.srt".to_string(),
        };

        let filter_module = FilterModule {
            enabled: true,
            remove_empty_lines: true,
            words_list: vec!["test".to_string()],
        };

        let input = Arc::new(Mutex::new(srt));
        let result = filter_module.process(input.clone()).unwrap();

        assert_eq!(result.lock().unwrap().subtitles.len(), 2);
        assert_eq!(result.lock().unwrap().subtitles[0].text, "Hello, World!");
        assert_eq!(result.lock().unwrap().subtitles[0].index, 1);
        assert_eq!(result.lock().unwrap().subtitles[1].text, "Another line.");
        assert_eq!(result.lock().unwrap().subtitles[1].index, 2);
    }
}
