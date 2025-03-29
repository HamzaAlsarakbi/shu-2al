use std::time::Duration;

use super::{direction::Direction, timestamp::Timestamp};

/// This module provides functionality to clean and format SRT (SubRip Subtitle) files.
/// It includes functions to read SRT files, remove empty lines, and format the subtitles.

/// Subtitle struct
/// Represents a subtitle entry with start time, end time, and text.
#[derive(Debug, Clone, PartialEq)]
pub struct Subtitle {
    /// Start time of the subtitle in the format "HH:MM:SS,ms"
    start_time: Timestamp,
    /// End time of the subtitle in the format "HH:MM:SS,ms"
    end_time: Timestamp,
    /// Text of the subtitle
    text: String,
}

#[cfg(test)]
impl Default for Subtitle {
    fn default() -> Self {
        Subtitle {
            start_time: Timestamp::from_string("00:00:01,000").unwrap(),
            end_time: Timestamp::from_string("00:00:05,000").unwrap(),
            text: "Hello, World!".to_string(),
        }
    }
}

const WORDS_LIST: [&str; 6] = [
    "شتركوا في القناة",
    "لا تنسوا الاشتراك في القناة",
    "لا تنسوا الاشتراك",
    "المترجم للقناة",
    "موسيقى",
    "patch",
];

impl Subtitle {
    /// Creates a new `Subtitle` instance from a slice of strings.
    /// The first line is the index, the second line contains the start and end time,
    /// and the following line contain the text.
    ///
    /// # Arguments
    ///
    /// * `lines` - A slice of strings representing the lines of a subtitle block.
    ///
    /// # Returns
    ///
    ///  * `Result<Subtitle, String>` - Returns a `Subtitle` instance if successful, or an error message if it fails.
    pub fn new(lines: &Vec<&str>) -> Result<Self, String> {
        // find index of the line with the start and end time
        let ts_i = lines
            .iter()
            .position(|&line| line.contains("-->"))
            .ok_or("No timestamp found")?;
        if ts_i + 1 >= lines.len() {
            return Err("No text provided".to_owned());
        }

        let start_time = lines[ts_i]
            .split(" --> ")
            .next()
            .ok_or("Invalid start timestamp")?
            .to_string();
        let end_time = lines[ts_i]
            .split(" --> ")
            .nth(1)
            .ok_or("Invalid end timestamp")?
            .to_string();
        let text = lines[ts_i + 1].trim().to_string();

        let subtitle = Subtitle {
            start_time: Timestamp::from_string(&start_time)?,
            end_time: Timestamp::from_string(&end_time)?,
            text,
        };

        if !subtitle.is_valid() {
            return Err("Invalid subtitle".to_owned());
        }

        Ok(subtitle)
    }

    /// Converts the `Subtitle` instance to a string representation.
    /// The format is:
    /// ```
    /// 00:00:01,000 --> 00:00:05,000
    /// Hello, World!
    ///
    /// ```
    ///
    /// # Returns
    ///
    /// * `String` - The string representation of the subtitle.
    ///
    /// # Notes
    ///
    /// You should add the number of the subtitle before the start time.
    /// For example:
    /// ```
    /// 1
    /// 00:00:01,000 --> 00:00:05,000
    /// Hello, World!
    ///
    /// ```
    pub fn to_string(&self) -> String {
        format!("{} --> {}\n{}\n", self.start_time, self.end_time, self.text)
    }

    /// Checks if the subtitle is valid.
    /// A subtitle is considered valid if it has a non-empty start time, end time, and text.
    ///
    /// # Returns
    ///
    /// * `bool` - Returns `true` if the subtitle is valid, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        !self.text.is_empty()
            && !WORDS_LIST.iter().any(|&word| self.text.contains(word))
            // and text isn't made up of special characters
            && !self.text.chars().all(|c| c.is_ascii_punctuation())
    }

    pub fn duration(&self) -> Duration {
        let start_time = self.start_time.to_millis();
        let end_time = self.end_time.to_millis();
        Duration::from_millis(end_time - start_time)
    }

    pub fn move_start(&mut self, delta: Duration, direction: Direction) -> Result<(), String> {
        self.start_time.move_ts(delta, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_new() {
        let lines = vec!["00:00:01,000 --> 00:00:05,000", "Hello, World!"];
        let subtitle = Subtitle::new(&lines).unwrap();
        assert_eq!(
            subtitle.start_time,
            Timestamp::from_string("00:00:01,000").unwrap()
        );
        assert_eq!(
            subtitle.end_time,
            Timestamp::from_string("00:00:05,000").unwrap()
        );
        assert_eq!(subtitle.text, "Hello, World!");
    }

    #[test]
    fn test_subtitle_new_invalid() {
        assert!(Subtitle::new(&vec!["1"]).is_err());
        assert!(Subtitle::new(&vec!["1", ""]).is_err());
        assert!(Subtitle::new(&vec!["", "1"]).is_err());
        assert!(Subtitle::new(&vec!["", ""]).is_err());

        let lines = vec!["1", "00:00:01,000 --> 00:00:05,000"];
        assert!(Subtitle::new(&lines).is_err());

        let lines = vec!["1", "00:00:01,000 --> 00:00:05,000", "Hello, World!"];
        assert!(Subtitle::new(&lines).is_ok());

        let lines = vec![
            "1",
            "00:00:01,000 --> 00:00:05,000",
            "Hello, World!",
            "Extra line",
        ];
        assert!(Subtitle::new(&lines).is_ok());

        let lines = vec!["Hello, World!", "00:00:01,000 --> 00:00:05,000"];
        assert!(Subtitle::new(&lines).is_err());

        let lines = vec!["Hello, World!", ""];
        assert!(Subtitle::new(&lines).is_err());

        let lines = vec!["00:00:01,000 --> 00:00:05,000", ""];
        assert!(Subtitle::new(&lines).is_err());

        let lines = vec!["", "Hello, World!"];
        assert!(Subtitle::new(&lines).is_err());
    }

    #[test]
    fn test_subtitle_to_string() {
        let subtitle = Subtitle {
            start_time: Timestamp::from_string("00:00:01,000").unwrap(),
            end_time: Timestamp::from_string("00:00:05,000").unwrap(),
            text: "Hello, World!".to_string(),
        };
        assert_eq!(
            subtitle.to_string(),
            "00:00:01,000 --> 00:00:05,000\nHello, World!\n"
        );
    }

    #[test]
    fn test_subtitle_is_valid() {
        let valid_subtitle = Subtitle {
            text: "Hello, World!".to_string(),
            ..Default::default()
        };
        assert!(valid_subtitle.is_valid());
    }

    #[test]
    fn test_subtitle_bad_text() {
        let invalid_subtitle = Subtitle {
            text: "".to_string(),
            ..Default::default()
        };
        assert!(!invalid_subtitle.is_valid());

        let invalid_subtitle = Subtitle {
            text: "شتركوا في القناة".to_string(),
            ..Default::default()
        };
        assert!(!invalid_subtitle.is_valid());

        let invalid_subtitle = Subtitle {
            text: ",".to_string(),
            ..Default::default()
        };
        assert!(!invalid_subtitle.is_valid());

        let invalid_subtitle = Subtitle {
            text: ".".to_string(),
            ..Default::default()
        };
        assert!(!invalid_subtitle.is_valid());
    }

    #[test]
    fn test_subtitle_duration() {
        let subtitle = Subtitle {
            start_time: Timestamp::from_string("00:00:01,000").unwrap(),
            end_time: Timestamp::from_string("00:00:05,000").unwrap(),
            text: "Hello, World!".to_string(),
        };
        assert_eq!(subtitle.duration(), Duration::new(4, 0));
    }
}
