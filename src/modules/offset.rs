use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    core::{direction::Direction, srt::SRT},
    modules::module::Module,
};

/// Offset module settings
pub struct OffsetModule {
    /// Whether the module is enabled or not.
    enabled: bool,
    /// Time offset to apply to the subtitle timings.
    offset: Duration,
    /// Direction of the offset (forward or backward).
    direction: Direction,
}

impl Module for OffsetModule {
    /// Applies a time offset to the subtitle timings.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an SRT object.
    ///
    /// # Returns
    ///
    /// * `Result<&SRT, SRTError>` - Returns a reference to the processed SRT object if successful, or an error message if it fails.
    fn process(
        &self,
        input: Arc<Mutex<SRT>>,
    ) -> Result<Arc<Mutex<SRT>>, crate::core::error::SRTError> {
        if !self.enabled {
            return Ok(input);
        }

        let mut lock = input.lock().unwrap();
        for subtitle in &mut lock.subtitles {
            subtitle.start_time.move_ts(&self.offset, &self.direction)?;
            subtitle.end_time.move_ts(&self.offset, &self.direction)?;
        }
        drop(lock);
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use crate::{
        core::{direction::Direction, srt::SRT, subtitle::Subtitle, timestamp::Timestamp},
        modules::{module::Module, offset::OffsetModule},
    };

    #[test]
    fn test_offset_module() {
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
        ];

        let srt = SRT {
            subtitles: subtitles.clone(),
            file_path: "test.srt".to_string(),
        };

        let offset_module = OffsetModule {
            enabled: true,
            offset: Duration::from_secs(2),
            direction: Direction::Forward,
        };

        let input = Arc::new(Mutex::new(srt));
        let result = offset_module.process(input.clone()).unwrap();

        assert_eq!(
            result.lock().unwrap().subtitles[0].start_time,
            Timestamp::from_string("00:00:03,000").unwrap()
        );
        assert_eq!(
            result.lock().unwrap().subtitles[0].end_time,
            Timestamp::from_string("00:00:07,000").unwrap()
        );
        assert_eq!(
            result.lock().unwrap().subtitles[1].start_time,
            Timestamp::from_string("00:00:08,000").unwrap()
        );
        assert_eq!(
            result.lock().unwrap().subtitles[1].end_time,
            Timestamp::from_string("00:00:12,000").unwrap()
        );
    }
}
