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
    enabled: bool,
    /// The offset in milliseconds to be applied to the subtitle timings.
    offset: Duration,
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
