use crate::modules::module::Module;

/// Offset module settings
pub struct OffsetModule {
    enabled: bool,
    /// The offset in milliseconds to be applied to the subtitle timings.
    offset_ms: i64,
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
    fn process(input: &crate::core::srt::SRT) -> Result<&crate::core::srt::SRT, crate::core::error::SRTError> {
        if !self.enabled {
            return Ok(input);
        }

        let mut output = input.clone();
        for subtitle in &mut output.subtitles {
            subtitle.start_time = (subtitle.start_time as i64 + self.offset_ms) as u64;
            subtitle.end_time = (subtitle.end_time as i64 + self.offset_ms) as u64;
        }
        Ok(&output)
    }

}