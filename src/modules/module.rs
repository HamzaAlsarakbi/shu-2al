use crate::core::{error::SRTError, srt::SRT};

pub trait Module {
    /// Processes a subtitle file and returns its content
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to an SRT object.
    ///
    /// # Returns
    ///
    /// * `Result<&SRT, SRTError>` - Returns a reference to the processed SRT object if successful, or an error message if it fails.
    fn process(input: &SRT) -> Result<&SRT, SRTError>;
}
