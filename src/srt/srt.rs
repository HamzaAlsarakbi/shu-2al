use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use super::subtitle::Subtitle;

pub struct SRT {
    file_path: String,
    /// The list of subtitles in the SRT file.
    subtitles: Vec<Subtitle>,
}

impl SRT {
    /// Creates a new `SRT` instance with the given file path.
    ///
    /// # Arguments
    ///
    /// * `file_path` - A string representing the path to the SRT file.
    ///
    /// # Returns
    ///
    /// * `SRT` - Returns a new `SRT` instance.
    pub fn new(file_path: &str) -> Self {
        SRT {
            file_path: file_path.to_string(),
            subtitles: Vec::new(),
        }
    }

    /// Reads the SRT file and populates the `subtitles` vector.
    ///
    /// # Returns
    ///
    /// * `Result<(), String>` - Returns `Ok(())` if successful, or an error message if it fails.
    pub fn read_file(&mut self) -> Result<(), String> {
        let file = File::open(&self.file_path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut lines: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim().to_string();
            if line.is_empty() {
                lines.clear();
                continue;
            }

            lines.push(line);

            if lines.len() > 1 {
                if let Ok(subtitle) = Subtitle::new(&lines.iter().map(|e| e.as_str()).collect()) {
                    self.subtitles.push(subtitle);
                    lines.clear();
                }
            }
        }

        Ok(())
    }

    /// Writes the subtitles to the SRT file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - A string representing the path to the SRT file.
    ///
    /// # Returns
    ///
    /// * `Result<(), String>` - Returns `Ok(())` if successful, or an error message if it fails.
    pub fn write_file(&self, file_path: &str) -> Result<(), String> {
        let file = File::create(file_path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        for (i, subtitle) in self.subtitles.iter().enumerate() {
            writeln!(writer, "{}", i + 1).map_err(|e| e.to_string())?;
            writeln!(writer, "{}", subtitle.to_string()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_read_file() {
        let test_file_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/test_1/input.srt");
        let mut srt = SRT::new(test_file_path);
        assert!(srt.read_file().is_ok());
        assert!(!srt.subtitles.is_empty());
    }

    // #[test]
    // fn test_srt_write_file() {
    //     let srt = SRT::new("test.srt");
    //     assert!(srt.write_file("output.srt").is_ok());
    // }
}
