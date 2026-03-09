use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSettings {
    pub enabled: bool,
    pub output_path: String,
}

impl Default for TargetSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            output_path: String::from("/tmp/output.srt"),
        }
    }
}
