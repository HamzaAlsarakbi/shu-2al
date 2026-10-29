//! `transcript.json`, as written by the worker's `asr` and `align` stages.
//! Times are seconds.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub version: u32,
    pub audio_s: f64,
    pub model: String,
    /// VAD speech regions as `[start, end]`.
    #[serde(default)]
    pub speech: Vec<[f64; 2]>,
    pub segments: Vec<Segment>,
}

/// One ASR chunk. `words` is present once the `align` stage succeeded on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub w: String,
    pub start: f64,
    pub end: f64,
    /// Mean CTC log-probability from the aligner (≤ 0; closer to 0 is surer).
    #[serde(default)]
    pub score: Option<f64>,
}
