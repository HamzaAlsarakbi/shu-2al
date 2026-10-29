//! Subtitle primitives for shu-2al: timestamps, cues, SRT/WebVTT parsing and
//! writing, Arabic text folding and hallucination filters.
//!
//! Everything here is pure: functions take `&str` and return values, so callers
//! own all file and network I/O.

mod cue;
mod error;
pub mod filter;
pub mod metrics;
pub mod normalize;
pub mod segment;
pub mod srt;
mod time;
pub mod transcript;
pub mod vtt;

pub use cue::{Cue, shift_all};
pub use error::ParseError;
pub use time::{ParseTimeError, Timestamp};
