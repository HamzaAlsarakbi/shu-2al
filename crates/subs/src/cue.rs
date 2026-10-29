use crate::Timestamp;

/// One subtitle event. `text` may hold several lines separated by `\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub start: Timestamp,
    pub end: Timestamp,
    pub text: String,
}

impl Cue {
    pub fn new(start: Timestamp, end: Timestamp, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
        }
    }

    pub fn duration_ms(&self) -> u64 {
        self.end.0.saturating_sub(self.start.0)
    }

    /// Moves the cue by `delta_ms` (negative = earlier), clamping at zero.
    pub fn shift(&mut self, delta_ms: i64) {
        self.start = self.start.shifted(delta_ms);
        self.end = self.end.shifted(delta_ms);
    }
}

/// Moves every cue by `delta_ms`, clamping at zero.
pub fn shift_all(cues: &mut [Cue], delta_ms: i64) {
    for cue in cues {
        cue.shift(delta_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: u64, end: u64) -> Cue {
        Cue::new(Timestamp(start), Timestamp(end), "x")
    }

    #[test]
    fn duration() {
        assert_eq!(cue(1000, 5000).duration_ms(), 4000);
    }

    #[test]
    fn shift_all_moves_both_ends() {
        let mut cues = vec![cue(1000, 5000), cue(6000, 10_000)];
        shift_all(&mut cues, 2000);
        assert_eq!(cues, vec![cue(3000, 7000), cue(8000, 12_000)]);
        shift_all(&mut cues, -4000);
        assert_eq!(cues, vec![cue(0, 3000), cue(4000, 8000)]);
    }
}
