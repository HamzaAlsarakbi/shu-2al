use std::fmt;
use std::str::FromStr;

/// A point in media time, stored as whole milliseconds.
///
/// Displays in SRT form (`01:02:03,456`); use [`Timestamp::to_vtt`] for WebVTT.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid timestamp {input:?}: {reason}")]
pub struct ParseTimeError {
    input: String,
    reason: &'static str,
}

impl Timestamp {
    pub const ZERO: Self = Self(0);

    pub const fn from_millis(ms: u64) -> Self {
        Self(ms)
    }

    pub const fn as_millis(self) -> u64 {
        self.0
    }

    /// Moves the timestamp by `delta_ms` (negative = earlier), clamping at zero.
    pub const fn shifted(self, delta_ms: i64) -> Self {
        Self(self.0.saturating_add_signed(delta_ms))
    }

    /// WebVTT form: `01:02:03.456`.
    pub fn to_vtt(self) -> String {
        self.format('.')
    }

    fn format(self, sep: char) -> String {
        let secs = self.0 / 1000;
        format!(
            "{:02}:{:02}:{:02}{sep}{:03}",
            secs / 3600,
            secs / 60 % 60,
            secs % 60,
            self.0 % 1000
        )
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format(','))
    }
}

/// Accepts `HH:MM:SS,mmm`, `HH:MM:SS.mmm` and WebVTT's short `MM:SS.mmm`.
/// Hours may exceed two digits; 1–2 fraction digits are read as tenths/hundredths.
impl FromStr for Timestamp {
    type Err = ParseTimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let err = |reason| ParseTimeError {
            input: s.to_owned(),
            reason,
        };
        let digits = |part: &str| -> Result<u64, ParseTimeError> {
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(err("expected digits"));
            }
            part.parse().map_err(|_| err("number too large"))
        };

        let (clock, frac) = s
            .rsplit_once([',', '.'])
            .ok_or_else(|| err("missing milliseconds"))?;
        if frac.len() > 3 {
            return Err(err("milliseconds must be 1-3 digits"));
        }
        let ms = digits(frac)? * 10u64.pow(3 - frac.len() as u32);

        let (h, m, sec) = match clock.split(':').collect::<Vec<_>>()[..] {
            [h, m, sec] => (digits(h)?, digits(m)?, digits(sec)?),
            [m, sec] => (0, digits(m)?, digits(sec)?),
            _ => return Err(err("expected HH:MM:SS")),
        };
        if m >= 60 || sec >= 60 {
            return Err(err("minutes and seconds must be below 60"));
        }
        Ok(Self(((h * 60 + m) * 60 + sec) * 1000 + ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    #[test]
    fn parses_srt_and_vtt_forms() {
        assert_eq!(ts("00:00:01,000"), Timestamp(1000));
        assert_eq!(ts("01:02:03.456"), Timestamp(3_723_456));
        assert_eq!(ts("02:03.456"), Timestamp(123_456));
        assert_eq!(ts("100:00:00,000"), Timestamp(360_000_000));
        assert_eq!(ts(" 00:00:01,5 "), Timestamp(1500));
        assert_eq!(ts("00:00:01,05"), Timestamp(1050));
    }

    #[test]
    fn rejects_malformed() {
        for bad in [
            "00:00:01",
            "00:00:01,000,000",
            "00:00:01,abc",
            "00:00:01,0000",
            "00:60:00,000",
            "00:00:60,000",
            "1:2:3:4,000",
            "",
            "-1:00:00,000",
        ] {
            assert!(bad.parse::<Timestamp>().is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn displays_both_forms() {
        let t = Timestamp(3_723_456);
        assert_eq!(t.to_string(), "01:02:03,456");
        assert_eq!(t.to_vtt(), "01:02:03.456");
        assert_eq!(Timestamp::ZERO.to_string(), "00:00:00,000");
    }

    #[test]
    fn round_trips() {
        for ms in [0, 1, 999, 1000, 59_999, 3_599_999, 86_400_000] {
            assert_eq!(ts(&Timestamp(ms).to_string()), Timestamp(ms));
        }
    }

    #[test]
    fn orders_by_time() {
        assert!(ts("00:00:59,999") < ts("00:01:00,000"));
        assert!(ts("01:00:00,000") > ts("00:59:59,999"));
    }

    #[test]
    fn shifts_and_clamps_at_zero() {
        assert_eq!(ts("00:00:01,000").shifted(2000), ts("00:00:03,000"));
        assert_eq!(ts("00:00:02,000").shifted(-2000), Timestamp::ZERO);
        assert_eq!(ts("00:00:01,000").shifted(-5000), Timestamp::ZERO);
    }
}
