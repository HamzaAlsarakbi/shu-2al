use std::time::Duration;

use super::direction::Direction;

#[derive(Debug, Clone)]
pub struct Timestamp {
    hours: u32,
    minutes: u32,
    seconds: u32,
    milliseconds: u32,
}

impl Timestamp {
    /// Creates a new `Timestamp` instance from the given hours, minutes, seconds, and milliseconds.
    ///
    /// # Arguments
    ///
    /// * `hours` - The number of hours.
    /// * `minutes` - The number of minutes.
    /// * `seconds` - The number of seconds.
    /// * `milliseconds` - The number of milliseconds.
    ///
    /// # Returns
    ///
    /// * `Timestamp` - Returns a new `Timestamp` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use srt::timestamp::Timestamp;
    ///
    /// let timestamp = Timestamp::new("00:00:01,000");
    ///
    /// assert_eq!(timestamp.hours, 0);
    /// assert_eq!(timestamp.minutes, 0);
    /// assert_eq!(timestamp.seconds, 1);
    /// assert_eq!(timestamp.milliseconds, 0);
    /// ```
    pub fn from_string(timestamp_str: &str) -> Result<Self, String> {
        let parts: Vec<&str> = timestamp_str.split(':').collect();
        if parts.len() != 3 {
            return Err("Invalid timestamp format".to_string());
        }

        let hours: u32 = parts[0].parse().map_err(|_| "Invalid hours")?;
        let minutes: u32 = parts[1].parse().map_err(|_| "Invalid minutes")?;
        let seconds_parts: Vec<&str> = parts[2].split(',').collect();
        if seconds_parts.len() != 2 {
            return Err("Invalid seconds format".to_string());
        }
        let seconds: u32 = seconds_parts[0].parse().map_err(|_| "Invalid seconds")?;
        let milliseconds: u32 = seconds_parts[1]
            .parse()
            .map_err(|_| "Invalid milliseconds")?;

        Ok(Timestamp {
            hours,
            minutes,
            seconds,
            milliseconds,
        })
    }

    /// Converts the `Timestamp` instance to  milliseconds.
    ///
    /// # Returns
    ///
    /// * `u64` - The timestamp in milliseconds.
    pub fn to_millis(&self) -> u64 {
        let total_seconds = self.hours * 3600 + self.minutes * 60 + self.seconds;
        let total_milliseconds = total_seconds * 1000 + self.milliseconds;
        total_milliseconds as u64
    }

    /// Creates a new `Timestamp` instance from the given millis value.
    ///
    /// # Arguments
    ///
    /// * `millis` - The timestamp in milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        let total_seconds = millis / 1000;
        let milliseconds = (millis % 1000) as u32;
        let seconds = (total_seconds % 60) as u32;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u32;
        let hours = (total_minutes / 60) as u32;

        Timestamp {
            hours,
            minutes,
            seconds,
            milliseconds,
        }
    }

    /// Moves the timestamp by the given duration in the specified direction.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to move the timestamp by.
    /// * `direction` - The direction to move the timestamp in (forward or backward).
    ///
    /// # Returns
    ///
    /// * `Result<(), String>` - Returns `Ok(())` if successful, or an error message if it fails.
    pub fn move_ts(&mut self, delta: Duration, direction: Direction) -> Result<(), String> {
        let total_milliseconds = self.to_millis() as i64;

        if total_milliseconds < 0 {
            return Err("Timestamp cannot be negative".to_string());
        }

        if delta.as_millis() > i64::MAX as u128 {
            return Err("Duration is too large".to_string());
        }

        let delta = match direction {
            Direction::Forward => delta.as_millis() as i64,
            Direction::Backward => -(delta.as_millis() as i64),
        };

        let new_timestamp = std::cmp::max(total_milliseconds + delta, 0);
        *self = Timestamp::from_millis(new_timestamp as u64);

        Ok(())
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        self.hours == other.hours
            && self.minutes == other.minutes
            && self.seconds == other.seconds
            && self.milliseconds == other.milliseconds
    }
}
impl Eq for Timestamp {}
impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:02}:{:02}:{:02},{:03}",
            self.hours, self.minutes, self.seconds, self.milliseconds
        )
    }
}
impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.hours != other.hours {
            return self.hours.cmp(&other.hours);
        }
        if self.minutes != other.minutes {
            return self.minutes.cmp(&other.minutes);
        }
        if self.seconds != other.seconds {
            return self.seconds.cmp(&other.seconds);
        }
        self.milliseconds.cmp(&other.milliseconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_from_string() {
        let timestamp = Timestamp::from_string("00:00:01,000").unwrap();
        assert_eq!(timestamp.hours, 0);
        assert_eq!(timestamp.minutes, 0);
        assert_eq!(timestamp.seconds, 1);
        assert_eq!(timestamp.milliseconds, 0);
    }

    #[test]
    fn test_timestamp_from_string_invalid() {
        assert!(Timestamp::from_string("00:00:01").is_err());
        assert!(Timestamp::from_string("00:00:01,000,000").is_err());
        assert!(Timestamp::from_string("00:00:01,abc").is_err());
    }
    #[test]
    fn test_timestamp_display() {
        let timestamp = Timestamp::from_string("00:00:01,000").unwrap();
        assert_eq!(timestamp.to_string(), "00:00:01,000");
    }

    #[test]
    fn test_timestamp_partial_eq() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:01,000").unwrap();
        assert_eq!(timestamp1, timestamp2);
    }

    #[test]
    fn test_timestamp_partial_ord() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert!(timestamp1 < timestamp2);
    }
    #[test]
    fn test_timestamp_ord() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert!(timestamp1 < timestamp2);
    }
    #[test]
    fn test_timestamp_eq() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:01,000").unwrap();
        assert_eq!(timestamp1, timestamp2);
    }
    #[test]
    fn test_timestamp_ne() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert_ne!(timestamp1, timestamp2);
    }
    #[test]
    fn test_timestamp_lt() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert!(timestamp1 < timestamp2);
    }
    #[test]
    fn test_timestamp_le() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert!(timestamp1 <= timestamp2);
    }
    #[test]
    fn test_timestamp_gt() {
        let timestamp1 = Timestamp::from_string("00:00:02,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:01,000").unwrap();
        assert!(timestamp1 > timestamp2);
    }
    #[test]
    fn test_timestamp_ge() {
        let timestamp1 = Timestamp::from_string("00:00:02,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:01,000").unwrap();
        assert!(timestamp1 >= timestamp2);
    }
    #[test]
    fn test_timestamp_cmp() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert_eq!(timestamp1.cmp(&timestamp2), std::cmp::Ordering::Less);
    }
    #[test]
    fn test_timestamp_partial_cmp() {
        let timestamp1 = Timestamp::from_string("00:00:01,000").unwrap();
        let timestamp2 = Timestamp::from_string("00:00:02,000").unwrap();
        assert_eq!(
            timestamp1.partial_cmp(&timestamp2),
            Some(std::cmp::Ordering::Less)
        );
    }
    #[test]
    fn test_timestamp_to_millis() {
        let timestamp = Timestamp::from_string("00:00:01,000").unwrap();
        assert_eq!(timestamp.to_millis(), 1000);
    }
    #[test]
    fn test_timestamp_from_millis() {
        let timestamp = Timestamp::from_millis(1000);
        assert_eq!(timestamp.hours, 0);
        assert_eq!(timestamp.minutes, 0);
        assert_eq!(timestamp.seconds, 1);
        assert_eq!(timestamp.milliseconds, 0);
    }
    #[test]
    fn test_timestamp_move_ts_forward() {
        let mut timestamp = Timestamp::from_string("00:00:01,000").unwrap();
        timestamp
            .move_ts(Duration::new(2, 0), Direction::Forward)
            .unwrap();
        assert_eq!(timestamp.to_string(), "00:00:03,000");
    }
    #[test]
    fn test_timestamp_move_ts_backward() {
        let mut timestamp = Timestamp::from_string("00:00:02,000").unwrap();
        timestamp
            .move_ts(Duration::new(2, 0), Direction::Backward)
            .unwrap();
        assert_eq!(timestamp.to_string(), "00:00:00,000");
    }

    #[test]
    fn test_timestamp_move_ts_backward_clamped() {
        let mut timestamp = Timestamp::from_string("00:00:01,000").unwrap();
        timestamp
            .move_ts(Duration::new(2, 0), Direction::Backward)
            .unwrap();
        assert_eq!(timestamp.to_string(), "00:00:00,000");
    }
}
