/// This module defines the `Direction` enum, which represents the direction of a subtitle.
#[derive(Debug, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Forward
    }
}
