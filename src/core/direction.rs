/// This module defines the `Direction` enum, which represents the direction
/// in which we move timestamps in a subtitle file.
#[derive(Debug, Clone, Default)]
pub enum Direction {
    #[default]
    Forward,
    Backward,
}
