use serde::{Deserialize, Serialize};

use super::FrameSchedule;

/// Algorithm-independent execution options consumed by [`super::Runner`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunOptions {
    /// Maximum number of complete passes through the frame schedule.
    pub max_iterations: usize,
    /// Number of acquisition frames supplied to each algorithm step; must be positive.
    pub batch_size: usize,
    /// Rule used to order acquisition frames independently on each iteration.
    pub schedule: FrameSchedule,
    /// Whether callbacks requesting frame hooks are invoked after each processed frame.
    pub enable_frame_callbacks: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            batch_size: 1,
            schedule: FrameSchedule::Sequential,
            enable_frame_callbacks: false,
        }
    }
}
