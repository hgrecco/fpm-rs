use serde::{Deserialize, Serialize};

use super::FrameSchedule;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunOptions {
    pub max_iterations: usize,
    pub batch_size: usize,
    pub schedule: FrameSchedule,
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
