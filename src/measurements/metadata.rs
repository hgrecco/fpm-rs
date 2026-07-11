use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub frame_index: usize,
    pub illumination_index: Option<usize>,
    /// Frame index in the original acquisition before subsetting or reordering.
    #[serde(default)]
    pub original_frame_index: Option<usize>,
    /// Illumination index in the original acquisition before subsetting.
    #[serde(default)]
    pub original_illumination_index: Option<usize>,
    pub exposure_time: f64,
    pub weight: f64,
    pub label: Option<String>,
}

impl FrameMetadata {
    pub fn new(frame_index: usize) -> Self {
        Self {
            frame_index,
            illumination_index: Some(frame_index),
            original_frame_index: Some(frame_index),
            original_illumination_index: Some(frame_index),
            exposure_time: 1.0,
            weight: 1.0,
            label: None,
        }
    }
}
