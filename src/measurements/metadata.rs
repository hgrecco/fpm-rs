use serde::{Deserialize, Serialize};

/// Acquisition metadata associated with one measurement frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Zero-based index in the current measurement stack.
    pub frame_index: usize,
    /// Optional individual illumination-source index; `None` can denote coded illumination.
    pub illumination_index: Option<usize>,
    /// Frame index in the original acquisition before subsetting or reordering.
    #[serde(default)]
    pub original_frame_index: Option<usize>,
    /// Illumination index in the original acquisition before subsetting.
    #[serde(default)]
    pub original_illumination_index: Option<usize>,
    /// Positive exposure time in caller-defined units; ratios are used for normalization.
    pub exposure_time: f64,
    /// Non-negative reconstruction weight; zero excludes the frame from the objective.
    pub weight: f64,
    /// Optional human-readable acquisition label.
    pub label: Option<String>,
}

impl FrameMetadata {
    /// Creates unit-exposure, unit-weight metadata whose current and original frame and
    /// illumination indices all equal `frame_index`.
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
