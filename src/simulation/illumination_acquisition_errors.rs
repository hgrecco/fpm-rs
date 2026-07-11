use serde::{Deserialize, Serialize};

/// Non-geometric illumination errors introduced during acquisition.
///
/// Physical source geometry belongs on the concrete illumination source used
/// to compile the true and reconstruction models.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IlluminationAcquisitionErrors {
    /// Relative one-sigma frame-gain variation.
    pub frame_gain_relative_std: f64,
    /// Acquisition-frame indices for which illumination fails.
    pub missing_frames: Vec<usize>,
    /// Source index assigned to each compiled source slot.
    pub source_permutation: Option<Vec<usize>>,
}

impl IlluminationAcquisitionErrors {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frame_gain_relative_std(mut self, relative_standard_deviation: f64) -> Self {
        self.frame_gain_relative_std = relative_standard_deviation;
        self
    }

    pub fn missing_frames(mut self, indices: Vec<usize>) -> Self {
        self.missing_frames = indices;
        self
    }

    pub fn source_permutation(mut self, permutation: Vec<usize>) -> Self {
        self.source_permutation = Some(permutation);
        self
    }
}
