use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IlluminationErrorModel {
    /// Global source displacement in Fourier-grid pixels.
    pub global_shift: (f64, f64),
    pub rotation_degrees: f64,
    pub scale_error: f64,
    /// Per-source Fourier-grid jitter standard deviation in pixels.
    pub per_led_jitter_std: f64,
    /// Relative one-sigma frame-gain variation.
    pub intensity_variation: f64,
    /// Acquisition-frame indices whose sources fail to illuminate the sample.
    pub missing_sources: Vec<usize>,
    /// Acquisition order as a permutation of source indices.
    pub source_order: Option<Vec<usize>>,
}

impl IlluminationErrorModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn global_shift(mut self, shift: (f64, f64)) -> Self {
        self.global_shift = shift;
        self
    }

    pub fn rotation_deg(mut self, degrees: f64) -> Self {
        self.rotation_degrees = degrees;
        self
    }

    pub fn scale_error(mut self, relative_error: f64) -> Self {
        self.scale_error = relative_error;
        self
    }

    pub fn per_led_jitter_std(mut self, standard_deviation: f64) -> Self {
        self.per_led_jitter_std = standard_deviation;
        self
    }

    pub fn intensity_variation(mut self, relative_standard_deviation: f64) -> Self {
        self.intensity_variation = relative_standard_deviation;
        self
    }

    pub fn missing_sources(mut self, indices: Vec<usize>) -> Self {
        self.missing_sources = indices;
        self
    }

    pub fn source_order(mut self, order: Vec<usize>) -> Self {
        self.source_order = Some(order);
        self
    }
}
