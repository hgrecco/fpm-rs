use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Array2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticRequest {
    Loss,
    PerFrameError,
    ObjectAmplitude,
    ObjectPhase,
    Pupil,
    ResidualImages,
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub loss: Option<f64>,
    pub per_frame_error: Option<Vec<f64>>,
    pub object_amplitude: Option<Array2<f64>>,
    pub object_phase: Option<Array2<f64>>,
    pub pupil_amplitude: Option<Array2<f64>>,
    pub pupil_phase: Option<Array2<f64>>,
    pub residual_images: Option<Vec<Array2<f64>>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepDiagnostics {
    /// Sum of per-frame losses multiplied by frame weights.
    pub loss_sum: f64,
    /// Number of frames visited, including zero-weight frames.
    pub frame_count: usize,
    pub weight_sum: f64,
    pub per_frame_loss: BTreeMap<usize, f64>,
}

impl StepDiagnostics {
    pub fn push_frame(&mut self, frame: usize, loss: f64, weight: f64) {
        self.loss_sum += weight * loss;
        self.frame_count += 1;
        self.weight_sum += weight;
        self.per_frame_loss.insert(frame, loss);
    }

    pub fn mean_loss(&self) -> Option<f64> {
        (self.weight_sum > 0.0).then(|| self.loss_sum / self.weight_sum)
    }

    pub fn merge(&mut self, other: Self) {
        self.loss_sum += other.loss_sum;
        self.frame_count += other.frame_count;
        self.weight_sum += other.weight_sum;
        self.per_frame_loss.extend(other.per_frame_loss);
    }
}
