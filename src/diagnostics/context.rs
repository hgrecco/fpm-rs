use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use ndarray::Array2;

use super::{FrameDiagnosticRecord, RawFrameStatisticsRecord};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticRequest {
    Loss,
    PerFrameError,
    RawFrameStats,
    FrameSummaries,
    ObjectAmplitude,
    ObjectPhase,
    Pupil,
    ResidualImages,
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub loss: Option<f64>,
    pub per_frame_error: Option<Vec<f64>>,
    pub raw_frame_stats: Option<Vec<RawFrameStatisticsRecord>>,
    pub frame_diagnostics: Option<Vec<FrameDiagnosticRecord>>,
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
    pub per_frame_loss: std::collections::BTreeMap<usize, f64>,
    #[serde(default)]
    admm_primal_residual_sum_squares: f64,
    #[serde(default)]
    admm_dual_residual_sum_squares: f64,
    #[serde(default)]
    admm_residual_count: usize,
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

    /// Root-mean-square ADMM consensus residual, `A x - z`, over active modes.
    pub fn admm_primal_residual_rms(&self) -> Option<f64> {
        (self.admm_residual_count > 0).then(|| {
            (self.admm_primal_residual_sum_squares / self.admm_residual_count as f64).sqrt()
        })
    }

    /// Root-mean-square ADMM dual residual, `rho (z_k - z_{k-1})`, over active modes.
    pub fn admm_dual_residual_rms(&self) -> Option<f64> {
        (self.admm_residual_count > 0)
            .then(|| (self.admm_dual_residual_sum_squares / self.admm_residual_count as f64).sqrt())
    }

    pub(crate) fn push_admm_dual_change(&mut self, change: Complex64, penalty: f64) {
        self.admm_dual_residual_sum_squares += penalty * penalty * change.norm_sqr();
    }

    pub(crate) fn push_admm_primal_residual(&mut self, residual: Complex64) {
        self.admm_primal_residual_sum_squares += residual.norm_sqr();
        self.admm_residual_count += 1;
    }

    pub fn merge(&mut self, other: Self) {
        self.loss_sum += other.loss_sum;
        self.frame_count += other.frame_count;
        self.weight_sum += other.weight_sum;
        self.per_frame_loss.extend(other.per_frame_loss);
        self.admm_primal_residual_sum_squares += other.admm_primal_residual_sum_squares;
        self.admm_dual_residual_sum_squares += other.admm_dual_residual_sum_squares;
        self.admm_residual_count += other.admm_residual_count;
    }
}
