use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{Array2, Result, error::Error};

use super::{FrameDiagnostics, RawFrameStats};

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
    pub raw_frame_stats: Option<Vec<RawFrameStats>>,
    pub frame_diagnostics: Option<Vec<FrameDiagnostics>>,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroundTruthMetrics {
    pub amplitude_rmse: f64,
    pub amplitude_nrmse: f64,
    pub complex_rmse: f64,
    pub complex_nrmse: f64,
    pub phase_rmse: Option<f64>,
    pub phase_mae: Option<f64>,
    pub fourier_nrmse: Option<f64>,
}

pub fn compute_ground_truth_metrics(
    truth: &Array2<Complex64>,
    reconstruction: &Array2<Complex64>,
) -> Result<GroundTruthMetrics> {
    if truth.shape() != reconstruction.shape() || truth.is_empty() {
        return Err(Error::InvalidShape(format!(
            "reconstruction shape {:?} differs from ground truth {:?}",
            reconstruction.shape(),
            truth.shape()
        )));
    }
    let cross: Complex64 = truth
        .as_slice()
        .iter()
        .zip(reconstruction.as_slice())
        .map(|(&truth, &reconstruction)| truth.conj() * reconstruction)
        .sum();
    let phase_offset = cross.arg();
    let correction = Complex64::from_polar(1.0, -phase_offset);

    let mut amplitude_squared = 0.0;
    let mut complex_squared = 0.0;
    let mut phase_squared = 0.0;
    let mut phase_absolute = 0.0;
    let mut truth_amplitude_squared = 0.0;
    let mut truth_complex_squared = 0.0;
    for (&truth, &reconstruction) in truth.as_slice().iter().zip(reconstruction.as_slice()) {
        let aligned = reconstruction * correction;
        let truth_amplitude = truth.norm();
        let aligned_amplitude = aligned.norm();
        amplitude_squared += (aligned_amplitude - truth_amplitude).powi(2);
        complex_squared += (aligned - truth).norm_sqr();
        truth_amplitude_squared += truth_amplitude * truth_amplitude;
        truth_complex_squared += truth.norm_sqr();
        let phase_error = wrap_phase(aligned.arg() - truth.arg());
        phase_squared += phase_error * phase_error;
        phase_absolute += phase_error.abs();
    }
    let count = truth.len() as f64;
    let amplitude_rmse = (amplitude_squared / count).sqrt();
    let complex_rmse = (complex_squared / count).sqrt();
    let truth_amplitude_rms = (truth_amplitude_squared / count).sqrt();
    let truth_complex_rms = (truth_complex_squared / count).sqrt();
    Ok(GroundTruthMetrics {
        amplitude_rmse,
        amplitude_nrmse: amplitude_rmse / truth_amplitude_rms.max(f64::EPSILON),
        complex_rmse,
        complex_nrmse: complex_rmse / truth_complex_rms.max(f64::EPSILON),
        phase_rmse: Some((phase_squared / count).sqrt()),
        phase_mae: Some(phase_absolute / count),
        fourier_nrmse: None,
    })
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
