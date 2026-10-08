//! Optimization objectives used by reconstruction algorithms.

use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

/// Scalar data-fidelity objective between predicted and measured intensity frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    /// Mean squared error between predicted and measured amplitudes.
    #[default]
    AmplitudeMse,
    /// Mean squared error between predicted and measured intensities.
    IntensityMse,
    /// Mean Poisson negative log likelihood, omitting measurement-only constants.
    PoissonNegativeLogLikelihood,
    /// Huber loss on amplitude residuals with the implementation's fixed unit transition.
    HuberAmplitude,
}

/// Computes the selected mean frame loss from equal non-empty intensity slices.
///
/// Lengths must match and be non-zero, and values must be finite. Amplitude,
/// Poisson, and Huber-amplitude losses clamp negative intensities to zero;
/// intensity MSE uses the supplied values. Invalid lengths or non-finite
/// values return an error.
pub fn loss(predicted: &[f64], measured: &[f64], loss_type: LossType) -> Result<f64> {
    if predicted.len() != measured.len() || predicted.is_empty() {
        return Err(Error::InvalidShape(format!(
            "loss inputs have lengths {} and {}",
            predicted.len(),
            measured.len()
        )));
    }
    let mut sum = 0.0;
    for (&predicted, &measured) in predicted.iter().zip(measured) {
        if !predicted.is_finite() || !measured.is_finite() {
            return Err(Error::Numerical(
                "loss input contains a non-finite value".into(),
            ));
        }
        sum += point_loss(predicted, measured, loss_type);
    }
    Ok(sum / predicted.len() as f64)
}

pub(crate) fn point_loss(predicted: f64, measured: f64, loss_type: LossType) -> f64 {
    match loss_type {
        LossType::AmplitudeMse => {
            let residual = predicted.max(0.0).sqrt() - measured.max(0.0).sqrt();
            residual * residual
        }
        LossType::IntensityMse => {
            let residual = predicted - measured;
            residual * residual
        }
        LossType::PoissonNegativeLogLikelihood => {
            let mean = predicted.max(1e-12);
            mean - measured.max(0.0) * mean.ln()
        }
        LossType::HuberAmplitude => {
            let residual = predicted.max(0.0).sqrt() - measured.max(0.0).sqrt();
            let absolute = residual.abs();
            if absolute <= 1.0 {
                0.5 * residual * residual
            } else {
                absolute - 0.5
            }
        }
    }
}
