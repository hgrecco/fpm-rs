//! Metrics comparing a reference and candidate complex field.

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
};

/// Aggregate errors between reference and globally phase-aligned complex fields.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComplexFieldComparisonMetrics {
    /// Root-mean-square amplitude error.
    pub amplitude_rmse: f64,
    /// Amplitude L2 error normalized by reference-field L2 norm.
    pub amplitude_nrmse: f64,
    /// Root-mean-square complex residual after global-phase alignment.
    pub complex_rmse: f64,
    /// Complex residual L2 norm divided by reference-field L2 norm.
    pub complex_nrmse: f64,
    /// Root-mean-square wrapped phase error in radians over non-dark reference pixels.
    pub phase_rmse: f64,
    /// Mean absolute wrapped phase error in radians over non-dark reference pixels.
    pub phase_mae: f64,
    /// Centered Fourier-spectrum residual L2 norm divided by reference-spectrum norm.
    pub fourier_nrmse: f64,
    /// Fitted global candidate-to-reference phase offset in radians.
    pub global_phase_offset: f64,
}

/// Compares equal-shaped fields after fitting one global phase offset.
pub fn compare_complex_fields(
    reference: ArrayView2<'_, Complex64>,
    candidate: ArrayView2<'_, Complex64>,
) -> Result<ComplexFieldComparisonMetrics> {
    compare_complex_fields_masked(reference, candidate, None)
}

/// Compares equal-shaped fields over non-zero entries of an optional byte mask.
pub fn compare_complex_fields_masked(
    reference: ArrayView2<'_, Complex64>,
    candidate: ArrayView2<'_, Complex64>,
    valid_mask: Option<ArrayView2<'_, u8>>,
) -> Result<ComplexFieldComparisonMetrics> {
    if reference.dim() != candidate.dim() || reference.is_empty() {
        return Err(Error::InvalidShape(format!(
            "reference shape {:?} differs from candidate {:?}",
            reference.dim(),
            candidate.dim()
        )));
    }
    if valid_mask.is_some_and(|mask| mask.dim() != reference.dim()) {
        return Err(Error::InvalidShape(
            "complex-field mask shape differs from inputs".into(),
        ));
    }
    let mask_values: Vec<u8> = valid_mask
        .map(|mask| mask.iter().copied().collect())
        .unwrap_or_else(|| vec![1; reference.len()]);
    let count = mask_values.iter().filter(|&&value| value != 0).count();
    if count == 0 {
        return Err(Error::InvalidParameter {
            name: "valid_mask",
            reason: "must select at least one field sample".into(),
        });
    }
    let cross: Complex64 = reference
        .iter()
        .zip(candidate.iter())
        .zip(&mask_values)
        .filter(|&(_, &valid)| valid != 0)
        .map(|((&reference, &candidate), _)| candidate * reference.conj())
        .sum();
    let global_phase_offset = cross.arg();
    let correction = Complex64::from_polar(1.0, -global_phase_offset);
    let mut amplitude_squared = 0.0;
    let mut complex_squared = 0.0;
    let mut phase_squared = 0.0;
    let mut phase_absolute = 0.0;
    let mut reference_amplitude_squared = 0.0;
    let mut reference_complex_squared = 0.0;
    for ((&reference, &candidate), &valid) in
        reference.iter().zip(candidate.iter()).zip(&mask_values)
    {
        if valid == 0 {
            continue;
        }
        let aligned = candidate * correction;
        amplitude_squared += (aligned.norm() - reference.norm()).powi(2);
        complex_squared += (aligned - reference).norm_sqr();
        reference_amplitude_squared += reference.norm_sqr();
        reference_complex_squared += reference.norm_sqr();
        let phase_error = wrap_phase(aligned.arg() - reference.arg());
        phase_squared += phase_error * phase_error;
        phase_absolute += phase_error.abs();
    }
    let reference_spectrum = spectrum(masked(reference, valid_mask)?.view())?;
    let candidate_spectrum = spectrum(masked(candidate, valid_mask)?.view())?;
    let fourier_squared: f64 = reference_spectrum
        .iter()
        .zip(candidate_spectrum.iter())
        .map(|(&reference, &candidate)| (candidate * correction - reference).norm_sqr())
        .sum();
    let fourier_reference_squared: f64 = reference_spectrum
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let count = count as f64;
    let amplitude_rmse = (amplitude_squared / count).sqrt();
    let complex_rmse = (complex_squared / count).sqrt();
    Ok(ComplexFieldComparisonMetrics {
        amplitude_rmse,
        amplitude_nrmse: amplitude_rmse
            / (reference_amplitude_squared / count)
                .sqrt()
                .max(f64::EPSILON),
        complex_rmse,
        complex_nrmse: complex_rmse / (reference_complex_squared / count).sqrt().max(f64::EPSILON),
        phase_rmse: (phase_squared / count).sqrt(),
        phase_mae: phase_absolute / count,
        fourier_nrmse: (fourier_squared / fourier_reference_squared.max(f64::EPSILON)).sqrt(),
        global_phase_offset,
    })
}

fn masked(
    values: ArrayView2<'_, Complex64>,
    mask: Option<ArrayView2<'_, u8>>,
) -> Result<Array2<Complex64>> {
    let data = match mask {
        Some(mask) => values
            .iter()
            .zip(mask.iter())
            .map(|(&value, &valid)| {
                if valid != 0 {
                    value
                } else {
                    Complex64::default()
                }
            })
            .collect(),
        None => values.iter().copied().collect(),
    };
    Ok(Array2::from_shape_vec(values.dim(), data)?)
}

fn spectrum(values: ArrayView2<'_, Complex64>) -> Result<Array2<Complex64>> {
    let shape = values.dim();
    let backend = CpuBackend::new(shape, shape)?;
    let mut spectrum: Vec<_> = values.iter().copied().collect();
    let mut column = vec![Complex64::default(); shape.0];
    backend.fft2(&mut spectrum, shape, FftDirection::Forward, &mut column)?;
    Ok(Array2::from_shape_vec(shape, spectrum)?)
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
