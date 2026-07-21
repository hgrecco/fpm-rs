//! Metrics comparing a reference and candidate complex field.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Array2, Result,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComplexFieldComparisonMetrics {
    pub amplitude_rmse: f64,
    pub amplitude_nrmse: f64,
    pub complex_rmse: f64,
    pub complex_nrmse: f64,
    pub phase_rmse: f64,
    pub phase_mae: f64,
    pub fourier_nrmse: f64,
    pub global_phase_offset: f64,
}

pub fn compare_complex_fields(
    reference: &Array2<Complex64>,
    candidate: &Array2<Complex64>,
) -> Result<ComplexFieldComparisonMetrics> {
    compare_complex_fields_masked(reference, candidate, None)
}

pub fn compare_complex_fields_masked(
    reference: &Array2<Complex64>,
    candidate: &Array2<Complex64>,
    valid_mask: Option<&Array2<u8>>,
) -> Result<ComplexFieldComparisonMetrics> {
    if reference.shape() != candidate.shape() || reference.is_empty() {
        return Err(Error::InvalidShape(format!(
            "reference shape {:?} differs from candidate {:?}",
            reference.shape(),
            candidate.shape()
        )));
    }
    if valid_mask.is_some_and(|mask| mask.shape() != reference.shape()) {
        return Err(Error::InvalidShape(
            "complex-field mask shape differs from inputs".into(),
        ));
    }
    let valid = |index: usize| valid_mask.is_none_or(|mask| mask.as_slice()[index] != 0);
    let count = (0..reference.len()).filter(|&index| valid(index)).count();
    if count == 0 {
        return Err(Error::InvalidParameter {
            name: "valid_mask",
            reason: "must select at least one field sample".into(),
        });
    }
    let cross: Complex64 = reference
        .as_slice()
        .iter()
        .zip(candidate.as_slice())
        .enumerate()
        .filter(|(index, _)| valid(*index))
        .map(|(_, (&reference, &candidate))| candidate * reference.conj())
        .sum();
    let global_phase_offset = cross.arg();
    let correction = Complex64::from_polar(1.0, -global_phase_offset);
    let mut amplitude_squared = 0.0;
    let mut complex_squared = 0.0;
    let mut phase_squared = 0.0;
    let mut phase_absolute = 0.0;
    let mut reference_amplitude_squared = 0.0;
    let mut reference_complex_squared = 0.0;
    for (index, (&reference, &candidate)) in reference
        .as_slice()
        .iter()
        .zip(candidate.as_slice())
        .enumerate()
    {
        if !valid(index) {
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
    let reference_spectrum = spectrum(&masked(reference, valid_mask)?)?;
    let candidate_spectrum = spectrum(&masked(candidate, valid_mask)?)?;
    let fourier_squared: f64 = reference_spectrum
        .as_slice()
        .iter()
        .zip(candidate_spectrum.as_slice())
        .map(|(&reference, &candidate)| (candidate * correction - reference).norm_sqr())
        .sum();
    let fourier_reference_squared: f64 = reference_spectrum
        .as_slice()
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

fn masked(values: &Array2<Complex64>, mask: Option<&Array2<u8>>) -> Result<Array2<Complex64>> {
    Array2::from_vec(
        values.shape(),
        values
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if mask.is_none_or(|mask| mask.as_slice()[index] != 0) {
                    value
                } else {
                    Complex64::default()
                }
            })
            .collect(),
    )
}

fn spectrum(values: &Array2<Complex64>) -> Result<Array2<Complex64>> {
    let shape = values.shape();
    let backend = CpuBackend::new(shape, shape)?;
    let mut spectrum = values.as_slice().to_vec();
    let mut column = vec![Complex64::default(); shape.0];
    backend.fft2(&mut spectrum, shape, FftDirection::Forward, &mut column)?;
    Array2::from_vec(shape, spectrum)
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
