//! Allocation-aware helpers for two-dimensional complex sample fields.
//!
//! Use [`crate::complex::amplitude`] and [`crate::complex::phase`] to derive real arrays,
//! or [`crate::complex::from_amplitude_phase`] to construct a complex field. Inputs are borrowed
//! [`ndarray`] views and results are newly allocated in standard row-major order.

use ndarray::{Array2, ArrayView2, Zip};
use num_complex::Complex64;

use crate::{Error, Result};

/// Double-precision complex scalar used for fields, spectra, and pupils.
pub type Complex = Complex64;

/// Computes amplitude from any logical two-dimensional layout.
///
/// The returned array is newly allocated in standard row-major order.
pub fn amplitude(field: ArrayView2<'_, Complex64>) -> Array2<f64> {
    field.mapv(|value| value.norm())
}

/// Computes phase from any logical two-dimensional layout.
///
/// The returned array is newly allocated in standard row-major order.
pub fn phase(field: ArrayView2<'_, Complex64>) -> Array2<f64> {
    field.mapv(|value| value.arg())
}

/// Combines amplitude and phase from arbitrary, matching ndarray layouts.
///
/// The inputs are borrowed without copying. The output allocation is standard
/// row-major and follows their logical traversal order.
pub fn from_amplitude_phase(
    amplitude: ArrayView2<'_, f64>,
    phase: ArrayView2<'_, f64>,
) -> Result<Array2<Complex64>> {
    if amplitude.dim() != phase.dim() {
        return Err(Error::InvalidShape(format!(
            "amplitude {:?} and phase {:?} differ",
            amplitude.dim(),
            phase.dim()
        )));
    }
    let mut output = Array2::from_elem(amplitude.dim(), Complex64::default());
    Zip::from(&mut output)
        .and(amplitude)
        .and(phase)
        .for_each(|destination, &amplitude, &phase| {
            *destination = Complex64::from_polar(amplitude, phase);
        });
    Ok(output)
}
