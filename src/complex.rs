use num_complex::Complex64;

use crate::{Array2, Result};

pub type Complex = Complex64;
pub type ComplexArray = Array2<Complex64>;

pub fn amplitude(field: &ComplexArray) -> Array2<f64> {
    field.map(|value| value.norm())
}

pub fn phase(field: &ComplexArray) -> Array2<f64> {
    field.map(|value| value.arg())
}

pub fn from_amplitude_phase(amplitude: &Array2<f64>, phase: &Array2<f64>) -> Result<ComplexArray> {
    if amplitude.shape() != phase.shape() {
        return Err(crate::Error::InvalidShape(format!(
            "amplitude {:?} and phase {:?} differ",
            amplitude.shape(),
            phase.shape()
        )));
    }
    ComplexArray::from_vec(
        amplitude.shape(),
        amplitude
            .as_slice()
            .iter()
            .zip(phase.as_slice())
            .map(|(&amplitude, &phase)| Complex64::from_polar(amplitude, phase))
            .collect(),
    )
}
