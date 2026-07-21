//! Metrics calculated from one complex-valued field.

use num_complex::Complex64;

use crate::{
    Array2, Result,
    backend::{Backend, CpuBackend, FftDirection},
};

/// Azimuthally averaged power in annular bins of a two-dimensional Fourier transform.
///
/// [`radial_fourier_spectrum`] uses integer-radius bins centered on the DC
/// component. `radius_px[b]` is the radius, in Fourier-grid pixels, of bin
/// `b`; `power[b]` is the mean normalized power in that bin; and
/// `sample_count[b]` is the number of Fourier samples contributing to it.
///
/// Each underlying Fourier-pixel power is normalized so the sum before radial
/// averaging equals the input field's squared norm (Parseval's identity). No
/// physical-frequency calibration is applied: supplying that requires the
/// image sampling pitch.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialFourierSpectrum {
    pub radius_px: Vec<f64>,
    pub power: Vec<f64>,
    pub sample_count: Vec<usize>,
}

/// Calculate the radial Fourier power spectrum of a two-dimensional complex field.
///
/// This applies a forward FFT, treats DC as the centre of the Fourier grid, and
/// averages normalized power over annular bins with integer Fourier-pixel radius.
pub fn radial_fourier_spectrum(field: &Array2<Complex64>) -> Result<RadialFourierSpectrum> {
    let shape = field.shape();
    let mut spectrum = field.as_slice().to_vec();
    let backend = CpuBackend::new(shape, shape)?;
    let mut column_scratch = vec![Complex64::default(); shape.0];
    backend.fft2(
        &mut spectrum,
        shape,
        FftDirection::Forward,
        &mut column_scratch,
    )?;

    let center_row = shape.0 / 2;
    let center_column = shape.1 / 2;
    let max_radius = ((center_row as f64).hypot(center_column as f64)).floor() as usize;
    let mut power = vec![0.0; max_radius + 1];
    let mut sample_count = vec![0_usize; max_radius + 1];
    let normalization = field.len() as f64;

    for row in 0..shape.0 {
        let frequency_row = if row <= shape.0 / 2 {
            row as isize
        } else {
            row as isize - shape.0 as isize
        };
        for column in 0..shape.1 {
            let frequency_column = if column <= shape.1 / 2 {
                column as isize
            } else {
                column as isize - shape.1 as isize
            };
            let radius = ((frequency_row * frequency_row + frequency_column * frequency_column)
                as f64)
                .sqrt()
                .floor() as usize;
            power[radius] += spectrum[row * shape.1 + column].norm_sqr() * normalization;
            sample_count[radius] += 1;
        }
    }

    for (value, count) in power.iter_mut().zip(&sample_count) {
        *value /= *count as f64;
    }

    Ok(RadialFourierSpectrum {
        radius_px: (0..power.len()).map(|bin| bin as f64).collect(),
        power,
        sample_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_field_has_power_only_at_dc() {
        let field = Array2::from_vec((4, 4), vec![Complex64::new(1.0, 0.0); 16]).unwrap();

        let spectrum = radial_fourier_spectrum(&field).unwrap();

        assert_eq!(spectrum.radius_px, vec![0.0, 1.0, 2.0]);
        assert_eq!(spectrum.sample_count, vec![1, 8, 7]);
        assert!((spectrum.power[0] - 16.0).abs() < 1e-12);
        assert!(spectrum.power[1..].iter().all(|value| value.abs() < 1e-12));
    }
}
