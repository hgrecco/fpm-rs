use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::{
    Result,
    array_layout::{StandardArray2, checked_len_2d},
    array_serde::Array2Data,
    error::Error,
    experiment::Optics,
};

use super::Sampling;

/// Owned sampled complex pupil and same-shaped binary aperture support.
///
/// Arrays are shaped `(height, width)` on the low-resolution Fourier grid and stored
/// in standard row-major order. Values encode amplitude and phase transfer; support
/// entries are exactly zero or one.
#[derive(Clone, Debug, PartialEq)]
pub struct Pupil {
    pub(crate) values: StandardArray2<Complex64>,
    pub(crate) support: StandardArray2<u8>,
}

impl Pupil {
    /// Stores owned pupil arrays without copying their elements.
    ///
    /// Both inputs must have identical, non-zero shapes and C-contiguous
    /// standard row-major layout. Nonstandard inputs are rejected rather than
    /// copied. Support entries must be exactly zero or one.
    pub fn new(values: Array2<Complex64>, support: Array2<u8>) -> Result<Self> {
        let values = StandardArray2::try_from(values)?;
        let support = StandardArray2::try_from(support)?;
        if support.dim() != values.dim() {
            return Err(Error::InvalidShape(format!(
                "pupil support shape {:?} does not match value shape {:?}",
                support.dim(),
                values.dim()
            )));
        }
        if values.dim().0 == 0 || values.dim().1 == 0 {
            return Err(Error::InvalidShape(format!(
                "pupil dimensions must be non-zero, got {:?}",
                values.dim()
            )));
        }
        if values
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "pupil contains non-finite complex values".into(),
            ));
        }
        if support.as_slice().iter().any(|&value| value > 1) {
            return Err(Error::InvalidModel(
                "pupil support values must be exactly zero or one".into(),
            ));
        }
        Ok(Self { values, support })
    }

    /// Builds a sampled circular pupil from microscope optics.
    ///
    /// Samples outside `sqrt(kx^2 + ky^2) <= 2 pi NA / lambda` are set to zero.
    /// Inside the support, optional defocus contributes the paraxial phase
    /// `-z * (kx^2 + ky^2) / (2 k0)`, where `k0 = 2 pi n / lambda`.
    ///
    /// If [`crate::experiment::PupilAberration`] is present, this method adds
    /// crate-specific phase terms
    /// `astigmatism * rho^2 * cos(2 theta)`,
    /// `coma * (3 rho^3 - 2 rho) * cos(theta)`, and
    /// `spherical * (6 rho^4 - 6 rho^2 + 1)`, with amplitude
    /// `exp(-edge_apodization * rho^2)`. These coefficients are direct radian
    /// weights, not normalized Zernike coefficients.
    pub fn circular(shape: (usize, usize), sampling: &Sampling, optics: &Optics) -> Result<Self> {
        sampling.validate()?;
        optics.validate()?;
        let cutoff = std::f64::consts::TAU * optics.objective_na / optics.wavelength;
        let medium_k = optics.medium_wavenumber();
        let length = checked_len_2d(shape)?;
        if length == 0 {
            return Err(Error::InvalidShape(format!(
                "pupil dimensions must be non-zero, got {shape:?}"
            )));
        }
        let mut values = Vec::with_capacity(length);
        let mut support = Vec::with_capacity(length);
        let aberration = optics.pupil_aberration.as_ref();
        for row in 0..shape.0 {
            let ky = (row as f64 - (shape.0 / 2) as f64) * sampling.dky;
            for column in 0..shape.1 {
                let kx = (column as f64 - (shape.1 / 2) as f64) * sampling.dkx;
                let radius = kx.hypot(ky);
                let inside = radius <= cutoff;
                support.push(u8::from(inside));
                if !inside {
                    values.push(Complex64::new(0.0, 0.0));
                    continue;
                }
                let rho = if cutoff > 0.0 { radius / cutoff } else { 0.0 };
                let theta = ky.atan2(kx);
                let mut phase = 0.0;
                if let Some(defocus) = optics.defocus_distance {
                    phase -= defocus * (kx * kx + ky * ky) / (2.0 * medium_k);
                }
                let mut amplitude = 1.0;
                if let Some(aberration) = aberration {
                    phase += aberration.astigmatism * rho * rho * (2.0 * theta).cos();
                    phase += aberration.coma * (3.0 * rho.powi(3) - 2.0 * rho) * theta.cos();
                    phase += aberration.spherical * (6.0 * rho.powi(4) - 6.0 * rho * rho + 1.0);
                    amplitude = (-aberration.edge_apodization * rho * rho).exp();
                }
                values.push(Complex64::from_polar(amplitude, phase));
            }
        }
        Ok(Self {
            values: StandardArray2::from_shape_vec(shape, values)?,
            support: StandardArray2::from_shape_vec(shape, support)?,
        })
    }

    /// Returns the pupil array shape as `(height, width)`.
    pub fn shape(&self) -> (usize, usize) {
        self.values.dim()
    }

    /// Borrows the pupil values without allocating or copying.
    pub fn values(&self) -> ArrayView2<'_, Complex64> {
        self.values.ndarray_view()
    }

    /// Mutably borrows pupil elements without permitting structural mutation.
    pub fn values_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.values.ndarray_view_mut()
    }

    /// Borrows the binary pupil support without allocating or copying.
    pub fn support(&self) -> ArrayView2<'_, u8> {
        self.support.ndarray_view()
    }

    /// Replaces owned complex values without changing support.
    ///
    /// `values` must be finite, standard row-major, and have [`Self::shape`].
    pub fn replace_values(&mut self, values: Array2<Complex64>) -> Result<()> {
        let values = StandardArray2::try_from(values)?;
        if values.dim() != self.shape() {
            return Err(Error::InvalidShape(format!(
                "replacement pupil shape {:?} does not match {:?}",
                values.dim(),
                self.shape()
            )));
        }
        if values
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "pupil contains non-finite complex values".into(),
            ));
        }
        self.values = values;
        Ok(())
    }

    /// Sets every complex pupil value outside the binary aperture to zero in place.
    pub fn apply_support(&mut self) {
        for (value, &inside) in self
            .values
            .as_slice_mut()
            .iter_mut()
            .zip(self.support.as_slice())
        {
            if inside == 0 {
                *value = Complex64::new(0.0, 0.0);
            }
        }
    }
}

impl Serialize for Pupil {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Representation {
            values: Array2Data<Complex64>,
            support: Array2Data<u8>,
        }

        Representation {
            values: Array2Data::from_view(self.values()),
            support: Array2Data::from_view(self.support()),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Pupil {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            values: Array2Data<Complex64>,
            support: Array2Data<u8>,
        }

        let representation = Representation::deserialize(deserializer)?;
        let values = representation
            .values
            .into_array()
            .map_err(D::Error::custom)?;
        let support = representation
            .support
            .into_array()
            .map_err(D::Error::custom)?;
        Self::new(values, support).map_err(D::Error::custom)
    }
}
