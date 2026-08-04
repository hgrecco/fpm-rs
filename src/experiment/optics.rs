use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Physical microscope parameters in SI units.
///
/// # Example
///
/// ```
/// use fpm_rs::experiment::Optics;
///
/// # fn main() -> fpm_rs::Result<()> {
/// let optics = Optics {
///     wavelength: 532e-9,
///     objective_na: 0.1,
///     magnification: 4.0,
///     camera_pixel_size: 6.5e-6,
///     medium_index: 1.0,
///     defocus_distance: None,
///     pupil_aberration: None,
/// };
/// optics.validate()?;
/// assert_eq!(optics.object_pixel_size(), 6.5e-6 / 4.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Optics {
    /// Illumination wavelength in vacuum, in metres.
    pub wavelength: f64,
    /// Objective numerical aperture in the sample medium; must not exceed [`Self::medium_index`].
    pub objective_na: f64,
    /// Lateral image magnification, as a positive dimensionless ratio.
    pub magnification: f64,
    /// Physical detector-pixel pitch in metres.
    pub camera_pixel_size: f64,
    /// Refractive index between the source, sample, and objective.
    pub medium_index: f64,
    /// Axial sample displacement from the focal plane in metres.
    pub defocus_distance: Option<f64>,
    /// Optional sampled-pupil aberration model.
    ///
    /// Coefficients are interpreted by [`crate::model::Pupil::circular`] as
    /// crate-specific radial-polynomial weights in radians, not as normalized
    /// Zernike coefficients.
    pub pupil_aberration: Option<PupilAberration>,
}

/// Crate-specific pupil phase and amplitude perturbation.
///
/// For normalized pupil radius `rho` and polar angle `theta`, the sampled phase
/// contribution is
///
/// `astigmatism * rho^2 * cos(2 theta)
///  + coma * (3 rho^3 - 2 rho) * cos(theta)
///  + spherical * (6 rho^4 - 6 rho^2 + 1)`.
///
/// These are direct radian coefficients used by [`crate::model::Pupil::circular`],
/// not normalized Zernike coefficients. Defocus is stored separately on
/// [`Optics`] as [`Optics::defocus_distance`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PupilAberration {
    /// Radian coefficient multiplying `rho^2 * cos(2 theta)`.
    pub astigmatism: f64,
    /// Radian coefficient multiplying `(3 rho^3 - 2 rho) * cos(theta)`.
    pub coma: f64,
    /// Radian coefficient multiplying `6 rho^4 - 6 rho^2 + 1`.
    pub spherical: f64,
    /// Radial pupil-amplitude decay strength. The amplitude at the pupil edge
    /// is `exp(-edge_apodization)`.
    pub edge_apodization: f64,
}

impl PupilAberration {
    /// Checks finite phase coefficients and a finite, non-negative edge-apodization strength.
    pub fn validate(&self) -> Result<()> {
        if [
            self.astigmatism,
            self.coma,
            self.spherical,
            self.edge_apodization,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || self.edge_apodization < 0.0
        {
            return Err(Error::InvalidParameter {
                name: "pupil_aberration",
                reason: "phase coefficients must be finite and edge apodization must be finite and non-negative"
                    .into(),
            });
        }
        Ok(())
    }
}

impl Optics {
    /// Validates positive finite physical parameters, `objective_na <= medium_index`,
    /// finite optional defocus, and the optional [`PupilAberration`].
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("wavelength", self.wavelength),
            ("objective_na", self.objective_na),
            ("magnification", self.magnification),
            ("camera_pixel_size", self.camera_pixel_size),
            ("medium_index", self.medium_index),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: format!("must be finite and positive, got {value}"),
                });
            }
        }
        if self.objective_na > self.medium_index {
            return Err(Error::InvalidParameter {
                name: "objective_na",
                reason: "cannot exceed the medium refractive index".into(),
            });
        }
        if self
            .defocus_distance
            .is_some_and(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "defocus_distance",
                reason: "must be finite".into(),
            });
        }
        if let Some(aberration) = &self.pupil_aberration {
            aberration.validate()?;
        }
        Ok(())
    }

    /// Returns the sample-plane pixel pitch `camera_pixel_size / magnification`, in metres.
    pub fn object_pixel_size(&self) -> f64 {
        self.camera_pixel_size / self.magnification
    }

    /// Returns the medium angular wavenumber `2π n / wavelength`, in radians per metre.
    pub fn medium_wavenumber(&self) -> f64 {
        std::f64::consts::TAU * self.medium_index / self.wavelength
    }
}
