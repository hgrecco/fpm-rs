use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Physical microscope parameters in SI units.
///
/// The sample-plane detector pitch must satisfy the coherent-field sampling
/// invariant
///
/// `camera_pixel_size / magnification < wavelength_vacuum_m / (2 * objective_na)`.
///
/// Equality is rejected because the circular pupil cutoff would lie on the
/// one-sided discrete Nyquist boundary.
///
/// # References
///
/// - [G. Zheng, R. Horstmeyer, and C. Yang, “Wide-field, high-resolution
///   Fourier ptychographic microscopy,” *Nature Photonics* **7**, 739–745
///   (2013).](https://doi.org/10.1038/nphoton.2013.187)
///
/// # Example
///
/// ```
/// use fpm_rs::experiment::Optics;
///
/// # fn main() -> fpm_rs::Result<()> {
/// let optics = Optics {
///     wavelength_vacuum_m: 532e-9,
///     objective_na: 0.1,
///     magnification: 4.0,
///     camera_pixel_size: 6.5e-6,
///     illumination_refractive_index: 1.0,
///     objective_medium_refractive_index: 1.0,
///     defocus_distance: None,
///     pupil_aberration: None,
/// };
/// optics.validate()?;
/// assert_eq!(optics.object_pixel_size(), 6.5e-6 / 4.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optics {
    /// Illumination wavelength in vacuum, in metres.
    pub wavelength_vacuum_m: f64,
    /// Objective numerical aperture; must not exceed [`Self::objective_medium_refractive_index`].
    pub objective_na: f64,
    /// Lateral image magnification, as a positive dimensionless ratio.
    pub magnification: f64,
    /// Physical detector-pixel pitch in metres.
    ///
    /// After division by [`Self::magnification`], this must be strictly less
    /// than `wavelength_vacuum_m / (2 * objective_na)`.
    pub camera_pixel_size: f64,
    /// Refractive index between illumination sources and the sample.
    pub illumination_refractive_index: f64,
    /// Refractive index in the objective-side medium.
    pub objective_medium_refractive_index: f64,
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
#[serde(deny_unknown_fields)]
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
    /// Validates positive finite physical parameters, coherent-field sampling,
    /// objective-medium compatibility, finite optional defocus, and the optional
    /// [`PupilAberration`].
    ///
    /// Coherent-field sampling requires
    /// `camera_pixel_size / magnification < wavelength_vacuum_m / (2 * objective_na)`.
    /// Equality is invalid because it places the pupil cutoff on the discrete
    /// Nyquist boundary.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("wavelength_vacuum_m", self.wavelength_vacuum_m),
            ("objective_na", self.objective_na),
            ("magnification", self.magnification),
            ("camera_pixel_size", self.camera_pixel_size),
            (
                "illumination_refractive_index",
                self.illumination_refractive_index,
            ),
            (
                "objective_medium_refractive_index",
                self.objective_medium_refractive_index,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: format!("must be finite and positive, got {value}"),
                });
            }
        }
        if self.objective_na > self.objective_medium_refractive_index {
            return Err(Error::InvalidParameter {
                name: "objective_na",
                reason: "cannot exceed the medium refractive index".into(),
            });
        }
        let object_pixel_size = self.object_pixel_size();
        let coherent_sampling_limit = self.wavelength_vacuum_m / (2.0 * self.objective_na);
        if !object_pixel_size.is_finite()
            || !coherent_sampling_limit.is_finite()
            || coherent_sampling_limit <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "camera_pixel_size",
                reason: "camera_pixel_size / magnification and wavelength_vacuum_m / (2 * objective_na) must produce finite positive sampling scales"
                    .into(),
            });
        }
        if object_pixel_size >= coherent_sampling_limit {
            return Err(Error::InvalidParameter {
                name: "camera_pixel_size",
                reason: format!(
                    "object-plane pitch camera_pixel_size / magnification ({object_pixel_size:.6e} m) must be strictly less than wavelength_vacuum_m / (2 * objective_na) ({coherent_sampling_limit:.6e} m) for coherent-field sampling; use a smaller detector pixel, greater magnification, or lower objective NA"
                ),
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

    /// Returns `2π n_illumination / wavelength_vacuum`, in radians per metre.
    pub fn illumination_wavenumber(&self) -> f64 {
        std::f64::consts::TAU * self.illumination_refractive_index / self.wavelength_vacuum_m
    }

    /// Returns the objective-medium angular wavenumber in radians per metre.
    pub fn objective_medium_wavenumber(&self) -> f64 {
        std::f64::consts::TAU * self.objective_medium_refractive_index / self.wavelength_vacuum_m
    }
}
