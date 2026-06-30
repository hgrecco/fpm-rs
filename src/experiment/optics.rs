use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Physical microscope parameters in SI units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Optics {
    pub wavelength: f64,
    pub objective_na: f64,
    pub magnification: f64,
    pub camera_pixel_size: f64,
    pub medium_index: f64,
    pub defocus: Option<f64>,
    pub initial_pupil_aberration: Option<PupilAberration>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PupilAberration {
    /// Defocus Zernike-like coefficient in radians at the pupil edge.
    pub defocus: f64,
    /// Two-fold astigmatism coefficient in radians.
    pub astigmatism: f64,
    /// Horizontal coma coefficient in radians.
    pub coma: f64,
    /// Spherical coefficient in radians.
    pub spherical: f64,
}

impl PupilAberration {
    pub fn validate(&self) -> Result<()> {
        if [self.defocus, self.astigmatism, self.coma, self.spherical]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "initial_pupil_aberration",
                reason: "all coefficients must be finite".into(),
            });
        }
        Ok(())
    }
}

impl Optics {
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
        if self.defocus.is_some_and(|value| !value.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "defocus",
                reason: "must be finite".into(),
            });
        }
        if let Some(aberration) = &self.initial_pupil_aberration {
            aberration.validate()?;
        }
        Ok(())
    }

    pub fn object_pixel_size(&self) -> f64 {
        self.camera_pixel_size / self.magnification
    }

    pub fn medium_wavenumber(&self) -> f64 {
        std::f64::consts::TAU * self.medium_index / self.wavelength
    }
}
