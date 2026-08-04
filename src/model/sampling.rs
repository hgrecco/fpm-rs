use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Storage and sign convention relating physical wave vectors to Fourier indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateConvention {
    /// Spectra are stored with zero frequency at the array centre. Positive
    /// illumination k shifts the crop centre toward increasing array indices.
    CenteredPositiveK,
}

/// Physical sampling metadata for low- and high-resolution grids.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sampling {
    /// Sample-plane low-resolution pixel pitch in metres.
    pub low_res_pixel_size: f64,
    /// Sample-plane high-resolution reconstruction pixel pitch in metres.
    pub high_res_pixel_size: f64,
    /// Fourier angular-frequency spacing along columns (`x`), in radians per metre.
    pub dkx: f64,
    /// Fourier angular-frequency spacing along rows (`y`), in radians per metre.
    pub dky: f64,
    /// Optional vacuum illumination wavelength in metres.
    pub wavelength: Option<f64>,
    /// Optional dimensionless synthetic numerical aperture.
    pub synthetic_na: Option<f64>,
    /// Convention used to store zero frequency and map positive wave vectors.
    pub coordinate_convention: CoordinateConvention,
}

impl Sampling {
    /// Creates centered-positive-k sampling from positive finite SI spacings.
    pub fn new(
        low_res_pixel_size: f64,
        high_res_pixel_size: f64,
        dkx: f64,
        dky: f64,
    ) -> Result<Self> {
        let sampling = Self {
            low_res_pixel_size,
            high_res_pixel_size,
            dkx,
            dky,
            wavelength: None,
            synthetic_na: None,
            coordinate_convention: CoordinateConvention::CenteredPositiveK,
        };
        sampling.validate()?;
        Ok(sampling)
    }

    /// Checks positive finite pixel/frequency spacings and positive finite optional
    /// wavelength and synthetic numerical aperture.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("low_res_pixel_size", self.low_res_pixel_size),
            ("high_res_pixel_size", self.high_res_pixel_size),
            ("dkx", self.dkx),
            ("dky", self.dky),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: "must be finite and positive".into(),
                });
            }
        }
        if self.high_res_pixel_size > self.low_res_pixel_size {
            return Err(Error::InvalidParameter {
                name: "high_res_pixel_size",
                reason: "must not exceed low-resolution pixel size".into(),
            });
        }
        if self
            .wavelength
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "wavelength",
                reason: "must be finite and positive when present".into(),
            });
        }
        if self
            .synthetic_na
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "synthetic_na",
                reason: "must be finite and positive when present".into(),
            });
        }
        Ok(())
    }
}
