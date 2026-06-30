use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateConvention {
    /// Spectra are stored with zero frequency at the array centre. Positive
    /// illumination k shifts the crop centre toward increasing array indices.
    CenteredPositiveK,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sampling {
    pub low_res_pixel_size: f64,
    pub high_res_pixel_size: f64,
    pub dkx: f64,
    pub dky: f64,
    pub wavelength: Option<f64>,
    pub synthetic_na: Option<f64>,
    pub coordinate_convention: CoordinateConvention,
}

impl Sampling {
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
