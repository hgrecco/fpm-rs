use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AberrationModel {
    pub defocus: f64,
    pub astigmatism: f64,
    pub coma: f64,
    pub spherical: f64,
    pub edge_apodization: f64,
    /// Intensity attenuation versus normalized illumination radius. A value
    /// `s` gives edge transmission `exp(-s)` and unit on-axis transmission.
    #[serde(default)]
    pub illumination_vignetting: f64,
}

impl AberrationModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn defocus(mut self, radians: f64) -> Self {
        self.defocus = radians;
        self
    }

    pub fn astigmatism(mut self, radians: f64) -> Self {
        self.astigmatism = radians;
        self
    }

    pub fn coma(mut self, radians: f64) -> Self {
        self.coma = radians;
        self
    }

    pub fn spherical(mut self, radians: f64) -> Self {
        self.spherical = radians;
        self
    }

    pub fn edge_apodization(mut self, strength: f64) -> Self {
        self.edge_apodization = strength;
        self
    }

    pub fn illumination_vignetting(mut self, strength: f64) -> Self {
        self.illumination_vignetting = strength;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if [
            self.defocus,
            self.astigmatism,
            self.coma,
            self.spherical,
            self.edge_apodization,
            self.illumination_vignetting,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || self.edge_apodization < 0.0
            || self.illumination_vignetting < 0.0
        {
            return Err(Error::InvalidParameter {
                name: "aberration",
                reason: "phase coefficients must be finite; apodization and vignetting must be finite and non-negative"
                    .into(),
            });
        }
        Ok(())
    }
}
