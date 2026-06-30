use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraModel {
    pub photons_per_pixel: f64,
    pub gain_counts_per_electron: f64,
    pub offset_counts: f64,
    pub read_noise_electrons: f64,
    pub dark_current_electrons: f64,
    pub bit_depth: Option<u8>,
    pub saturation_counts: Option<f64>,
    pub quantize: bool,
    pub bad_pixels: Vec<usize>,
    pub bad_pixel_value_counts: Option<f64>,
}

impl Default for CameraModel {
    fn default() -> Self {
        Self {
            photons_per_pixel: 1_000.0,
            gain_counts_per_electron: 1.0,
            offset_counts: 0.0,
            read_noise_electrons: 0.0,
            dark_current_electrons: 0.0,
            bit_depth: Some(16),
            saturation_counts: None,
            quantize: true,
            bad_pixels: Vec::new(),
            bad_pixel_value_counts: None,
        }
    }
}

impl CameraModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn photons_per_pixel(mut self, value: f64) -> Self {
        self.photons_per_pixel = value;
        self
    }

    pub fn gain(mut self, counts_per_electron: f64) -> Self {
        self.gain_counts_per_electron = counts_per_electron;
        self
    }

    pub fn offset_counts(mut self, value: f64) -> Self {
        self.offset_counts = value;
        self
    }

    pub fn read_noise_electrons(mut self, value: f64) -> Self {
        self.read_noise_electrons = value;
        self
    }

    pub fn dark_current_electrons(mut self, value: f64) -> Self {
        self.dark_current_electrons = value;
        self
    }

    pub fn bit_depth(mut self, bits: u8) -> Self {
        self.bit_depth = Some(bits);
        self
    }

    pub fn saturation(mut self, counts: f64) -> Self {
        self.saturation_counts = Some(counts);
        self
    }

    pub fn quantize(mut self, enabled: bool) -> Self {
        self.quantize = enabled;
        self
    }

    pub fn bad_pixels(mut self, indices: Vec<usize>, value_counts: f64) -> Self {
        self.bad_pixels = indices;
        self.bad_pixel_value_counts = Some(value_counts);
        self
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("photons_per_pixel", self.photons_per_pixel),
            ("gain_counts_per_electron", self.gain_counts_per_electron),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: "must be finite and positive".into(),
                });
            }
        }
        if !self.offset_counts.is_finite()
            || !self.read_noise_electrons.is_finite()
            || self.read_noise_electrons < 0.0
            || !self.dark_current_electrons.is_finite()
            || self.dark_current_electrons < 0.0
        {
            return Err(Error::InvalidParameter {
                name: "camera noise/offset",
                reason: "offset must be finite; read noise and dark current must be non-negative"
                    .into(),
            });
        }
        if self.bit_depth.is_some_and(|bits| bits == 0 || bits > 32) {
            return Err(Error::InvalidParameter {
                name: "bit_depth",
                reason: "must be between 1 and 32".into(),
            });
        }
        if self
            .saturation_counts
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "saturation_counts",
                reason: "must be finite and positive".into(),
            });
        }
        if self
            .bad_pixel_value_counts
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "bad_pixel_value_counts",
                reason: "must be finite and non-negative".into(),
            });
        }
        if !self.bad_pixels.is_empty()
            && self.bad_pixel_value_counts.is_none()
            && !self.maximum_count().is_finite()
        {
            return Err(Error::InvalidParameter {
                name: "bad_pixels",
                reason: "a finite bad-pixel value or camera maximum is required".into(),
            });
        }
        Ok(())
    }

    pub(crate) fn maximum_count(&self) -> f64 {
        let digital_maximum = self
            .bit_depth
            .map_or(f64::INFINITY, |bits| (2_f64).powi(bits as i32) - 1.0);
        self.saturation_counts
            .unwrap_or(f64::INFINITY)
            .min(digital_maximum)
    }
}
