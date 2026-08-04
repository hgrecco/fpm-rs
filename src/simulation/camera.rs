use rand::Rng;
use rand_distr::{Distribution, Normal, Poisson};
use serde::{Deserialize, Serialize};

use crate::{Result, array_layout::checked_len_2d, error::Error, model::ImagePlaneModel};

/// Concrete detector pipeline from optical intensity to digital counts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraModel {
    /// Expected photoelectrons per pixel at unit optical intensity.
    pub photons_per_pixel: f64,
    /// Linear conversion gain in digital camera counts per collected electron.
    pub gain_counts_per_electron: f64,
    /// Additive electronic bias in camera counts.
    pub offset_counts: f64,
    /// Gaussian read-noise standard deviation in electrons per pixel.
    pub read_noise_electrons: f64,
    /// Expected dark-current electrons per pixel per simulated exposure.
    pub dark_current_electrons: f64,
    /// Whether to Poisson-sample photoelectrons and dark current.
    pub shot_noise: bool,
    /// Multiplicative detector sensitivity for each pixel.
    pub pixel_sensitivity: Option<Vec<f64>>,
    /// Optional digitizer bit depth; maximum code is `2^bits - 1` counts.
    pub bit_depth: Option<u8>,
    /// Optional upper clipping threshold in camera counts.
    pub saturation_counts: Option<f64>,
    /// Whether to round final camera counts to the nearest integer.
    pub quantize: bool,
    /// Row-major pixel indices replaced after all other detector effects.
    pub bad_pixels: Vec<usize>,
    /// Replacement value for [`Self::bad_pixels`], in camera counts.
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
            shot_noise: false,
            pixel_sensitivity: None,
            bit_depth: Some(16),
            saturation_counts: None,
            quantize: true,
            bad_pixels: Vec::new(),
            bad_pixel_value_counts: None,
        }
    }
}

impl CameraModel {
    /// Creates the default noiseless linear detector with 1000 photons per unit intensity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Unit detector response without noise, clipping, or quantization.
    pub fn ideal() -> Self {
        Self {
            photons_per_pixel: 1.0,
            gain_counts_per_electron: 1.0,
            offset_counts: 0.0,
            read_noise_electrons: 0.0,
            dark_current_electrons: 0.0,
            shot_noise: false,
            pixel_sensitivity: None,
            bit_depth: None,
            saturation_counts: None,
            quantize: false,
            bad_pixels: Vec::new(),
            bad_pixel_value_counts: None,
        }
    }

    /// Sets the positive expected photoelectrons per pixel at unit optical intensity.
    pub fn photons_per_pixel(mut self, value: f64) -> Self {
        self.photons_per_pixel = value;
        self
    }

    /// Sets positive linear gain in camera counts per electron.
    pub fn gain(mut self, counts_per_electron: f64) -> Self {
        self.gain_counts_per_electron = counts_per_electron;
        self
    }

    /// Sets finite additive camera-count bias.
    pub fn offset_counts(mut self, value: f64) -> Self {
        self.offset_counts = value;
        self
    }

    /// Sets non-negative Gaussian read-noise standard deviation in electrons.
    pub fn read_noise_electrons(mut self, value: f64) -> Self {
        self.read_noise_electrons = value;
        self
    }

    /// Sets non-negative expected dark-current electrons per pixel and exposure.
    pub fn dark_current_electrons(mut self, value: f64) -> Self {
        self.dark_current_electrons = value;
        self
    }

    /// Enables or disables Poisson sampling of photoelectrons plus dark current.
    pub fn shot_noise(mut self, enabled: bool) -> Self {
        self.shot_noise = enabled;
        self
    }

    /// Sets one finite non-negative sensitivity multiplier per row-major detector pixel.
    pub fn pixel_sensitivity(mut self, values: Vec<f64>) -> Self {
        self.pixel_sensitivity = Some(values);
        self
    }

    /// Sets digitizer bit depth in `1..=53`, enabling the corresponding maximum code.
    pub fn bit_depth(mut self, bits: u8) -> Self {
        self.bit_depth = Some(bits);
        self
    }

    /// Sets a finite non-negative saturation threshold in camera counts.
    pub fn saturation(mut self, counts: f64) -> Self {
        self.saturation_counts = Some(counts);
        self
    }

    /// Enables or disables rounding final counts to integer-valued `f64` values.
    pub fn quantize(mut self, enabled: bool) -> Self {
        self.quantize = enabled;
        self
    }

    /// Replaces unique row-major detector indices with `value_counts` after measurement.
    pub fn bad_pixels(mut self, indices: Vec<usize>, value_counts: f64) -> Self {
        self.bad_pixels = indices;
        self.bad_pixel_value_counts = Some(value_counts);
        self
    }

    /// Validates gains, noise parameters, clipping/quantization settings, sensitivities,
    /// and uniqueness of bad-pixel indices independently of frame shape.
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
        if self.pixel_sensitivity.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        }) {
            return Err(Error::InvalidParameter {
                name: "pixel_sensitivity",
                reason: "values must be finite and non-negative".into(),
            });
        }
        let mut bad_pixels = self.bad_pixels.clone();
        bad_pixels.sort_unstable();
        if bad_pixels.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidParameter {
                name: "bad_pixels",
                reason: "indices must be unique".into(),
            });
        }
        Ok(())
    }

    /// Additionally checks sensitivity length and bad-pixel bounds for `frame_len` pixels.
    pub fn validate_for_frame(&self, frame_len: usize) -> Result<()> {
        self.validate()?;
        if self
            .pixel_sensitivity
            .as_ref()
            .is_some_and(|values| values.len() != frame_len)
        {
            return Err(Error::InvalidParameter {
                name: "pixel_sensitivity",
                reason: format!("must contain {frame_len} values"),
            });
        }
        if self.bad_pixels.iter().any(|&pixel| pixel >= frame_len) {
            return Err(Error::InvalidParameter {
                name: "bad_pixels",
                reason: format!("indices must be below the frame size {frame_len}"),
            });
        }
        Ok(())
    }

    /// Converts one optical-intensity frame into detector counts in place.
    pub fn measure_frame<R: Rng + ?Sized>(&self, intensity: &mut [f64], rng: &mut R) -> Result<()> {
        self.validate_for_frame(intensity.len())?;
        let read_noise = (self.read_noise_electrons > 0.0)
            .then(|| Normal::new(0.0, self.read_noise_electrons))
            .transpose()
            .map_err(|error| Error::Numerical(error.to_string()))?;
        for (pixel, value) in intensity.iter_mut().enumerate() {
            let sensitivity = self
                .pixel_sensitivity
                .as_ref()
                .map_or(1.0, |values| values[pixel]);
            let expected_electrons =
                value.max(0.0) * sensitivity * self.photons_per_pixel + self.dark_current_electrons;
            let mut electrons = if self.shot_noise && expected_electrons > 0.0 {
                Poisson::new(expected_electrons)
                    .map_err(|error| Error::Numerical(error.to_string()))?
                    .sample(rng)
            } else {
                expected_electrons
            };
            if let Some(distribution) = &read_noise {
                electrons += distribution.sample(rng);
            }
            let mut counts = electrons * self.gain_counts_per_electron + self.offset_counts;
            counts = counts.clamp(0.0, self.maximum_count());
            if self.quantize {
                counts = counts.round();
            }
            *value = counts;
        }
        for &pixel in &self.bad_pixels {
            let mut value = self
                .bad_pixel_value_counts
                .unwrap_or_else(|| self.maximum_count())
                .clamp(0.0, self.maximum_count());
            if self.quantize {
                value = value.round();
            }
            intensity[pixel] = value;
        }
        Ok(())
    }

    /// Compiles known uniform linear response into a reconstruction model.
    ///
    /// Pixel sensitivity, stochastic noise, clipping, quantization, and bad
    /// pixels remain detector effects and are not compiled into the model.
    pub fn compile_reconstruction_model(
        &self,
        mut model: ImagePlaneModel,
    ) -> Result<ImagePlaneModel> {
        let image_len = checked_len_2d(model.image_shape)?;
        self.validate_for_frame(image_len)?;
        let scale = self.photons_per_pixel * self.gain_counts_per_electron;
        model.frame_gains = Some(match &model.frame_gains {
            Some(gains) => gains.iter().map(|gain| gain * scale).collect(),
            None => vec![scale; model.frame_count()],
        });
        let additive_counts =
            self.dark_current_electrons * self.gain_counts_per_electron + self.offset_counts;
        if let Some(background) = &mut model.background {
            for value in background {
                *value = *value * scale + additive_counts;
            }
        } else if additive_counts != 0.0 {
            model.background = Some(vec![additive_counts; image_len]);
        }
        model.validate()?;
        Ok(model)
    }

    fn maximum_count(&self) -> f64 {
        let digital_maximum = self
            .bit_depth
            .map_or(f64::INFINITY, |bits| (2_f64).powi(bits as i32) - 1.0);
        self.saturation_counts
            .unwrap_or(f64::INFINITY)
            .min(digital_maximum)
    }
}
