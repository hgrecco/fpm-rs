use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use std::path::Path;

use crate::{
    Result,
    array_layout::{StandardArray2, checked_len_2d},
    complex,
    error::Error,
    image_io::{GrayscaleScaling, load_grayscale},
};

#[derive(Clone, Debug)]
pub struct SyntheticObject {
    pub(crate) field: StandardArray2<Complex64>,
    label: Option<String>,
}

impl SyntheticObject {
    /// Stores a finite complex field without copying its elements.
    ///
    /// The field must be non-empty and C-contiguous standard row-major layout.
    pub fn new(field: Array2<Complex64>) -> Result<Self> {
        let field = StandardArray2::try_from(field)?;
        validate_shape(field.dim())?;
        if field
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "field",
                reason: "values must have finite real and imaginary components".into(),
            });
        }
        Ok(Self { field, label: None })
    }

    pub fn from_amplitude_phase(
        amplitude: ArrayView2<'_, f64>,
        phase: ArrayView2<'_, f64>,
    ) -> Result<Self> {
        Self::new(complex::from_amplitude_phase(amplitude, phase)?)
    }

    pub fn amplitude_only(amplitude: Array2<f64>) -> Result<Self> {
        let amplitude = StandardArray2::try_from(amplitude)?;
        if amplitude
            .as_slice()
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "amplitude",
                reason: "values must be finite and non-negative".into(),
            });
        }
        Self::from_values(
            amplitude.dim(),
            amplitude
                .as_slice()
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
        )
    }

    pub fn phase_only(phase: Array2<f64>) -> Result<Self> {
        let phase = StandardArray2::try_from(phase)?;
        if phase.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "phase",
                reason: "values must be finite".into(),
            });
        }
        Self::from_values(
            phase.dim(),
            phase
                .as_slice()
                .iter()
                .map(|&value| Complex64::from_polar(1.0, value))
                .collect(),
        )
    }

    pub fn from_amplitude_image(path: impl AsRef<Path>) -> Result<Self> {
        let amplitude = load_grayscale(path, GrayscaleScaling::Unit)?;
        Self::amplitude_only(amplitude)
    }

    /// Loads normalized grayscale amplitude and phase images. Phase values map
    /// linearly from black/white to `-phase_extent`/`+phase_extent` radians.
    pub fn from_amplitude_phase_images(
        amplitude_path: impl AsRef<Path>,
        phase_path: impl AsRef<Path>,
        phase_extent: f64,
    ) -> Result<Self> {
        if !phase_extent.is_finite() || phase_extent <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "phase_extent",
                reason: "must be finite and positive".into(),
            });
        }
        let amplitude = load_grayscale(amplitude_path, GrayscaleScaling::Unit)?;
        let normalized_phase = load_grayscale(phase_path, GrayscaleScaling::Unit)?;
        if amplitude.dim() != normalized_phase.dim() {
            return Err(Error::InvalidShape(format!(
                "amplitude image shape {:?} differs from phase image shape {:?}",
                amplitude.dim(),
                normalized_phase.dim()
            )));
        }
        let phase = normalized_phase.mapv(|value| (2.0 * value - 1.0) * phase_extent);
        Self::from_amplitude_phase(amplitude.view(), phase.view())
    }

    pub fn constant(shape: (usize, usize), amplitude: f64, phase: f64) -> Result<Self> {
        validate_shape(shape)?;
        validate_amplitude(amplitude)?;
        let length = checked_len_2d(shape)?;
        Self::from_values(shape, vec![Complex64::from_polar(amplitude, phase); length])
    }

    pub fn phase_disk(shape: (usize, usize), radius_pixels: f64, phase_shift: f64) -> Result<Self> {
        validate_shape(shape)?;
        if !radius_pixels.is_finite() || radius_pixels <= 0.0 || !phase_shift.is_finite() {
            return Err(Error::InvalidParameter {
                name: "phase disk",
                reason: "radius must be positive and phase must be finite".into(),
            });
        }
        let center = ((shape.0 - 1) as f64 / 2.0, (shape.1 - 1) as f64 / 2.0);
        let mut values = Vec::with_capacity(checked_len_2d(shape)?);
        for row in 0..shape.0 {
            for column in 0..shape.1 {
                let radius = (row as f64 - center.0).hypot(column as f64 - center.1);
                values.push(Complex64::from_polar(
                    1.0,
                    if radius <= radius_pixels {
                        phase_shift
                    } else {
                        0.0
                    },
                ));
            }
        }
        Self::from_values(shape, values)
    }

    pub fn siemens_star(shape: (usize, usize), spokes: usize) -> Result<Self> {
        validate_shape(shape)?;
        if spokes < 2 {
            return Err(Error::InvalidParameter {
                name: "spokes",
                reason: "must be at least 2".into(),
            });
        }
        let center = ((shape.0 - 1) as f64 / 2.0, (shape.1 - 1) as f64 / 2.0);
        let maximum_radius = shape.0.min(shape.1) as f64 * 0.46;
        let mut values = Vec::with_capacity(checked_len_2d(shape)?);
        for row in 0..shape.0 {
            for column in 0..shape.1 {
                let y = row as f64 - center.0;
                let x = column as f64 - center.1;
                let radius = x.hypot(y);
                let amplitude =
                    if radius <= maximum_radius && ((x.atan2(y) * spokes as f64).sin() >= 0.0) {
                        0.25
                    } else {
                        1.0
                    };
                values.push(Complex64::new(amplitude, 0.0));
            }
        }
        Self::from_values(shape, values)
    }

    pub fn resolution_target(shape: (usize, usize)) -> Result<Self> {
        validate_shape(shape)?;
        let mut values = vec![Complex64::new(1.0, 0.0); checked_len_2d(shape)?];
        let groups = [2_usize, 3, 4, 6, 8];
        for (group, &period) in groups.iter().enumerate() {
            let top = group * shape.0 / groups.len();
            let bottom = (group + 1) * shape.0 / groups.len();
            for row in top..bottom {
                for column in 0..shape.1 {
                    let dark = if group.is_multiple_of(2) {
                        (column / period).is_multiple_of(2)
                    } else {
                        ((row - top) / period).is_multiple_of(2)
                    };
                    if dark {
                        values[row * shape.1 + column] = Complex64::new(0.2, 0.0);
                    }
                }
            }
        }
        Self::from_values(shape, values)
    }

    pub fn random_phase(shape: (usize, usize), standard_deviation: f64, seed: u64) -> Result<Self> {
        validate_shape(shape)?;
        if !standard_deviation.is_finite() || standard_deviation < 0.0 {
            return Err(Error::InvalidParameter {
                name: "standard_deviation",
                reason: "must be finite and non-negative".into(),
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let length = checked_len_2d(shape)?;
        let values = if standard_deviation == 0.0 {
            vec![Complex64::new(1.0, 0.0); length]
        } else {
            let distribution =
                Normal::new(0.0, standard_deviation).map_err(|error| Error::InvalidParameter {
                    name: "standard_deviation",
                    reason: error.to_string(),
                })?;
            (0..length)
                .map(|_| Complex64::from_polar(1.0, distribution.sample(&mut rng)))
                .collect()
        };
        Self::from_values(shape, values)
    }

    pub fn particle_field(shape: (usize, usize), particles: usize, seed: u64) -> Result<Self> {
        validate_shape(shape)?;
        let mut values = vec![Complex64::new(1.0, 0.0); checked_len_2d(shape)?];
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..particles {
            let row = rng.random_range(0..shape.0);
            let column = rng.random_range(0..shape.1);
            values[row * shape.1 + column] = Complex64::new(0.1, 0.0);
        }
        Self::from_values(shape, values)
    }

    /// A deterministic mixed amplitude/phase target useful for smoke tests.
    pub fn mixed_test_pattern(shape: (usize, usize)) -> Result<Self> {
        validate_shape(shape)?;
        let center = ((shape.0 - 1) as f64 / 2.0, (shape.1 - 1) as f64 / 2.0);
        let scale = shape.0.min(shape.1) as f64;
        let mut values = Vec::with_capacity(checked_len_2d(shape)?);
        for row in 0..shape.0 {
            for column in 0..shape.1 {
                let y = row as f64 - center.0;
                let x = column as f64 - center.1;
                let radius = x.hypot(y);
                let amplitude = if (x.abs() < 0.12 * scale && y.abs() < 0.35 * scale)
                    || radius < 0.12 * scale
                {
                    0.45
                } else {
                    1.0
                };
                let phase = 0.8 * (-radius * radius / (0.08 * scale * scale)).exp()
                    + 0.25 * (std::f64::consts::TAU * x / scale).sin();
                values.push(Complex64::from_polar(amplitude, phase));
            }
        }
        Self::from_values(shape, values)
    }

    /// Smooth random phase and absorption blobs approximating a weak biological
    /// specimen. This is a test object, not a tissue-specific physical model.
    pub fn biological_like(shape: (usize, usize), features: usize, seed: u64) -> Result<Self> {
        validate_shape(shape)?;
        if features == 0 {
            return Err(Error::InvalidParameter {
                name: "features",
                reason: "must be greater than zero".into(),
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let minimum_size = shape.0.min(shape.1) as f64;
        let lower_sigma = (0.02 * minimum_size).max(1.0);
        let upper_sigma = (0.12 * minimum_size).max(lower_sigma + 0.1);
        let blobs: Vec<_> = (0..features)
            .map(|_| {
                (
                    rng.random_range(0.0..shape.0 as f64),
                    rng.random_range(0.0..shape.1 as f64),
                    rng.random_range(lower_sigma..upper_sigma),
                    rng.random_range(-0.4..1.0),
                    rng.random_range(0.0..0.25),
                )
            })
            .collect();
        let mut values = Vec::with_capacity(checked_len_2d(shape)?);
        for row in 0..shape.0 {
            for column in 0..shape.1 {
                let mut phase = 0.0;
                let mut absorption = 0.0;
                for &(center_row, center_column, sigma, phase_strength, absorption_strength) in
                    &blobs
                {
                    let squared_radius =
                        (row as f64 - center_row).powi(2) + (column as f64 - center_column).powi(2);
                    let profile = (-squared_radius / (2.0 * sigma * sigma)).exp();
                    phase += phase_strength * profile;
                    absorption += absorption_strength * profile;
                }
                values.push(Complex64::from_polar(
                    (1.0_f64 - absorption).max(0.1),
                    phase,
                ));
            }
        }
        Self::from_values(shape, values)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.field.dim()
    }

    pub fn field(&self) -> ArrayView2<'_, Complex64> {
        self.field.ndarray_view()
    }

    pub fn field_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.field.ndarray_view_mut()
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    fn from_values(shape: (usize, usize), values: Vec<Complex64>) -> Result<Self> {
        Ok(Self {
            field: StandardArray2::from_shape_vec(shape, values)?,
            label: None,
        })
    }
}

impl TryFrom<Array2<Complex64>> for SyntheticObject {
    type Error = Error;

    fn try_from(field: Array2<Complex64>) -> Result<Self> {
        Self::new(field)
    }
}

fn validate_amplitude(amplitude: f64) -> Result<()> {
    if !amplitude.is_finite() || amplitude < 0.0 {
        Err(Error::InvalidParameter {
            name: "amplitude",
            reason: "must be finite and non-negative".into(),
        })
    } else {
        Ok(())
    }
}

fn validate_shape(shape: (usize, usize)) -> Result<()> {
    if shape.0 == 0 || shape.1 == 0 {
        Err(Error::InvalidShape(format!(
            "synthetic object dimensions must be non-zero, got {shape:?}"
        )))
    } else {
        Ok(())
    }
}
