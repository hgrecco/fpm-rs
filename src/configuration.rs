//! Versioned experiment and simulation configuration.
//!
//! The schema stores both concrete experiment descriptions and their compiled
//! models. Loading validates the descriptions, compiled models, dimensions,
//! source geometry, camera, and acquisition effects before returning data to a
//! caller.

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    array_layout::checked_len_2d,
    error::Error,
    experiment::{Illumination, Optics, ResolvedIllumination},
    model::{ImagePlaneModel, ReconstructionShape},
    simulation::{CameraModel, IlluminationAcquisitionErrors},
};

/// Current serialized experiment/simulation configuration format.
pub const CONFIGURATION_FORMAT_VERSION: u32 = 2;

/// Concrete optical experiment description used to compile an image-plane model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentDescription {
    /// Physical microscope parameters in SI units.
    pub optics: Optics,
    /// Concrete source geometry or calibrated illumination description.
    pub illumination: Illumination,
    /// Optional optical intensity background, broadcast or stored per frame.
    pub optical_background: Option<Vec<f64>>,
}

impl ExperimentDescription {
    /// Creates an experiment without optical intensity background.
    pub fn new(optics: Optics, illumination: Illumination) -> Self {
        Self {
            optics,
            illumination,
            optical_background: None,
        }
    }

    /// Sets non-negative optical intensity background values.
    ///
    /// During compilation the vector must contain either one row-major low-resolution
    /// frame or a full `(frame, row, column)` acquisition stack.
    pub fn with_optical_background(mut self, background: Vec<f64>) -> Self {
        self.optical_background = Some(background);
        self
    }

    /// Compiles optics and illumination into an owned [`ImagePlaneModel`].
    pub fn compile(
        &self,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
    ) -> Result<ImagePlaneModel> {
        let resolved = self.illumination.resolve(&self.optics)?;
        self.compile_resolved(image_shape, reconstruction_shape, &resolved)
    }

    fn compile_resolved(
        &self,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
        resolved: &ResolvedIllumination,
    ) -> Result<ImagePlaneModel> {
        let mut model =
            ImagePlaneModel::compile(&self.optics, resolved, image_shape, reconstruction_shape)?;
        model.background = self.optical_background.clone();
        model.validate()?;
        Ok(model)
    }

    /// Validates optics, source compilation, gains, multiplexing, and finite non-negative
    /// optical background values.
    pub fn validate(&self) -> Result<()> {
        self.optics.validate()?;
        self.illumination.resolve(&self.optics)?;
        if self.optical_background.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        }) {
            return Err(Error::InvalidParameter {
                name: "optical_background",
                reason: "values must be finite and non-negative".into(),
            });
        }
        Ok(())
    }
}

/// True and assumed reconstruction models compiled from concrete descriptions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledModelPair {
    /// Model representing the physical acquisition used for simulation or evaluation.
    pub true_model: ImagePlaneModel,
    /// Model assumed by reconstruction, which may encode a deliberate mismatch.
    pub reconstruction_model: ImagePlaneModel,
}

impl CompiledModelPair {
    /// Validates both models and requires equal image/object shapes and frame counts.
    pub fn validate(&self) -> Result<()> {
        self.true_model.validate()?;
        self.reconstruction_model.validate()?;
        if self.true_model.image_shape != self.reconstruction_model.image_shape
            || self.true_model.reconstruction_shape
                != self.reconstruction_model.reconstruction_shape
            || self.true_model.frame_count() != self.reconstruction_model.frame_count()
        {
            return Err(Error::InvalidModel(
                "compiled true and reconstruction models must have matching image, object, and frame dimensions"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Complete, versioned configuration for simulation and reconstruction.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfiguration {
    /// Must equal [`CONFIGURATION_FORMAT_VERSION`].
    pub format_version: u32,
    /// Physical description used to compile the true acquisition model.
    pub true_experiment: ExperimentDescription,
    /// Physical description assumed by reconstruction.
    pub reconstruction_experiment: ExperimentDescription,
    /// Shared low-resolution shape as `(height, width)`.
    pub image_shape: (usize, usize),
    /// Shared high-resolution shape as `(height, width)`.
    pub reconstruction_shape: (usize, usize),
    /// Stored true and reconstruction models compiled from the descriptions.
    pub compiled_models: CompiledModelPair,
    /// Optional detector-count response and noise model.
    pub camera: Option<CameraModel>,
    /// Optional non-geometric illumination errors injected during acquisition.
    pub illumination_acquisition_errors: Option<IlluminationAcquisitionErrors>,
    /// Deterministic seed for simulation randomness.
    pub random_seed: u64,
}

impl SimulationConfiguration {
    /// Compiles true and assumed descriptions on one explicit or automatically selected
    /// reconstruction grid and validates the complete configuration.
    pub fn new(
        true_experiment: ExperimentDescription,
        reconstruction_experiment: ExperimentDescription,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
    ) -> Result<Self> {
        let true_resolved = true_experiment
            .illumination
            .resolve(&true_experiment.optics)?;
        let reconstruction_resolved = reconstruction_experiment
            .illumination
            .resolve(&reconstruction_experiment.optics)?;
        let true_bounds = ImagePlaneModel::crop_displacement_bounds(
            &true_experiment.optics,
            image_shape,
            true_resolved.k_vectors(),
        )?;
        let reconstruction_bounds = ImagePlaneModel::crop_displacement_bounds(
            &reconstruction_experiment.optics,
            image_shape,
            reconstruction_resolved.k_vectors(),
        )?;
        let reconstruction_shape = ImagePlaneModel::resolve_reconstruction_shape(
            image_shape,
            reconstruction_shape,
            &[true_bounds, reconstruction_bounds],
        )?;
        let compiled_models = CompiledModelPair {
            true_model: true_experiment.compile_resolved(
                image_shape,
                ReconstructionShape::Exact(reconstruction_shape),
                &true_resolved,
            )?,
            reconstruction_model: reconstruction_experiment.compile_resolved(
                image_shape,
                ReconstructionShape::Exact(reconstruction_shape),
                &reconstruction_resolved,
            )?,
        };
        let configuration = Self {
            format_version: CONFIGURATION_FORMAT_VERSION,
            true_experiment,
            reconstruction_experiment,
            image_shape,
            reconstruction_shape,
            compiled_models,
            camera: None,
            illumination_acquisition_errors: None,
            random_seed: 0,
        };
        configuration.validate()?;
        Ok(configuration)
    }

    /// Adds a camera model after validating it for the configured frame shape.
    pub fn with_camera(mut self, camera: CameraModel) -> Result<Self> {
        self.camera = Some(camera);
        self.validate()?;
        Ok(self)
    }

    /// Adds source gain variation, missing frames, or a source permutation and validates it.
    pub fn with_illumination_acquisition_errors(
        mut self,
        errors: IlluminationAcquisitionErrors,
    ) -> Result<Self> {
        self.illumination_acquisition_errors = Some(errors);
        self.validate()?;
        Ok(self)
    }

    /// Sets the deterministic random seed used for simulation.
    pub fn with_random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Returns the assumed model with known linear camera response compiled in.
    ///
    /// The serialized `compiled_models.reconstruction_model` remains the
    /// strict optical model. Use this helper when constructing a
    /// reconstruction problem from detector counts loaded through a saved
    /// configuration.
    pub fn reconstruction_model_for_counts(&self) -> Result<ImagePlaneModel> {
        match &self.camera {
            Some(camera) => camera
                .compile_reconstruction_model(self.compiled_models.reconstruction_model.clone()),
            None => Ok(self.compiled_models.reconstruction_model.clone()),
        }
    }

    /// Checks format version; descriptions and compiled models; shape/frame agreement;
    /// camera constraints; and acquisition-error counts, indices, and permutations.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != CONFIGURATION_FORMAT_VERSION {
            return Err(Error::InvalidParameter {
                name: "configuration format_version",
                reason: format!(
                    "expected {CONFIGURATION_FORMAT_VERSION}, got {}",
                    self.format_version
                ),
            });
        }
        self.true_experiment.validate()?;
        self.reconstruction_experiment.validate()?;
        self.compiled_models.validate()?;
        validate_compiled_description(
            &self.true_experiment,
            &self.compiled_models.true_model,
            self.image_shape,
            self.reconstruction_shape,
        )?;
        validate_compiled_description(
            &self.reconstruction_experiment,
            &self.compiled_models.reconstruction_model,
            self.image_shape,
            self.reconstruction_shape,
        )?;
        if let Some(camera) = &self.camera {
            camera.validate_for_frame(checked_len_2d(self.image_shape)?)?;
        }
        if let Some(errors) = &self.illumination_acquisition_errors {
            validate_acquisition_errors(errors, &self.compiled_models.true_model)?;
        }
        Ok(())
    }

    /// Validates and writes this versioned configuration as pretty-printed JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads and fully validates a versioned JSON configuration.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        let configuration: Self = serde_json::from_reader(reader)?;
        configuration.validate()?;
        Ok(configuration)
    }
}

fn validate_compiled_description(
    description: &ExperimentDescription,
    stored: &ImagePlaneModel,
    image_shape: (usize, usize),
    reconstruction_shape: (usize, usize),
) -> Result<()> {
    if stored.image_shape != image_shape || stored.reconstruction_shape != reconstruction_shape {
        return Err(Error::InvalidModel(
            "compiled model dimensions differ from configuration dimensions".into(),
        ));
    }
    let expected = description.compile(
        image_shape,
        ReconstructionShape::Exact(reconstruction_shape),
    )?;
    if expected.frame_count() != stored.frame_count()
        || expected.source_count() != stored.source_count()
        || expected.crop_indices.crops != stored.crop_indices.crops
        || expected.pupil.support != stored.pupil.support
        || !complex_values_close(
            expected.pupil.values.as_slice(),
            stored.pupil.values.as_slice(),
        )
        || !vectors_close(&expected.k_vectors, &stored.k_vectors)
        || !optional_values_close(
            expected.frame_gains.as_deref(),
            stored.frame_gains.as_deref(),
        )
        || !optional_values_close(expected.background.as_deref(), stored.background.as_deref())
        || !multiplexing_close(
            expected.multiplexing_matrix.as_deref(),
            stored.multiplexing_matrix.as_deref(),
        )
        || !sampling_close(&expected.sampling, &stored.sampling)
        || !offsets_close(
            expected.subpixel_offsets.as_deref(),
            stored.subpixel_offsets.as_deref(),
        )
    {
        return Err(Error::InvalidModel(
            "compiled model is inconsistent with its experiment description".into(),
        ));
    }
    Ok(())
}

fn complex_values_close(expected: &[crate::Complex64], actual: &[crate::Complex64]) -> bool {
    expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(expected, actual)| {
            close(expected.re, actual.re) && close(expected.im, actual.im)
        })
}

fn optional_values_close(expected: Option<&[f64]>, actual: Option<&[f64]>) -> bool {
    match (expected, actual) {
        (None, None) => true,
        (Some(expected), Some(actual)) => {
            expected.len() == actual.len()
                && expected
                    .iter()
                    .zip(actual)
                    .all(|(&expected, &actual)| close(expected, actual))
        }
        _ => false,
    }
}

fn multiplexing_close(
    expected: Option<&[Vec<crate::experiment::SourceWeight>]>,
    actual: Option<&[Vec<crate::experiment::SourceWeight>]>,
) -> bool {
    match (expected, actual) {
        (None, None) => true,
        (Some(expected), Some(actual)) => {
            expected.len() == actual.len()
                && expected.iter().zip(actual).all(|(expected, actual)| {
                    expected.len() == actual.len()
                        && expected.iter().zip(actual).all(
                            |(
                                &(expected_source, expected_weight),
                                &(actual_source, actual_weight),
                            )| {
                                expected_source == actual_source
                                    && close(expected_weight, actual_weight)
                            },
                        )
                })
        }
        _ => false,
    }
}

fn sampling_close(expected: &crate::model::Sampling, actual: &crate::model::Sampling) -> bool {
    expected.coordinate_convention == actual.coordinate_convention
        && close(expected.low_res_pixel_size, actual.low_res_pixel_size)
        && close(expected.high_res_pixel_size, actual.high_res_pixel_size)
        && close(expected.dkx, actual.dkx)
        && close(expected.dky, actual.dky)
        && optional_scalar_close(expected.wavelength, actual.wavelength)
        && optional_scalar_close(expected.synthetic_na, actual.synthetic_na)
}

fn optional_scalar_close(expected: Option<f64>, actual: Option<f64>) -> bool {
    match (expected, actual) {
        (None, None) => true,
        (Some(expected), Some(actual)) => close(expected, actual),
        _ => false,
    }
}

fn vectors_close(
    expected: &[crate::experiment::KVector],
    actual: &[crate::experiment::KVector],
) -> bool {
    expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(expected, actual)| {
            close(expected.kx, actual.kx) && close(expected.ky, actual.ky)
        })
}

fn offsets_close(
    expected: Option<&[crate::model::FourierOffset]>,
    actual: Option<&[crate::model::FourierOffset]>,
) -> bool {
    match (expected, actual) {
        (None, None) => true,
        (Some(expected), Some(actual)) => {
            expected.len() == actual.len()
                && expected.iter().zip(actual).all(|(expected, actual)| {
                    close(expected.row, actual.row) && close(expected.column, actual.column)
                })
        }
        _ => false,
    }
}

fn close(expected: f64, actual: f64) -> bool {
    (expected - actual).abs() <= 1e-12 * expected.abs().max(actual.abs()).max(1.0)
}

fn validate_acquisition_errors(
    errors: &IlluminationAcquisitionErrors,
    model: &ImagePlaneModel,
) -> Result<()> {
    if !errors.frame_gain_relative_std.is_finite() || errors.frame_gain_relative_std < 0.0 {
        return Err(Error::InvalidParameter {
            name: "frame_gain_relative_std",
            reason: "must be finite and non-negative".into(),
        });
    }
    let mut missing = errors.missing_frames.clone();
    missing.sort_unstable();
    if missing.iter().any(|&frame| frame >= model.frame_count())
        || missing.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err(Error::InvalidParameter {
            name: "missing_frames",
            reason: format!(
                "must contain unique frame indices below {}",
                model.frame_count()
            ),
        });
    }
    if let Some(permutation) = &errors.source_permutation {
        let mut sorted = permutation.clone();
        sorted.sort_unstable();
        if permutation.len() != model.source_count()
            || sorted
                .iter()
                .enumerate()
                .any(|(expected, &actual)| expected != actual)
        {
            return Err(Error::InvalidParameter {
                name: "source_permutation",
                reason: format!("must be a permutation of 0..{}", model.source_count()),
            });
        }
    }
    Ok(())
}
