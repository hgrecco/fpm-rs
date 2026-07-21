use ndarray::Array2;
use num_complex::Complex64;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

use crate::{
    Result,
    array_layout::checked_len_2d,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
    measurements::{FrameMetadata, MeasurementStack},
    model::{ForwardModel, ImagePlaneModel, fftshift_copy},
};

use super::{
    CameraModel, IlluminationAcquisitionErrors, SimulationResult, SyntheticObject,
    result::SimulationParameters,
};

pub struct Simulator {
    true_model: ImagePlaneModel,
    reconstruction_model: Option<ImagePlaneModel>,
    object: Option<SyntheticObject>,
    camera: Option<CameraModel>,
    illumination_acquisition_errors: Option<IlluminationAcquisitionErrors>,
    seed: u64,
    ideal: bool,
}

impl Simulator {
    /// Creates a simulator whose input model describes the true experiment.
    ///
    /// Use [`Self::reconstruction_model`] with a separately compiled assumed
    /// optical model to simulate geometry or pupil mismatch.
    pub fn new(true_model: ImagePlaneModel) -> Self {
        Self {
            true_model,
            reconstruction_model: None,
            object: None,
            camera: None,
            illumination_acquisition_errors: None,
            seed: 0,
            ideal: false,
        }
    }

    pub fn ideal(model: ImagePlaneModel) -> Self {
        Self {
            ideal: true,
            ..Self::new(model)
        }
    }

    pub fn object(mut self, object: SyntheticObject) -> Self {
        self.object = Some(object);
        self
    }

    pub fn camera(mut self, camera: CameraModel) -> Self {
        self.camera = Some(camera);
        self.ideal = false;
        self
    }

    /// Adds non-geometric illumination acquisition errors.
    pub fn illumination_acquisition_errors(
        mut self,
        errors: IlluminationAcquisitionErrors,
    ) -> Self {
        self.illumination_acquisition_errors = Some(errors);
        self.ideal = false;
        self
    }

    /// Sets the optical model supplied to reconstruction after simulation.
    ///
    /// For an optical mismatch, compile this model and the true model from
    /// separate experiment descriptions. When a camera model is present,
    /// [`Self::simulate`] compiles the known linear camera response into this
    /// returned reconstruction model.
    pub fn reconstruction_model(mut self, model: ImagePlaneModel) -> Self {
        self.reconstruction_model = Some(model);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Simulates measurements and returns detector counts plus a count-space
    /// reconstruction model.
    ///
    /// This differs from serialized [`crate::configuration::SimulationConfiguration`],
    /// which stores a strict optical reconstruction model and exposes
    /// `reconstruction_model_for_counts()` for callers that load detector
    /// counts later.
    pub fn simulate(self) -> Result<SimulationResult> {
        self.true_model.validate()?;
        let object = self.object.ok_or(Error::InvalidParameter {
            name: "object",
            reason: "a ground-truth object is required".into(),
        })?;
        if object.shape() != self.true_model.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "object shape {:?} differs from reconstruction shape {:?}",
                object.shape(),
                self.true_model.reconstruction_shape
            )));
        }
        let mut reconstruction_model = self
            .reconstruction_model
            .unwrap_or_else(|| self.true_model.clone());
        reconstruction_model.validate()?;
        if reconstruction_model.image_shape != self.true_model.image_shape
            || reconstruction_model.reconstruction_shape != self.true_model.reconstruction_shape
            || reconstruction_model.frame_count() != self.true_model.frame_count()
        {
            return Err(Error::InvalidModel(
                "true and reconstruction models must have matching image, object, and frame dimensions"
                    .into(),
            ));
        }
        if let Some(camera) = &self.camera {
            reconstruction_model = camera.compile_reconstruction_model(reconstruction_model)?;
        }
        let mut true_model = self.true_model;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let missing_frames = self
            .illumination_acquisition_errors
            .as_ref()
            .map_or_else(Vec::new, |errors| errors.missing_frames.clone());
        if let Some(errors) = &self.illumination_acquisition_errors {
            apply_illumination_acquisition_errors(&mut true_model, errors, &mut rng)?;
        }
        true_model.validate()?;

        let object_spectrum = object_spectrum(&object, &true_model)?;
        let forward = ForwardModel::new(&true_model)?;
        let image_len = checked_len_2d(true_model.image_shape)?;
        let worker_count = std::thread::available_parallelism().map_or(1, |count| count.get());
        let stack_len = image_len
            .checked_mul(true_model.frame_count())
            .ok_or_else(|| Error::ShapeOverflow {
                shape: vec![
                    true_model.frame_count(),
                    true_model.image_shape.0,
                    true_model.image_shape.1,
                ],
            })?;
        let mut data = vec![0.0; stack_len];
        forward.forward_intensity_stack_into(
            object_spectrum.view(),
            &true_model.pupil,
            &mut data,
            worker_count,
        )?;
        // Keep camera processing serial so a seed produces the same
        // random stream regardless of the machine's available parallelism.
        for (frame, frame_data) in data.chunks_exact_mut(image_len).enumerate() {
            if missing_frames.contains(&frame) {
                // A failed source contributes no coherent field, but optical
                // background remains present and is still measured by the camera.
                for (pixel, value) in frame_data.iter_mut().enumerate() {
                    *value = true_model.background_value(frame, pixel)?;
                }
            }
            if let Some(camera) = &self.camera {
                camera.measure_frame(frame_data, &mut rng)?;
            }
        }
        let metadata = (0..true_model.frame_count())
            .map(|frame| {
                let mut metadata = FrameMetadata::new(frame);
                if missing_frames.contains(&frame) {
                    metadata.weight = 0.0;
                }
                metadata
            })
            .collect();
        let measurements = MeasurementStack::from_vec(data, true_model.image_shape, metadata)?;
        Ok(SimulationResult {
            measurements,
            ground_truth_object: object.field.into_inner(),
            true_model: true_model.clone(),
            reconstruction_model,
            camera: self.camera,
            illumination_acquisition_errors: self.illumination_acquisition_errors,
            parameters: SimulationParameters {
                ideal: self.ideal,
                frame_count: true_model.frame_count(),
                image_shape: true_model.image_shape,
                missing_frames,
            },
            random_seed: self.seed,
        })
    }
}

fn object_spectrum(object: &SyntheticObject, model: &ImagePlaneModel) -> Result<Array2<Complex64>> {
    let backend = CpuBackend::new(model.image_shape, model.reconstruction_shape)?;
    let mut values = object.field.as_slice().to_vec();
    let mut column =
        vec![Complex64::default(); model.image_shape.0.max(model.reconstruction_shape.0)];
    backend.fft2(
        &mut values,
        model.reconstruction_shape,
        FftDirection::Forward,
        &mut column,
    )?;
    let mut centered = vec![Complex64::default(); values.len()];
    fftshift_copy(&values, &mut centered, model.reconstruction_shape);
    Ok(Array2::from_shape_vec(
        model.reconstruction_shape,
        centered,
    )?)
}

fn apply_illumination_acquisition_errors(
    model: &mut ImagePlaneModel,
    errors: &IlluminationAcquisitionErrors,
    rng: &mut StdRng,
) -> Result<()> {
    if !errors.frame_gain_relative_std.is_finite() || errors.frame_gain_relative_std < 0.0 {
        return Err(Error::InvalidParameter {
            name: "frame_gain_relative_std",
            reason: "must be finite and non-negative".into(),
        });
    }
    let frame_count = model.frame_count();
    let source_count = model.source_count();
    let multiplexed = model.is_multiplexed();
    let mut sorted_missing = errors.missing_frames.clone();
    sorted_missing.sort_unstable();
    if sorted_missing.iter().any(|&frame| frame >= frame_count)
        || sorted_missing.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err(Error::InvalidParameter {
            name: "missing_frames",
            reason: format!("must contain unique frame indices below {frame_count}"),
        });
    }
    if let Some(permutation) = &errors.source_permutation {
        let mut sorted = permutation.clone();
        sorted.sort_unstable();
        if permutation.len() != source_count
            || sorted
                .iter()
                .enumerate()
                .any(|(expected, &actual)| expected != actual)
        {
            return Err(Error::InvalidParameter {
                name: "source_permutation",
                reason: format!("must be a permutation of 0..{source_count}"),
            });
        }
    }
    if let Some(permutation) = &errors.source_permutation {
        model.k_vectors = permutation
            .iter()
            .map(|&index| model.k_vectors[index])
            .collect();
        model.crop_indices.crops = permutation
            .iter()
            .map(|&index| model.crop_indices.crops[index])
            .collect();
        model.subpixel_offsets = model
            .subpixel_offsets
            .as_ref()
            .map(|offsets| permutation.iter().map(|&index| offsets[index]).collect());
        if !multiplexed && let Some(gains) = &model.frame_gains {
            model.frame_gains = Some(permutation.iter().map(|&index| gains[index]).collect());
        }
        let image_len = checked_len_2d(model.image_shape)?;
        let stack_len = image_len
            .checked_mul(frame_count)
            .ok_or_else(|| Error::ShapeOverflow {
                shape: vec![frame_count, model.image_shape.0, model.image_shape.1],
            })?;
        if !multiplexed
            && let Some(background) = &model.background
            && background.len() == stack_len
        {
            let mut reordered = Vec::with_capacity(background.len());
            for &index in permutation {
                let start = index
                    .checked_mul(image_len)
                    .ok_or_else(|| Error::ShapeOverflow {
                        shape: vec![index, model.image_shape.0, model.image_shape.1],
                    })?;
                let end = start
                    .checked_add(image_len)
                    .ok_or_else(|| Error::ShapeOverflow {
                        shape: vec![
                            index.saturating_add(1),
                            model.image_shape.0,
                            model.image_shape.1,
                        ],
                    })?;
                reordered.extend_from_slice(background.get(start..end).ok_or_else(|| {
                    Error::InvalidModel("background source permutation is out of range".into())
                })?);
            }
            model.background = Some(reordered);
        }
    }
    if errors.frame_gain_relative_std > 0.0 {
        let distribution = Normal::new(1.0, errors.frame_gain_relative_std)
            .map_err(|error| Error::Numerical(error.to_string()))?;
        let gains = model
            .frame_gains
            .get_or_insert_with(|| vec![1.0; frame_count]);
        for gain in gains {
            *gain *= distribution.sample(rng).max(0.01);
        }
    }
    Ok(())
}
