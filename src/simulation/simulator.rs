use num_complex::Complex64;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Poisson};

use crate::{
    Result,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
    measurements::{FrameMetadata, MeasurementStack},
    model::{ForwardModel, ImagePlaneModel, fftshift_copy},
};

use super::{
    AberrationModel, BackgroundModel, CameraModel, FlatFieldModel, IlluminationErrorModel,
    NoiseModel, SimulationResult, SyntheticObject, result::SimulationParameters,
};

pub struct Simulator {
    true_model: ImagePlaneModel,
    reconstruction_model: Option<ImagePlaneModel>,
    object: Option<SyntheticObject>,
    camera: Option<CameraModel>,
    noise: NoiseModel,
    illumination_errors: Option<IlluminationErrorModel>,
    aberration: Option<AberrationModel>,
    background: Option<BackgroundModel>,
    flat_field: Option<FlatFieldModel>,
    seed: u64,
    ideal: bool,
}

impl Simulator {
    pub fn new(true_model: ImagePlaneModel) -> Self {
        Self {
            true_model,
            reconstruction_model: None,
            object: None,
            camera: None,
            noise: NoiseModel::None,
            illumination_errors: None,
            aberration: None,
            background: None,
            flat_field: None,
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

    pub fn object(mut self, object: impl Into<SyntheticObject>) -> Self {
        self.object = Some(object.into());
        self
    }

    pub fn camera(mut self, camera: CameraModel) -> Self {
        self.camera = Some(camera);
        self.ideal = false;
        self
    }

    pub fn noise(mut self, noise: NoiseModel) -> Self {
        self.noise = noise;
        if noise != NoiseModel::None {
            self.ideal = false;
        }
        self
    }

    pub fn illumination_errors(mut self, errors: IlluminationErrorModel) -> Self {
        self.illumination_errors = Some(errors);
        self.ideal = false;
        self
    }

    pub fn aberration(mut self, aberration: AberrationModel) -> Self {
        self.aberration = Some(aberration);
        self.ideal = false;
        self
    }

    pub fn background(mut self, background: BackgroundModel) -> Self {
        self.background = Some(background);
        self.ideal = false;
        self
    }

    pub fn flat_field(mut self, flat_field: FlatFieldModel) -> Self {
        self.flat_field = Some(flat_field);
        self.ideal = false;
        self
    }

    pub fn reconstruction_model(mut self, model: ImagePlaneModel) -> Self {
        self.reconstruction_model = Some(model);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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
        if let Some(camera) = &self.camera {
            camera.validate()?;
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
            compile_camera_response(&mut reconstruction_model, camera)?;
        }
        let mut true_model = self.true_model;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let missing_frames = self
            .illumination_errors
            .as_ref()
            .map_or_else(Vec::new, |errors| errors.missing_sources.clone());
        if let Some(errors) = &self.illumination_errors {
            perturb_illumination(&mut true_model, errors, &mut rng)?;
        }
        if let Some(aberration) = &self.aberration {
            apply_aberration(&mut true_model, aberration)?;
        }
        if let Some(background) = &self.background {
            let image_len = true_model.image_shape.0 * true_model.image_shape.1;
            true_model.background = Some(match background {
                BackgroundModel::Constant(value) => {
                    if !value.is_finite() || *value < 0.0 {
                        return Err(Error::InvalidParameter {
                            name: "background",
                            reason: "constant background must be finite and non-negative".into(),
                        });
                    }
                    vec![*value; image_len]
                }
                BackgroundModel::PerPixel(values) => {
                    if values.len() != image_len || values.iter().any(|v| !v.is_finite()) {
                        return Err(Error::InvalidParameter {
                            name: "background",
                            reason: format!("must contain {image_len} finite values"),
                        });
                    }
                    values.clone()
                }
            });
        }
        true_model.validate()?;

        let object_spectrum = object_spectrum(&object, &true_model)?;
        let forward = ForwardModel::new(&true_model)?;
        let image_len = true_model.image_shape.0 * true_model.image_shape.1;
        if let Some(flat_field) = &self.flat_field
            && (flat_field.values.len() != image_len
                || flat_field
                    .values
                    .iter()
                    .any(|&value| !value.is_finite() || value < 0.0))
        {
            return Err(Error::InvalidParameter {
                name: "flat_field",
                reason: format!("must contain {image_len} finite non-negative values"),
            });
        }
        if let Some(camera) = &self.camera
            && camera.bad_pixels.iter().any(|&pixel| pixel >= image_len)
        {
            return Err(Error::InvalidParameter {
                name: "bad_pixels",
                reason: format!("indices must be below the frame size {image_len}"),
            });
        }
        let worker_count = std::thread::available_parallelism().map_or(1, |count| count.get());
        let mut data = vec![0.0; image_len * true_model.frame_count()];
        forward.forward_intensity_stack_into(
            &object_spectrum,
            &true_model.pupil,
            &mut data,
            worker_count,
        )?;
        // Keep camera/noise processing serial so a seed produces the same
        // random stream regardless of the machine's available parallelism.
        for (frame, frame_data) in data.chunks_exact_mut(image_len).enumerate() {
            let source_present = !missing_frames.contains(&frame);
            for (pixel, measured_value) in frame_data.iter_mut().enumerate() {
                let predicted = *measured_value;
                let ideal_value = if source_present {
                    predicted
                } else {
                    // A failed source contributes no coherent field, but camera-
                    // independent optical background remains present.
                    true_model.background_value(frame, pixel)?
                };
                let flat_value = self
                    .flat_field
                    .as_ref()
                    .map_or(1.0, |flat| flat.values[pixel]);
                let value = ideal_value * flat_value;
                let mut measured =
                    apply_camera_and_noise(value, self.camera.as_ref(), self.noise, &mut rng)?;
                if let Some(camera) = &self.camera
                    && camera.bad_pixels.contains(&pixel)
                {
                    measured = camera
                        .bad_pixel_value_counts
                        .unwrap_or_else(|| camera.maximum_count())
                        .clamp(0.0, camera.maximum_count());
                    if camera.quantize {
                        measured = measured.round();
                    }
                }
                *measured_value = measured;
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
            ground_truth_object: object.field,
            true_model: true_model.clone(),
            reconstruction_model,
            camera: self.camera,
            noise: self.noise,
            illumination_errors: self.illumination_errors,
            aberration: self.aberration,
            background: self.background,
            flat_field: self.flat_field,
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

fn compile_camera_response(model: &mut ImagePlaneModel, camera: &CameraModel) -> Result<()> {
    camera.validate()?;
    let scale = camera.photons_per_pixel * camera.gain_counts_per_electron;
    model.frame_gains = Some(match &model.frame_gains {
        Some(gains) => gains.iter().map(|gain| gain * scale).collect(),
        None => vec![scale; model.frame_count()],
    });
    let additive_counts =
        camera.dark_current_electrons * camera.gain_counts_per_electron + camera.offset_counts;
    if let Some(background) = &mut model.background {
        for value in background {
            *value = *value * scale + additive_counts;
        }
    } else if additive_counts != 0.0 {
        model.background = Some(vec![
            additive_counts;
            model.image_shape.0 * model.image_shape.1
        ]);
    }
    model.validate()
}

fn object_spectrum(
    object: &SyntheticObject,
    model: &ImagePlaneModel,
) -> Result<crate::Array2<Complex64>> {
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
    crate::Array2::from_vec(model.reconstruction_shape, centered)
}

fn apply_camera_and_noise(
    intensity: f64,
    camera: Option<&CameraModel>,
    noise: NoiseModel,
    rng: &mut StdRng,
) -> Result<f64> {
    let intensity = intensity.max(0.0);
    if let Some(camera) = camera {
        let expected_electrons =
            intensity * camera.photons_per_pixel + camera.dark_current_electrons;
        let mut electrons = match noise {
            NoiseModel::Poisson | NoiseModel::PoissonGaussian => {
                poisson_sample(expected_electrons, rng)?
            }
            NoiseModel::None | NoiseModel::Gaussian => expected_electrons,
        };
        if matches!(noise, NoiseModel::Gaussian | NoiseModel::PoissonGaussian)
            && camera.read_noise_electrons > 0.0
        {
            let normal = Normal::new(0.0, camera.read_noise_electrons).map_err(|error| {
                Error::Numerical(format!("cannot construct read-noise distribution: {error}"))
            })?;
            electrons += normal.sample(rng);
        }
        let mut counts = electrons * camera.gain_counts_per_electron + camera.offset_counts;
        counts = counts.clamp(0.0, camera.maximum_count());
        if camera.quantize {
            counts = counts.round();
        }
        Ok(counts)
    } else {
        match noise {
            NoiseModel::None => Ok(intensity),
            NoiseModel::Poisson => poisson_sample(intensity, rng),
            NoiseModel::Gaussian => gaussian_sample(intensity, 0.01 * intensity.max(1.0), rng),
            NoiseModel::PoissonGaussian => {
                let poisson = poisson_sample(intensity, rng)?;
                gaussian_sample(poisson, 0.01 * intensity.max(1.0), rng)
            }
        }
        .map(|value| value.max(0.0))
    }
}

fn poisson_sample(mean: f64, rng: &mut StdRng) -> Result<f64> {
    if mean == 0.0 {
        return Ok(0.0);
    }
    let distribution = Poisson::new(mean).map_err(|error| {
        Error::Numerical(format!("cannot construct Poisson distribution: {error}"))
    })?;
    Ok(distribution.sample(rng))
}

fn gaussian_sample(mean: f64, standard_deviation: f64, rng: &mut StdRng) -> Result<f64> {
    if standard_deviation == 0.0 {
        return Ok(mean);
    }
    let distribution = Normal::new(mean, standard_deviation).map_err(|error| {
        Error::Numerical(format!("cannot construct Gaussian distribution: {error}"))
    })?;
    Ok(distribution.sample(rng))
}

fn perturb_illumination(
    model: &mut ImagePlaneModel,
    errors: &IlluminationErrorModel,
    rng: &mut StdRng,
) -> Result<()> {
    if [
        errors.global_shift.0,
        errors.global_shift.1,
        errors.rotation_degrees,
        errors.scale_error,
        errors.per_led_jitter_std,
        errors.intensity_variation,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || errors.per_led_jitter_std < 0.0
        || errors.intensity_variation < 0.0
        || errors.scale_error <= -1.0
    {
        return Err(Error::InvalidParameter {
            name: "illumination_errors",
            reason: "values must be finite; deviations non-negative; scale error greater than -1"
                .into(),
        });
    }
    let frame_count = model.frame_count();
    let source_count = model.source_count();
    let multiplexed = model.is_multiplexed();
    let mut sorted_missing = errors.missing_sources.clone();
    sorted_missing.sort_unstable();
    if sorted_missing.iter().any(|&frame| frame >= frame_count)
        || sorted_missing.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err(Error::InvalidParameter {
            name: "missing_sources",
            reason: format!("must contain unique frame indices below {frame_count}"),
        });
    }
    if let Some(order) = &errors.source_order {
        let mut sorted = order.clone();
        sorted.sort_unstable();
        if order.len() != source_count
            || sorted
                .iter()
                .enumerate()
                .any(|(expected, &actual)| expected != actual)
        {
            return Err(Error::InvalidParameter {
                name: "source_order",
                reason: format!("must be a permutation of 0..{source_count}"),
            });
        }
    }
    let angle = errors.rotation_degrees.to_radians();
    let (sin_angle, cos_angle) = angle.sin_cos();
    let jitter = (errors.per_led_jitter_std > 0.0)
        .then(|| Normal::new(0.0, errors.per_led_jitter_std))
        .transpose()
        .map_err(|error| Error::Numerical(error.to_string()))?;
    for vector in &mut model.k_vectors {
        let x = vector.kx * (1.0 + errors.scale_error);
        let y = vector.ky * (1.0 + errors.scale_error);
        let jitter_x = jitter.as_ref().map_or(0.0, |normal| normal.sample(rng));
        let jitter_y = jitter.as_ref().map_or(0.0, |normal| normal.sample(rng));
        vector.kx =
            cos_angle * x - sin_angle * y + (errors.global_shift.0 + jitter_x) * model.sampling.dkx;
        vector.ky =
            sin_angle * x + cos_angle * y + (errors.global_shift.1 + jitter_y) * model.sampling.dky;
    }
    let mut subpixel_offsets = Vec::with_capacity(source_count);
    for (crop, vector) in model.crop_indices.crops.iter_mut().zip(&model.k_vectors) {
        let (new_crop, offset) = ImagePlaneModel::crop_for_k_vector(
            vector,
            &model.sampling,
            model.image_shape,
            model.reconstruction_shape,
        )?;
        *crop = new_crop;
        subpixel_offsets.push(offset);
    }
    model.subpixel_offsets = Some(subpixel_offsets);
    if errors.intensity_variation > 0.0 {
        let distribution = Normal::new(1.0, errors.intensity_variation)
            .map_err(|error| Error::Numerical(error.to_string()))?;
        model.frame_gains = Some(
            (0..model.frame_count())
                .map(|_| distribution.sample(rng).max(0.01))
                .collect(),
        );
    }
    if let Some(order) = &errors.source_order {
        model.k_vectors = order.iter().map(|&index| model.k_vectors[index]).collect();
        model.crop_indices.crops = order
            .iter()
            .map(|&index| model.crop_indices.crops[index])
            .collect();
        model.subpixel_offsets = model
            .subpixel_offsets
            .as_ref()
            .map(|offsets| order.iter().map(|&index| offsets[index]).collect());
        if !multiplexed && let Some(gains) = &model.frame_gains {
            model.frame_gains = Some(order.iter().map(|&index| gains[index]).collect());
        }
        let image_len = model.image_shape.0 * model.image_shape.1;
        if !multiplexed
            && let Some(background) = &model.background
            && background.len() == image_len * frame_count
        {
            let mut reordered = Vec::with_capacity(background.len());
            for &index in order {
                let start = index * image_len;
                reordered.extend_from_slice(&background[start..start + image_len]);
            }
            model.background = Some(reordered);
        }
    }
    Ok(())
}

fn apply_aberration(model: &mut ImagePlaneModel, aberration: &AberrationModel) -> Result<()> {
    aberration.validate()?;
    let shape = model.image_shape;
    let radius_scale = (shape.0.min(shape.1) as f64 / 2.0).max(1.0);
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let pixel = row * shape.1 + column;
            if !model.pupil.support[pixel] {
                continue;
            }
            let y = (row as f64 - shape.0 as f64 / 2.0) / radius_scale;
            let x = (column as f64 - shape.1 as f64 / 2.0) / radius_scale;
            let rho = x.hypot(y).min(1.0);
            let theta = y.atan2(x);
            let phase = aberration.defocus * (2.0 * rho * rho - 1.0)
                + aberration.astigmatism * rho * rho * (2.0 * theta).cos()
                + aberration.coma * (3.0 * rho.powi(3) - 2.0 * rho) * theta.cos()
                + aberration.spherical * (6.0 * rho.powi(4) - 6.0 * rho * rho + 1.0);
            let amplitude = (-aberration.edge_apodization * rho * rho).exp();
            model.pupil.values.as_mut_slice()[pixel] *= Complex64::from_polar(amplitude, phase);
        }
    }
    apply_illumination_vignetting(model, aberration.illumination_vignetting);
    Ok(())
}

fn apply_illumination_vignetting(model: &mut ImagePlaneModel, strength: f64) {
    if strength == 0.0 {
        return;
    }
    let maximum_radius = model
        .k_vectors
        .iter()
        .map(|vector| vector.kx.hypot(vector.ky))
        .fold(0.0, f64::max);
    if maximum_radius <= f64::EPSILON {
        return;
    }
    let transmissions: Vec<_> = model
        .k_vectors
        .iter()
        .map(|vector| {
            let radius = vector.kx.hypot(vector.ky) / maximum_radius;
            (-strength * radius * radius).exp().max(f64::MIN_POSITIVE)
        })
        .collect();
    if let Some(matrix) = &mut model.multiplexing_matrix {
        for row in matrix {
            for (source, weight) in row {
                *weight = (*weight * transmissions[*source]).max(f64::MIN_POSITIVE);
            }
        }
    } else {
        let frame_count = model.frame_count();
        let gains = model
            .frame_gains
            .get_or_insert_with(|| vec![1.0; frame_count]);
        for (gain, &transmission) in gains.iter_mut().zip(&transmissions) {
            *gain = (*gain * transmission).max(f64::MIN_POSITIVE);
        }
    }
}
