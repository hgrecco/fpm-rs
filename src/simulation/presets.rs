//! Small deterministic simulation presets for tests and benchmarks.
//!
//! These functions configure the existing [`super::Simulator`] and return its
//! normal [`super::SimulationResult`]. They intentionally do not introduce a
//! second synthetic-dataset abstraction.

use crate::{
    Result,
    experiment::{LEDArray, Optics, PupilAberration},
    model::ImagePlaneModel,
};

use super::{CameraModel, SimulationResult, Simulator, SyntheticObject};

pub const NOISELESS_MIXED_PRESET: &str = "noiseless_mixed_v1";
pub const ABERRATED_PUPIL_PRESET: &str = "aberrated_pupil_v1";
pub const POISSON_GAUSSIAN_PRESET: &str = "poisson_gaussian_v1";

/// A 3×3 LED, 32×32 measurement, 64×64 mixed-object ideal acquisition.
pub fn noiseless_mixed_fpm(seed: u64) -> Result<SimulationResult> {
    let model = base_model(base_optics())?;
    Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((64, 64))?)
        .seed(seed)
        .simulate()
}

/// A known defocus/astigmatism mismatch suitable for EPRY pupil recovery.
pub fn aberrated_pupil_fpm(seed: u64) -> Result<SimulationResult> {
    let assumed_optics = base_optics();
    let true_optics = Optics {
        defocus_distance: Some(-18e-6),
        pupil_aberration: Some(PupilAberration {
            astigmatism: 0.25,
            spherical: 0.12,
            ..PupilAberration::default()
        }),
        ..assumed_optics.clone()
    };
    let true_model = base_model(true_optics)?;
    let reconstruction_model = base_model(assumed_optics)?;
    Simulator::new(true_model)
        .object(SyntheticObject::mixed_test_pattern((64, 64))?)
        .reconstruction_model(reconstruction_model)
        .seed(seed)
        .simulate()
}

/// A count-domain acquisition with shot noise, Gaussian read noise, offset,
/// quantization, and a finite 16-bit detector range.
pub fn poisson_gaussian_fpm(seed: u64) -> Result<SimulationResult> {
    let model = base_model(base_optics())?;
    let camera = CameraModel::new()
        .photons_per_pixel(400.0)
        .gain(1.5)
        .offset_counts(100.0)
        .read_noise_electrons(2.0)
        .shot_noise(true)
        .bit_depth(16)
        .quantize(true);
    Simulator::new(model)
        .object(SyntheticObject::mixed_test_pattern((64, 64))?)
        .camera(camera)
        .seed(seed)
        .simulate()
}

fn base_optics() -> Optics {
    Optics {
        wavelength: 532e-9,
        objective_na: 0.10,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        medium_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    }
}

fn base_model(optics: Optics) -> Result<ImagePlaneModel> {
    let illumination = LEDArray::new()
        .grid_shape((3, 3))
        .pitch(4e-3)
        .distance(90e-3)
        .center((1.0, 1.0));
    ImagePlaneModel::from_experiment(&optics, &illumination, (32, 32), (64, 64))
}
