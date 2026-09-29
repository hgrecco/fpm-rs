//! Bright-field circle initialization followed by reuse of the physical model.

use fpm_rs::{
    Result,
    experiment::{ArrayPose, Illumination, Optics, PlanarLedArray},
    illumination_calibration::{CalibrationParameterSpec, PlanarArrayCalibrationParameters},
    illumination_initialization::{BrightfieldCircleInitializer, BrightfieldCircleOptions},
    model::{ImagePlaneModel, ReconstructionShape},
    simulation::{Simulator, SyntheticObject},
};

fn main() -> Result<()> {
    let optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let nominal = illumination([0.0, 0.0, -80e-3])?;
    let truth = illumination([0.4e-3, -0.3e-3, -80e-3])?;
    let image_shape = (48, 56);
    let reconstruction_shape = (96, 112);
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        &truth,
        image_shape,
        ReconstructionShape::Exact(reconstruction_shape),
    )?;
    let nominal_model = ImagePlaneModel::from_experiment(
        &optics,
        &nominal,
        image_shape,
        ReconstructionShape::Exact(reconstruction_shape),
    )?;
    let simulation = Simulator::ideal(true_model)
        .object(SyntheticObject::mixed_test_pattern(reconstruction_shape)?)
        .reconstruction_model(nominal_model.clone())
        .simulate()?;

    let translation =
        CalibrationParameterSpec::new(-1e-3, 1e-3, 0.2e-3).finite_difference_step(1e-6);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(translation.clone()), Some(translation), None])
        .build()?;
    let result = BrightfieldCircleInitializer::new(parameters)
        .options(BrightfieldCircleOptions {
            center_search_radius_na: 0.012,
            pupil_radius_search_na: 0.012,
            gaussian_sigma_pixels: 1.5,
            minimum_edge_contrast: 1e-4,
            ..Default::default()
        })
        .initialize(&simulation.measurements, &optics, &nominal, &nominal_model)?;

    println!(
        "accepted {}/{} circles; translation = ({:.3e}, {:.3e}, {:.3e}) m",
        result.diagnostics.accepted_observations,
        result.diagnostics.candidate_frames,
        result.initialized_parameters.translation_m[0],
        result.initialized_parameters.translation_m[1],
        result.initialized_parameters.translation_m[2],
    );
    result.initialized_model.validate()?;
    result.save_json("planar-array-initialization.json")?;
    Ok(())
}

fn illumination(translation_m: [f64; 3]) -> Result<Illumination> {
    Illumination::from_geometry(PlanarLedArray::new(
        (5, 5),
        (4e-3, 4e-3),
        (2.0, 2.0),
        ArrayPose::from_translation(translation_m),
    ))
}
