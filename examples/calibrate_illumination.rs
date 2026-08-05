//! Synthetic physical planar-array calibration followed by model reuse.

use fpm_rs::{
    Result,
    algorithms::{Fpie, JointReconstruction},
    experiment::{ArrayPose, Illumination, Optics, PlanarLedArray, SourceCalibration},
    illumination_calibration::{
        BoundedFiniteDifferenceOptimizer, CalibrationParameterSpec, IlluminationCalibration,
        PlanarArrayCalibrationParameters,
    },
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::ReconstructionProblem,
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
    let nominal = Illumination::from_geometry(PlanarLedArray::new(
        (3, 3),
        (4e-3, 4e-3),
        (1.0, 1.0),
        ArrayPose::from_translation([0.0, 0.0, -80e-3]),
    ))?;
    let mut true_offsets = vec![[0.0; 3]; 9];
    true_offsets[0] = [0.1e-3, -0.05e-3, 0.02e-3];
    true_offsets[8] = [-0.1e-3, 0.05e-3, -0.02e-3];
    let truth = Illumination::from_geometry(
        PlanarLedArray::new(
            (3, 3),
            (4.05e-3, 3.95e-3),
            (1.0, 1.0),
            ArrayPose::from_translation_and_extrinsic_xyz_radians(
                [0.3e-3, -0.2e-3, -79e-3],
                [0.004, -0.003, 0.01],
            ),
        )
        .with_position_offsets_m(true_offsets),
    )?
    .with_calibration(SourceCalibration::new(Some(vec![
        0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0,
    ])));
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        &truth,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )?;
    let assumed_model = ImagePlaneModel::from_experiment(
        &optics,
        &nominal,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )?;
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::resolution_target((36, 36))?)
        .reconstruction_model(assumed_model.clone())
        .seed(42)
        .simulate()?;
    let problem = ReconstructionProblem::new(simulation.measurements, assumed_model)?;

    let lateral_translation =
        CalibrationParameterSpec::new(-1e-3, 1e-3, 2.5e-4).finite_difference_step(2e-5);
    let axial_translation =
        CalibrationParameterSpec::new(-90e-3, -70e-3, 1e-3).finite_difference_step(1e-5);
    let rotation = CalibrationParameterSpec::new(-0.05, 0.05, 0.01).finite_difference_step(1e-4);
    let pitch = CalibrationParameterSpec::new(3.5e-3, 4.5e-3, 0.1e-3).finite_difference_step(1e-6);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([
            Some(lateral_translation.clone()),
            Some(lateral_translation),
            Some(axial_translation),
        ])
        .rotation_specs(std::array::from_fn(|_| Some(rotation.clone())))
        .pitch_specs(std::array::from_fn(|_| Some(pitch.clone())))
        .position_offsets([0, 8])
        .relative_source_power(true)
        .build()?;
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 2,
            initial_step_size: 0.5,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics.clone(),
        nominal,
        calibration,
        2,
    )
    .object_iterations_per_outer(2)
    .run(&problem)?;

    println!(
        "tx={:.3e} m, tz={:.3e} m, rz={:.3e} rad, pitch=({:.3e}, {:.3e}) m, physical loss={:.4e}",
        result.final_parameters.translation_m[0],
        result.final_parameters.translation_m[2],
        result.final_parameters.rotation_rad[2],
        result.final_parameters.pitch_m[0],
        result.final_parameters.pitch_m[1],
        result
            .loss_history
            .last()
            .map_or(f64::NAN, |row| row.total_loss),
    );
    println!(
        "model trial updates: geometry={}, intensity-only={}, rejected optimizer steps={}",
        result.diagnostics.geometry_recompilations,
        result.diagnostics.multiplicative_updates,
        result.diagnostics.rejected_steps,
    );

    let continuation_model = ImagePlaneModel::from_experiment(
        &optics,
        &result.calibrated_illumination,
        problem.model.image_shape(),
        ReconstructionShape::Exact(problem.model.reconstruction_shape()),
    )?;
    continuation_model.validate()?;
    Ok(())
}
