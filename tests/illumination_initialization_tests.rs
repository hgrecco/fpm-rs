use fpm_rs::{
    Error,
    algorithms::{Fpie, ReconstructionAlgorithm},
    backend::CpuBackend,
    experiment::{ArrayPose, Illumination, Optics, PlanarLedArray},
    illumination_calibration::{CalibrationParameterSpec, PlanarArrayCalibrationParameters},
    illumination_initialization::{
        BrightfieldCircleInitializer, BrightfieldCircleOptions, PlanarArrayInitializationAction,
        PlanarArrayInitializationCallback, PlanarArrayInitializationProgress,
    },
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};
use std::sync::Arc;

struct CancelImmediately;

impl PlanarArrayInitializationCallback for CancelImmediately {
    fn on_progress(
        &mut self,
        _progress: &PlanarArrayInitializationProgress,
    ) -> fpm_rs::Result<PlanarArrayInitializationAction> {
        Ok(PlanarArrayInitializationAction::Cancel)
    }
}

fn optics() -> Optics {
    Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    }
}

fn illumination(translation: [f64; 3]) -> Illumination {
    Illumination::from_geometry(PlanarLedArray::new(
        (5, 5),
        (4e-3, 4e-3),
        (2.0, 2.0),
        ArrayPose::from_translation(translation),
    ))
    .unwrap()
}

fn source_error(optics: &Optics, actual: &Illumination, expected: &Illumination) -> f64 {
    let actual = actual.resolve(optics).unwrap();
    let expected = expected.resolve(optics).unwrap();
    actual
        .k_vectors()
        .iter()
        .zip(expected.k_vectors())
        .map(|(actual, expected)| (actual.kx - expected.kx).hypot(actual.ky - expected.ky))
        .sum::<f64>()
        / actual.source_count() as f64
}

#[test]
fn simulated_brightfield_initialization_reduces_planar_translation_error() {
    let optics = optics();
    let nominal = illumination([0.0, 0.0, -80e-3]);
    let truth = illumination([0.4e-3, -0.3e-3, -80e-3]);
    let image_shape = (48, 56);
    let reconstruction_shape = (96, 112);
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        &truth,
        image_shape,
        ReconstructionShape::Exact(reconstruction_shape),
    )
    .unwrap();
    let nominal_model = ImagePlaneModel::from_experiment(
        &optics,
        &nominal,
        image_shape,
        ReconstructionShape::Exact(reconstruction_shape),
    )
    .unwrap();
    let simulation = Simulator::ideal(true_model)
        .object(SyntheticObject::mixed_test_pattern(reconstruction_shape).unwrap())
        .reconstruction_model(nominal_model.clone())
        .simulate()
        .unwrap();
    let translation =
        CalibrationParameterSpec::new(-1.0e-3, 1.0e-3, 0.2e-3).finite_difference_step(1e-6);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(translation.clone()), Some(translation), None])
        .build()
        .unwrap();
    let initializer =
        BrightfieldCircleInitializer::new(parameters).options(BrightfieldCircleOptions {
            center_search_radius_na: 0.012,
            pupil_radius_search_na: 0.012,
            gaussian_sigma_pixels: 1.5,
            minimum_edge_contrast: 1e-4,
            maximum_fit_steps: 150,
            fit_initial_step_size: 0.25,
            ..BrightfieldCircleOptions::default()
        });
    let result = initializer
        .initialize(&simulation.measurements, &optics, &nominal, &nominal_model)
        .unwrap();

    let before = source_error(&optics, &nominal, &truth);
    let after = source_error(&optics, &result.initialized_illumination, &truth);
    assert!(
        after < before,
        "source error did not improve: {before} -> {after}"
    );
    assert!(result.diagnostics.final_residual_rms_na < result.diagnostics.initial_residual_rms_na);
    let initial_parameter_error = 0.4e-3_f64.hypot(-0.3e-3);
    let final_parameter_error = (result.initialized_parameters.translation_m[0] - 0.4e-3)
        .hypot(result.initialized_parameters.translation_m[1] + 0.3e-3);
    assert!(final_parameter_error < initial_parameter_error);
    assert_eq!(result.initialized_model.pupil(), nominal_model.pupil());
    assert_eq!(result.runtime.measurement_passes, 2);
    assert!(result.observations.iter().any(|observation| {
        observation.source_index == 12
            && observation
                .rejection_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("on-axis"))
    }));

    let backend = Arc::new(CpuBackend::new(image_shape, reconstruction_shape).unwrap());
    let mut callback = CancelImmediately;
    assert!(matches!(
        initializer.initialize_with_callback(
            &simulation.measurements,
            &optics,
            &nominal,
            &nominal_model,
            backend,
            &mut callback,
        ),
        Err(Error::Numerical(message)) if message.contains("cancelled")
    ));

    let warm_problem = ReconstructionProblem::new(
        simulation.measurements.clone(),
        result.initialized_model.clone(),
    )
    .unwrap();
    let warm_result = Fpie::default().iterations(1).run(&warm_problem).unwrap();
    assert_eq!(warm_result.runtime.completed_iterations, 1);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialization.json");
    result.save_json(&path).unwrap();
    let restored =
        fpm_rs::illumination_initialization::PlanarArrayInitializationResult::load_json(path)
            .unwrap();
    assert_eq!(restored.parameter_names, result.parameter_names);
    assert_eq!(restored.observations.len(), result.observations.len());
    for (restored, original) in restored
        .initialized_parameters
        .translation_m
        .iter()
        .zip(result.initialized_parameters.translation_m)
    {
        assert!((restored - original).abs() < 1e-15);
    }

    let bundle = result
        .write_bundle(directory.path().join("initialization-bundle"))
        .unwrap();
    let verification = bundle.verify().unwrap();
    assert_eq!(verification.artifact_count, 3);
    assert!(verification.total_bytes > 0);
    assert_eq!(
        bundle.result.initialized_parameters,
        result.initialized_parameters
    );
    let mut bytes = std::fs::read(&bundle.observations_artifact.path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&bundle.observations_artifact.path, bytes).unwrap();
    assert!(matches!(
        bundle.verify(),
        Err(Error::ArtifactHashMismatch { .. })
    ));
}
