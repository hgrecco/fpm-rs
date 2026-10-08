mod common;

use fpm_rs::{
    Error,
    algorithms::{Fpie, JointReconstruction, ReconstructionAlgorithm},
    backend::CpuBackend,
    experiment::{
        AcquisitionPlan, ArrayPose, Illumination, IlluminationFrame, Optics, PlanarLedArray,
        SourceCalibration, SourceContribution,
    },
    illumination_calibration::{
        CalibrationParameterSpec, IlluminationCalibration, PlanarArrayCalibrationParameters,
    },
    illumination_initialization::{
        BrightfieldCircleInitializer, BrightfieldCircleOptions, PlanarArrayInitializationAction,
        PlanarArrayInitializationCallback, PlanarArrayInitializationProgress,
    },
    measurements::{LazyMeasurementStack, MeasurementStack},
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};
use std::{
    fs::File,
    io::BufWriter,
    sync::{Arc, atomic::Ordering},
};

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

#[test]
fn circle_warm_start_improves_the_same_budget_joint_calibration() {
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
        BrightfieldCircleInitializer::new(parameters.clone()).options(BrightfieldCircleOptions {
            center_search_radius_na: 0.012,
            pupil_radius_search_na: 0.012,
            gaussian_sigma_pixels: 1.5,
            minimum_edge_contrast: 1e-4,
            maximum_fit_steps: 150,
            fit_initial_step_size: 0.25,
            ..BrightfieldCircleOptions::default()
        });
    let initialized = initializer
        .initialize(&simulation.measurements, &optics, &nominal, &nominal_model)
        .unwrap();

    let cold_problem =
        ReconstructionProblem::new(simulation.measurements.clone(), nominal_model).unwrap();
    let warm_problem = ReconstructionProblem::new(
        simulation.measurements,
        initialized.initialized_model.clone(),
    )
    .unwrap();
    let cold = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics.clone(),
        nominal,
        IlluminationCalibration::new(parameters.clone()),
        1,
    )
    .run(&cold_problem)
    .unwrap();
    let warm = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics.clone(),
        initialized.initialized_illumination.clone(),
        IlluminationCalibration::new(parameters),
        1,
    )
    .run(&warm_problem)
    .unwrap();
    let cold_error = source_error(&optics, &cold.calibrated_illumination, &truth);
    let warm_error = source_error(&optics, &warm.calibrated_illumination, &truth);
    assert!(
        warm_error < cold_error,
        "warm source error did not improve under the same joint-calibration budget: {cold_error} -> {warm_error}"
    );
}

#[test]
fn circle_initializer_rejects_a_textureless_specimen() {
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
        .object(SyntheticObject::constant(reconstruction_shape, 1.0, 0.0).unwrap())
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
            minimum_edge_contrast: 0.2,
            ..BrightfieldCircleOptions::default()
        });
    assert!(matches!(
        initializer.initialize(&simulation.measurements, &optics, &nominal, &nominal_model),
        Err(Error::InvalidMeasurements(_))
    ));
}

fn test_initializer() -> BrightfieldCircleInitializer {
    let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 0.2e-3).finite_difference_step(1e-6);
    BrightfieldCircleInitializer::new(
        PlanarArrayCalibrationParameters::builder()
            .translation_specs([Some(spec.clone()), Some(spec), None])
            .build()
            .unwrap(),
    )
    .options(BrightfieldCircleOptions {
        center_search_radius_na: 0.012,
        pupil_radius_search_na: 0.012,
        gaussian_sigma_pixels: 1.5,
        minimum_edge_contrast: 1e-4,
        maximum_fit_steps: 150,
        fit_initial_step_size: 0.25,
        ..Default::default()
    })
}

#[test]
fn adverse_specimens_preserve_streaming_backend_and_acquisition_parity() {
    let optics = optics();
    let image_shape = (48, 56);
    let high_shape = (96, 112);
    let field = SyntheticObject::mixed_test_pattern(high_shape).unwrap();
    for specimen in ["amplitude", "phase", "mixed"] {
        for noise in [0.0, 0.01] {
            // A fixed deterministic additive pattern exercises a perturbed
            // detector without asserting a published noise model.
            let object = SyntheticObject::new(field.field().mapv(|value| match specimen {
                "amplitude" => fpm_rs::Complex64::new(value.norm(), 0.0),
                "phase" => fpm_rs::Complex64::from_polar(1.0, value.arg()),
                _ => value,
            }))
            .unwrap();
            let nominal = illumination([0.0, 0.0, -80e-3]);
            let truth = illumination([0.4e-3, -0.3e-3, -80e-3]);
            let model = ImagePlaneModel::from_experiment(
                &optics,
                &nominal,
                image_shape,
                ReconstructionShape::Exact(high_shape),
            )
            .unwrap();
            let true_model = ImagePlaneModel::from_experiment(
                &optics,
                &truth,
                image_shape,
                ReconstructionShape::Exact(high_shape),
            )
            .unwrap();
            let mut measurements = Simulator::ideal(true_model)
                .object(object)
                .simulate()
                .unwrap()
                .measurements;
            let mean =
                measurements.as_slice().iter().sum::<f64>() / measurements.as_slice().len() as f64;
            for frame in 0..measurements.frame_count() {
                for (pixel, value) in measurements
                    .frame_mut(frame)
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *value =
                        (*value + noise * mean * ((pixel * 17 + frame * 31) as f64).sin()).max(0.0);
                }
            }
            // The public TIFF loader accepts integer grayscale pages. Apply
            // the same lossless integer samples to both providers.
            let maximum = measurements.as_slice().iter().copied().fold(0.0, f64::max);
            let scale = 60_000.0 / maximum;
            for frame in 0..measurements.frame_count() {
                for value in measurements.frame_mut(frame).unwrap() {
                    *value = (*value * scale).round();
                }
            }
            let initializer = test_initializer();
            let resident = initializer
                .initialize(&measurements, &optics, &nominal, &model)
                .unwrap();
            assert!(
                resident.diagnostics.accepted_observations >= 2,
                "{specimen}/{noise}"
            );

            // Provider/backend parity uses exactly the same detector samples.
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("frames.tiff");
            let mut encoder =
                tiff::encoder::TiffEncoder::new(BufWriter::new(File::create(&path).unwrap()))
                    .unwrap();
            for frame in 0..measurements.frame_count() {
                encoder
                    .write_image::<tiff::encoder::colortype::Gray16>(
                        image_shape.1 as u32,
                        image_shape.0 as u32,
                        &measurements
                            .frame(frame)
                            .unwrap()
                            .iter()
                            .map(|&value| value as u16)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            drop(encoder);
            let lazy = LazyMeasurementStack::from_tiff_stack(&path, Vec::new()).unwrap();
            let (backend, calls) = common::CountingBackend::new(image_shape, high_shape).unwrap();
            let streamed = initializer
                .initialize_with_backend(&lazy, &optics, &nominal, &model, backend)
                .unwrap();
            assert_eq!(
                calls.load(Ordering::Relaxed),
                2 * resident.diagnostics.candidate_frames
            );
            assert_eq!(resident.observations, streamed.observations);
            assert_eq!(
                resident.initialized_parameters,
                streamed.initialized_parameters
            );

            let order: Vec<_> = (0..25).rev().collect();
            let permuted = nominal
                .clone()
                .with_acquisition(AcquisitionPlan::sequential(order.clone()).unwrap());
            let permuted_model = ImagePlaneModel::from_experiment(
                &optics,
                &permuted,
                image_shape,
                ReconstructionShape::Exact(high_shape),
            )
            .unwrap();
            let reordered = MeasurementStack::from_frames(
                &order
                    .iter()
                    .map(|&source| measurements.frame(source).unwrap().to_vec())
                    .collect::<Vec<_>>(),
                image_shape,
            )
            .unwrap();
            let permuted_result = initializer
                .initialize(&reordered, &optics, &permuted, &permuted_model)
                .unwrap();
            assert_eq!(
                resident.initialized_parameters.translation_m,
                permuted_result.initialized_parameters.translation_m
            );
            for (original, permuted) in resident
                .observations
                .iter()
                .zip(&permuted_result.observations)
            {
                assert_eq!(original.source_index, permuted.source_index);
                assert_eq!(original.detected_na, permuted.detected_na);
                assert_eq!(permuted.frame_index, 24 - original.frame_index);
            }
        }
    }
}

#[test]
fn initialization_preserves_known_calibration_and_refreshes_geometry_atomically() {
    let optics = optics();
    let image_shape = (48, 56);
    let high_shape = (96, 112);
    let powers: Vec<_> = (0..25).map(|i| 0.6 + 0.03 * i as f64).collect();
    let acquisition = AcquisitionPlan::from_sparse(
        (0..25)
            .map(|source| {
                IlluminationFrame::new(
                    vec![SourceContribution::new(source, 0.8 + 0.01 * source as f64)],
                    0.7 + 0.02 * source as f64,
                )
            })
            .collect(),
    )
    .unwrap();
    let nominal = illumination([0.0, 0.0, -80e-3])
        .with_calibration(SourceCalibration::new(Some(powers)))
        .with_acquisition(acquisition);
    let truth = nominal.clone().with_geometry(PlanarLedArray::new(
        (5, 5),
        (4e-3, 4e-3),
        (2.0, 2.0),
        ArrayPose::from_translation([0.4e-3, -0.3e-3, -80e-3]),
    ));
    let background = vec![0.03; image_shape.0 * image_shape.1];
    let nominal_model = ImagePlaneModel::from_experiment(
        &optics,
        &nominal,
        image_shape,
        ReconstructionShape::Exact(high_shape),
    )
    .unwrap()
    .with_background(Some(background.clone()))
    .unwrap();
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        &truth,
        image_shape,
        ReconstructionShape::Exact(high_shape),
    )
    .unwrap()
    .with_background(Some(background))
    .unwrap();
    let measurements = Simulator::ideal(true_model)
        .object(SyntheticObject::mixed_test_pattern(high_shape).unwrap())
        .simulate()
        .unwrap()
        .measurements;
    let serialized_before = serde_json::to_string(&nominal_model).unwrap();
    let result = test_initializer()
        .initialize(&measurements, &optics, &nominal, &nominal_model)
        .unwrap();
    assert_eq!(
        serde_json::to_string(&nominal_model).unwrap(),
        serialized_before
    );
    assert_eq!(
        result.initialized_illumination.calibration(),
        nominal.calibration()
    );
    assert_eq!(
        result.initialized_illumination.acquisition(),
        nominal.acquisition()
    );
    assert_eq!(result.initialized_model.pupil(), nominal_model.pupil());
    assert_eq!(
        result.initialized_model.background(),
        nominal_model.background()
    );
    assert_eq!(
        result.initialized_model.frame_gains(),
        nominal_model.frame_gains()
    );
    assert_eq!(
        result.initialized_model.multiplexing_matrix(),
        nominal_model.multiplexing_matrix()
    );
    let recompiled = ImagePlaneModel::from_experiment(
        &optics,
        &result.initialized_illumination,
        image_shape,
        ReconstructionShape::Exact(high_shape),
    )
    .unwrap();
    assert_eq!(result.initialized_model.k_vectors(), recompiled.k_vectors());
    assert_eq!(
        result.initialized_model.subpixel_offsets(),
        recompiled.subpixel_offsets()
    );

    let mut unsafe_subset = test_initializer();
    unsafe_subset.options.frame_indices = Some(vec![0]);
    assert!(
        matches!(unsafe_subset.initialize(&measurements, &optics, &nominal, &nominal_model),
        Err(Error::InvalidMeasurements(message)) if message.contains("bright-field boundary"))
    );
    assert_eq!(
        serde_json::to_string(&nominal_model).unwrap(),
        serialized_before
    );
}
