mod common;

use approx::assert_abs_diff_eq;
use fpm_rs::{
    Complex64,
    algorithms::{
        Admm, AlternatingProjection, Epry, Fpie, GradientDescent, ReconstructionAlgorithm,
        objective::{LossType, loss},
    },
    callbacks::CheckpointEvery,
    evaluation::{evaluate_reconstruction, evaluate_reconstruction_with_problem},
    experiment::{ArrayPose, Illumination, Optics, PlanarLedArray, PupilAberration},
    measurements::{LazyMeasurementStack, MeasurementRead},
    model::{ForwardModel, FourierOffset, ImagePlaneModel, Pupil, ReconstructionShape},
    reconstruction::{
        Batch, ReconstructionCheckpoint, ReconstructionProblem, ReconstructionState,
        ReconstructionTrace, RunOptions, Runner,
    },
    simulation::{CameraModel, Simulator, SyntheticObject},
};
use image::{ImageBuffer, Luma};
use ndarray::{Array2, ShapeBuilder};
use std::sync::atomic::Ordering;

fn admm_metric_values(
    result: &fpm_rs::reconstruction::ReconstructionResult,
    metric: &str,
) -> Vec<f64> {
    result
        .trace
        .algorithm_metrics
        .iter()
        .filter(|record| record.namespace == "admm" && record.metric == metric)
        .map(|record| record.value)
        .collect()
}

#[test]
fn ap_reconstructs_and_reports_history() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::resolution_target((16, 16)).unwrap();
    let simulation = Simulator::ideal(model).object(object).simulate().unwrap();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = AlternatingProjection::default()
        .iterations(8)
        .run(&problem)
        .unwrap();
    assert_eq!(result.object.dim(), (16, 16));
    assert_eq!(result.amplitude.dim(), (16, 16));
    assert_eq!(result.phase.dim(), (16, 16));
    assert_eq!(result.recovered_pupil.shape(), (8, 8));
    assert_eq!(result.trace.iterations.len(), 8);
    let first = result.trace.iterations.first().unwrap().objective;
    let last = result.trace.iterations.last().unwrap().objective;
    assert!(
        last < first,
        "expected loss decrease, got {first} -> {last}"
    );
    let metrics =
        evaluate_reconstruction_with_problem(&result, &problem, truth.view(), None, None).unwrap();
    let residuals: Vec<_> = metrics
        .intensity
        .unwrap()
        .per_frame
        .into_iter()
        .map(|frame| frame.normalized_l2)
        .collect();
    assert_eq!(residuals.len(), problem.model.frame_count());
    assert!(residuals.iter().all(|value| value.is_finite()));
}

#[test]
fn reconstruction_initialization_rejects_nonstandard_owned_objects() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let shape = problem.model.reconstruction_shape();
    let object = Array2::from_elem(shape.f(), Complex64::new(1.0, 0.0));
    assert!(matches!(
        ReconstructionState::from_object(&problem, object),
        Err(fpm_rs::Error::NonStandardLayout { .. })
    ));
}

#[test]
fn reconstruction_runs_directly_from_lazy_measurements() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for frame in 0..simulation.measurements.frame_count() {
        let path = directory.path().join(format!("frame_{frame}.png"));
        let values: Vec<u16> = simulation
            .measurements
            .frame(frame)
            .unwrap()
            .iter()
            .map(|value| (value * 1_000.0).round().clamp(0.0, u16::MAX as f64) as u16)
            .collect();
        let image: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_vec(
            simulation.measurements.image_shape().1 as u32,
            simulation.measurements.image_shape().0 as u32,
            values,
        )
        .unwrap();
        image.save(&path).unwrap();
        paths.push(path);
    }
    let lazy = LazyMeasurementStack::from_image_files(&paths, Vec::new()).unwrap();
    let problem = ReconstructionProblem::new(lazy, simulation.reconstruction_model).unwrap();
    assert_eq!(problem.measurements.cached_frame_count(), 0);
    let result = AlternatingProjection::default()
        .iterations(1)
        .run(&problem)
        .unwrap();
    assert_eq!(result.runtime.completed_iterations, 1);
    assert_eq!(problem.measurements.cached_frame_count(), 1);
}

#[cfg(feature = "parquet")]
#[test]
fn reconstruction_result_bundle_round_trips_and_validates() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = AlternatingProjection::default()
        .iterations(2)
        .run(&problem)
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("result");
    let bundle = result
        .write_bundle(
            &path,
            fpm_rs::reconstruction::BundleExportOptions::default(),
        )
        .unwrap();
    let loaded = bundle.result().unwrap();
    for (&loaded, &original) in loaded.object.iter().zip(result.object.iter()) {
        assert_abs_diff_eq!(loaded.re, original.re, epsilon = 1e-14);
        assert_abs_diff_eq!(loaded.im, original.im, epsilon = 1e-14);
    }
    for (&loaded, &original) in loaded
        .object_spectrum
        .iter()
        .zip(result.object_spectrum.iter())
    {
        assert_abs_diff_eq!(loaded.re, original.re, epsilon = 1e-14);
        assert_abs_diff_eq!(loaded.im, original.im, epsilon = 1e-14);
    }
    for (&loaded, &original) in loaded.amplitude.iter().zip(result.amplitude.iter()) {
        assert_abs_diff_eq!(loaded, original, epsilon = 1e-14);
    }
    for (&loaded, &original) in loaded.phase.iter().zip(result.phase.iter()) {
        assert_abs_diff_eq!(loaded, original, epsilon = 1e-14);
    }
    for (&loaded, &original) in loaded
        .recovered_pupil
        .values()
        .iter()
        .zip(result.recovered_pupil.values().iter())
    {
        assert_abs_diff_eq!(loaded.re, original.re, epsilon = 1e-14);
        assert_abs_diff_eq!(loaded.im, original.im, epsilon = 1e-14);
    }
    assert_eq!(loaded.trace.iterations.len(), 2);
    assert_eq!(loaded.runtime.completed_iterations, 2);

    let invalid_path = directory.path().join("invalid_result");
    let mut invalid = result.clone();
    invalid.amplitude = Array2::zeros((1, 1));
    assert!(
        invalid
            .write_bundle(
                &invalid_path,
                fpm_rs::reconstruction::BundleExportOptions::default(),
            )
            .is_err()
    );
    assert!(!invalid_path.exists());

    let mut serialized: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path.join("manifest.json")).unwrap()).unwrap();
    serialized["bundle_format_version"] = serde_json::json!(999);
    std::fs::write(
        path.join("manifest.json"),
        serde_json::to_vec(&serialized).unwrap(),
    )
    .unwrap();
    assert!(fpm_rs::read_bundle(path).is_err());
}

#[test]
fn runner_uses_injected_backend_for_initialization_updates_and_result() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let (backend, calls) = common::CountingBackend::new(
        problem.model.image_shape(),
        problem.model.reconstruction_shape(),
    )
    .unwrap();
    let result = Runner::new(
        AlternatingProjection::default(),
        RunOptions {
            max_iterations: 2,
            ..RunOptions::default()
        },
    )
    .with_backend(backend)
    .run(&problem)
    .unwrap();
    assert_eq!(result.runtime.completed_iterations, 2);
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn fpie_loss_decreases_on_noiseless_data() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Fpie::default().iterations(8).run(&problem).unwrap();
    assert!(
        result.trace.iterations.last().unwrap().objective
            < result.trace.iterations.first().unwrap().objective
    );
}

#[test]
fn multiplexed_admm_reports_finite_consensus_residuals() {
    let model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.65), (1, 0.35)],
            vec![(2, 0.25), (3, 0.30), (4, 0.45)],
        ])
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Admm::default()
        .iterations(16)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    let primal = admm_metric_values(&result, "primal_residual_rms");
    let dual = admm_metric_values(&result, "dual_residual_rms");
    assert_eq!(primal.len(), result.trace.iterations.len());
    assert_eq!(dual.len(), result.trace.iterations.len());
    assert!(primal.iter().all(|value| value.is_finite()));
    assert!(dual.iter().all(|value| value.is_finite()));
    assert!(primal.last().unwrap() < primal.first().unwrap());
    assert!(dual.last().unwrap() < dual.first().unwrap());
}

#[test]
fn admm_loss_decreases_on_noiseless_data() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Admm::default()
        .iterations(12)
        .object_step(0.6)
        .penalty(1.0)
        .run(&problem)
        .unwrap();
    assert!(
        result.trace.iterations.last().unwrap().objective
            < result.trace.iterations.first().unwrap().objective
    );
    let primal = admm_metric_values(&result, "primal_residual_rms");
    let dual = admm_metric_values(&result, "dual_residual_rms");
    assert!(
        primal.last().unwrap() < primal.first().unwrap(),
        "expected ADMM primal residual to decrease: {:?} -> {:?}",
        primal.first(),
        primal.last()
    );
    assert!(
        dual.last().unwrap() < dual.first().unwrap(),
        "expected ADMM dual residual to decrease: {:?} -> {:?}",
        dual.first(),
        dual.last()
    );
    assert!(
        result
            .scalar_diagnostics
            .keys()
            .all(|key| !key.starts_with("admm"))
    );
}

#[test]
fn admm_checkpoint_resume_preserves_optimizer_state() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let directory = tempfile::tempdir().unwrap();
    Admm::default()
        .iterations(2)
        .object_step(0.6)
        .run_with_callbacks(
            &problem,
            vec![Box::new(CheckpointEvery::new(2, directory.path()))],
        )
        .unwrap();
    let checkpoint =
        ReconstructionCheckpoint::load(directory.path().join("checkpoint_00002.json")).unwrap();
    assert!(checkpoint.algorithm_auxiliary().is_some());
    let resumed = Admm::default()
        .iterations(5)
        .object_step(0.6)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    let uninterrupted = Admm::default()
        .iterations(5)
        .object_step(0.6)
        .run(&problem)
        .unwrap();
    for (&resumed, &uninterrupted) in resumed
        .object_spectrum
        .iter()
        .zip(uninterrupted.object_spectrum.iter())
    {
        assert_abs_diff_eq!(resumed.re, uninterrupted.re, epsilon = 1e-12);
        assert_abs_diff_eq!(resumed.im, uninterrupted.im, epsilon = 1e-12);
    }
}

#[test]
fn multiplexed_admm_checkpoint_preserves_per_mode_state() {
    let model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.6), (1, 0.4)],
            vec![(2, 0.3), (3, 0.2), (4, 0.5)],
        ])
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let directory = tempfile::tempdir().unwrap();
    Admm::default()
        .iterations(2)
        .object_step(0.5)
        .run_with_callbacks(
            &problem,
            vec![Box::new(CheckpointEvery::new(2, directory.path()))],
        )
        .unwrap();
    let checkpoint = ReconstructionCheckpoint::load_for_problem(
        directory.path().join("checkpoint_00002.json"),
        &problem,
    )
    .unwrap();
    let fpm_rs::reconstruction::AlgorithmAuxiliaryState::Admm(auxiliary) =
        checkpoint.algorithm_auxiliary().unwrap();
    assert_eq!(
        auxiliary.auxiliary_fields.len(),
        5 * problem.measurements.frame_len()
    );
    let resumed = Admm::default()
        .iterations(4)
        .object_step(0.5)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    let uninterrupted = Admm::default()
        .iterations(4)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    for (&resumed, &uninterrupted) in resumed
        .object_spectrum
        .iter()
        .zip(uninterrupted.object_spectrum.iter())
    {
        assert_abs_diff_eq!(resumed.re, uninterrupted.re, epsilon = 1e-12);
        assert_abs_diff_eq!(resumed.im, uninterrupted.im, epsilon = 1e-12);
    }
}

#[test]
fn admm_honors_masks_and_known_sensor_calibration() {
    let model = common::direct_model()
        .unwrap()
        .with_frame_gains(Some(vec![3.0; 5]))
        .unwrap()
        .with_background(Some(vec![7.0; 64]))
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut corrupted = simulation.measurements.clone();
    for frame in 0..corrupted.frame_count() {
        corrupted.frame_mut(frame).unwrap()[0] = 1e12;
    }
    let mut mask = Array2::from_elem(corrupted.image_shape(), 1_u8);
    mask[(0, 0)] = 0;
    let clean = simulation.measurements.with_masks(mask.clone()).unwrap();
    let corrupted = corrupted.with_masks(mask).unwrap();
    let clean_problem =
        ReconstructionProblem::new(clean, simulation.reconstruction_model.clone()).unwrap();
    let corrupted_problem =
        ReconstructionProblem::new(corrupted, simulation.reconstruction_model).unwrap();
    let clean_result = Admm::default().iterations(4).run(&clean_problem).unwrap();
    let corrupted_result = Admm::default()
        .iterations(4)
        .run(&corrupted_problem)
        .unwrap();
    for (&clean, &corrupted) in clean_result
        .object_spectrum
        .iter()
        .zip(corrupted_result.object_spectrum.iter())
    {
        assert_abs_diff_eq!(clean.re, corrupted.re, epsilon = 1e-12);
        assert_abs_diff_eq!(clean.im, corrupted.im, epsilon = 1e-12);
    }
}

#[test]
fn amplitude_gradient_loss_decreases_on_single_source_data() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = GradientDescent::default()
        .iterations(8)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    assert!(
        result.trace.iterations.last().unwrap().objective
            < result.trace.iterations.first().unwrap().objective
    );
}

#[test]
fn projection_and_gradient_updates_support_subpixel_crops() {
    let model = common::direct_model()
        .unwrap()
        .with_subpixel_offsets(vec![
            FourierOffset::new(-0.2, 0.3),
            FourierOffset::new(0.25, -0.35),
            FourierOffset::new(0.1, 0.2),
            FourierOffset::new(-0.3, -0.15),
            FourierOffset::new(0.2, 0.25),
        ])
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();

    let projection = AlternatingProjection::default()
        .iterations(8)
        .object_step(0.4)
        .run(&problem)
        .unwrap();
    let gradient = GradientDescent::default()
        .iterations(8)
        .object_step(0.3)
        .batch_size(2)
        .run(&problem)
        .unwrap();
    for result in [projection, gradient] {
        assert!(
            result.trace.iterations.last().unwrap().objective
                < result.trace.iterations.first().unwrap().objective
        );
    }
}

#[test]
fn gradient_recovers_known_source_offsets_with_fixed_object() {
    let reconstruction_model = common::direct_model().unwrap();
    let true_offsets = vec![
        FourierOffset::new(0.20, -0.15),
        FourierOffset::new(-0.18, 0.22),
        FourierOffset::new(0.16, 0.12),
        FourierOffset::new(-0.20, -0.14),
        FourierOffset::new(0.14, -0.20),
    ];
    let true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(true_offsets.clone())
        .unwrap();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let mut state = ReconstructionState::from_object(&problem, truth).unwrap();
    let mut algorithm = GradientDescent::default()
        .object_step(1e-10)
        .recover_illumination(true)
        .illumination_step(0.5)
        .illumination_finite_difference(0.02)
        .illumination_bounds(0.75);
    let batch = Batch::new((0..problem.model.frame_count()).collect(), 0);
    for iteration in 0..8 {
        algorithm
            .step(&problem, &mut state, &batch, iteration)
            .unwrap();
    }

    let corrections = state.illumination_corrections().unwrap();
    let initial_error: f64 = true_offsets
        .iter()
        .map(|offset| offset.row * offset.row + offset.column * offset.column)
        .sum();
    let final_error: f64 = corrections
        .iter()
        .zip(&true_offsets)
        .map(|(&(row, column), truth)| (row - truth.row).powi(2) + (column - truth.column).powi(2))
        .sum();
    assert!(
        final_error < 0.5 * initial_error,
        "expected source-position recovery, error {initial_error} -> {final_error}; corrections={corrections:?}"
    );
}

#[test]
fn joint_illumination_calibration_reduces_model_mismatch() {
    let reconstruction_model = common::direct_model().unwrap();
    let true_offsets = vec![FourierOffset::new(0.22, -0.18); reconstruction_model.source_count()];
    let true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(true_offsets.clone())
        .unwrap();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();

    let uncalibrated = GradientDescent::default()
        .iterations(20)
        .object_step(0.3)
        .batch_size(problem.model.frame_count())
        .run(&problem)
        .unwrap();
    let calibrated = GradientDescent::default()
        .iterations(20)
        .object_step(0.3)
        .batch_size(problem.model.frame_count())
        .recover_illumination(true)
        .illumination_step(0.05)
        .illumination_finite_difference(0.02)
        .illumination_bounds(0.75)
        .run(&problem)
        .unwrap();
    let corrections = calibrated.calibrated_illumination.as_ref().unwrap();
    let initial_error: f64 = true_offsets
        .iter()
        .map(|offset| offset.row.powi(2) + offset.column.powi(2))
        .sum();
    let final_error: f64 = corrections
        .iter()
        .zip(&true_offsets)
        .map(|(&(row, column), truth)| (row - truth.row).powi(2) + (column - truth.column).powi(2))
        .sum();
    assert!(
        calibrated.trace.final_objective().unwrap() < uncalibrated.trace.final_objective().unwrap(),
        "calibration did not improve loss: {} vs {}",
        calibrated.trace.final_objective().unwrap(),
        uncalibrated.trace.final_objective().unwrap()
    );
    assert!(
        final_error < initial_error,
        "calibration error did not improve: {initial_error} -> {final_error}; {corrections:?}"
    );
    let corrected_metrics = evaluate_reconstruction_with_problem(
        &calibrated,
        &problem,
        simulation.ground_truth_object.view(),
        Some(&simulation.true_model),
        None,
    )
    .unwrap();
    assert!(
        corrected_metrics.illumination.unwrap().position_rmse
            < (initial_error / true_offsets.len() as f64).sqrt()
    );
    let corrected_residual: f64 = corrected_metrics
        .intensity
        .unwrap()
        .per_frame
        .iter()
        .map(|frame| frame.normalized_l2)
        .sum();
    let mut ignored_corrections = calibrated.clone();
    ignored_corrections.calibrated_illumination = None;
    let uncorrected_residual: f64 = evaluate_reconstruction_with_problem(
        &ignored_corrections,
        &problem,
        simulation.ground_truth_object.view(),
        None,
        None,
    )
    .unwrap()
    .intensity
    .unwrap()
    .per_frame
    .iter()
    .map(|frame| frame.normalized_l2)
    .sum();
    assert!(corrected_residual < uncorrected_residual);
}

#[test]
fn illumination_calibration_tracks_sources_in_multiplexed_frames() {
    let reconstruction_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.6), (1, 0.4)],
            vec![(2, 0.5), (3, 0.5)],
            vec![(4, 0.7), (0, 0.3)],
        ])
        .unwrap();
    let true_offsets = vec![
        FourierOffset::new(0.16, -0.12),
        FourierOffset::new(-0.14, 0.18),
        FourierOffset::new(0.12, 0.10),
        FourierOffset::new(-0.16, -0.11),
        FourierOffset::new(0.10, -0.16),
    ];
    let true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(true_offsets.clone())
        .unwrap();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let mut state = ReconstructionState::from_object(&problem, truth).unwrap();
    let mut algorithm = GradientDescent::default()
        .object_step(1e-10)
        .recover_illumination(true)
        .illumination_step(0.4)
        .illumination_finite_difference(0.02);
    let batch = Batch::new((0..problem.model.frame_count()).collect(), 0);
    for iteration in 0..10 {
        algorithm
            .step(&problem, &mut state, &batch, iteration)
            .unwrap();
    }

    let corrections = state.illumination_corrections().unwrap();
    assert_eq!(corrections.len(), problem.model.source_count());
    assert_ne!(corrections.len(), problem.model.frame_count());
    let initial_error: f64 = true_offsets
        .iter()
        .map(|offset| offset.row.powi(2) + offset.column.powi(2))
        .sum();
    let final_error: f64 = corrections
        .iter()
        .zip(&true_offsets)
        .map(|(&(row, column), truth)| (row - truth.row).powi(2) + (column - truth.column).powi(2))
        .sum();
    assert!(
        final_error < initial_error,
        "multiplexed source correction did not improve: {initial_error} -> {final_error}; {corrections:?}"
    );
}

#[test]
fn gradient_jointly_recovers_pupil_and_multiplexed_source_offsets() {
    let reconstruction_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.6), (1, 0.4)],
            vec![(2, 0.5), (3, 0.5)],
            vec![(4, 0.7), (0, 0.3)],
        ])
        .unwrap();
    let true_offsets = vec![
        FourierOffset::new(0.12, -0.10),
        FourierOffset::new(-0.11, 0.14),
        FourierOffset::new(0.10, 0.08),
        FourierOffset::new(-0.13, -0.09),
        FourierOffset::new(0.08, -0.12),
    ];
    let mut true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(true_offsets.clone())
        .unwrap();
    let shape = true_model.image_shape();
    let radius_scale = (shape.0.min(shape.1) as f64 / 2.0).max(1.0);
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            if true_model.pupil().support()[(row, column)] == 0 {
                continue;
            }
            let y = (row as f64 - shape.0 as f64 / 2.0) / radius_scale;
            let x = (column as f64 - shape.1 as f64 / 2.0) / radius_scale;
            let rho = x.hypot(y).min(1.0);
            let theta = y.atan2(x);
            let phase = 0.5 * rho * rho + 0.15 * rho * rho * (2.0 * theta).cos();
            true_model.pupil_mut().values_mut()[(row, column)] *= Complex64::from_polar(1.0, phase);
        }
    }
    true_model.validate().unwrap();
    let initial_pupil = reconstruction_model.pupil().clone();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let true_pupil = simulation.true_model.pupil().clone();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let mut state = ReconstructionState::from_object(&problem, truth).unwrap();
    let mut algorithm = GradientDescent::default()
        .object_step(1e-10)
        .recover_pupil(true)
        .pupil_step(0.1)
        .recover_illumination(true)
        .illumination_step(0.2)
        .illumination_finite_difference(0.02);
    let batch = Batch::new((0..problem.model.frame_count()).collect(), 0);
    for iteration in 0..20 {
        algorithm
            .step(&problem, &mut state, &batch, iteration)
            .unwrap();
    }

    let initial_pupil_error = pupil_phase_rmse(
        initial_pupil
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        true_pupil
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        true_pupil
            .support()
            .iter()
            .map(|&value| value != 0)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let recovered_pupil_error = pupil_phase_rmse(
        state
            .pupil()
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        true_pupil
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        true_pupil
            .support()
            .iter()
            .map(|&value| value != 0)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let corrections = state.illumination_corrections().unwrap();
    let initial_offset_error: f64 = true_offsets
        .iter()
        .map(|offset| offset.row.powi(2) + offset.column.powi(2))
        .sum();
    let recovered_offset_error: f64 = corrections
        .iter()
        .zip(&true_offsets)
        .map(|(&(row, column), truth)| (row - truth.row).powi(2) + (column - truth.column).powi(2))
        .sum();
    assert!(
        recovered_pupil_error < initial_pupil_error,
        "pupil error did not improve: {initial_pupil_error} -> {recovered_pupil_error}"
    );
    assert!(
        recovered_offset_error < initial_offset_error,
        "source error did not improve: {initial_offset_error} -> {recovered_offset_error}; {corrections:?}"
    );
}

#[test]
fn illumination_calibration_resumes_exactly_from_checkpoint() {
    let reconstruction_model = common::direct_model().unwrap();
    let true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(vec![
            FourierOffset::new(0.18, -0.14);
            reconstruction_model.source_count()
        ])
        .unwrap();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let directory = tempfile::tempdir().unwrap();
    GradientDescent::default()
        .iterations(2)
        .object_step(0.3)
        .batch_size(problem.model.frame_count())
        .recover_illumination(true)
        .illumination_step(0.05)
        .recover_pupil(true)
        .pupil_step(0.03)
        .run_with_callbacks(
            &problem,
            vec![Box::new(CheckpointEvery::new(2, directory.path()))],
        )
        .unwrap();
    let checkpoint =
        ReconstructionCheckpoint::load(directory.path().join("checkpoint_00002.json")).unwrap();
    let resumed = GradientDescent::default()
        .iterations(4)
        .object_step(0.3)
        .batch_size(problem.model.frame_count())
        .recover_illumination(true)
        .illumination_step(0.05)
        .recover_pupil(true)
        .pupil_step(0.03)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    let uninterrupted = GradientDescent::default()
        .iterations(4)
        .object_step(0.3)
        .batch_size(problem.model.frame_count())
        .recover_illumination(true)
        .illumination_step(0.05)
        .recover_pupil(true)
        .pupil_step(0.03)
        .run(&problem)
        .unwrap();

    for (&resumed, &uninterrupted) in resumed
        .calibrated_illumination
        .as_ref()
        .unwrap()
        .iter()
        .zip(uninterrupted.calibrated_illumination.as_ref().unwrap())
    {
        assert_abs_diff_eq!(resumed.0, uninterrupted.0, epsilon = 1e-12);
        assert_abs_diff_eq!(resumed.1, uninterrupted.1, epsilon = 1e-12);
    }
    for (&resumed, &uninterrupted) in resumed
        .object_spectrum
        .iter()
        .zip(uninterrupted.object_spectrum.iter())
    {
        assert_abs_diff_eq!(resumed.re, uninterrupted.re, epsilon = 1e-14);
        assert_abs_diff_eq!(resumed.im, uninterrupted.im, epsilon = 1e-14);
    }
    for (&resumed, &uninterrupted) in resumed
        .recovered_pupil
        .values()
        .iter()
        .zip(uninterrupted.recovered_pupil.values().iter())
    {
        assert_abs_diff_eq!(resumed.re, uninterrupted.re, epsilon = 1e-12);
        assert_abs_diff_eq!(resumed.im, uninterrupted.im, epsilon = 1e-12);
    }
}

#[test]
fn amplitude_gradient_reconstructs_multiplexed_data() {
    let model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.7), (1, 0.3)],
            vec![(1, 0.6), (2, 0.4)],
            vec![(2, 0.5), (3, 0.5)],
            vec![(3, 0.4), (4, 0.6)],
            vec![(4, 0.3), (0, 0.7)],
        ])
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let ap = AlternatingProjection::default()
        .iterations(12)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    let fpie = Fpie::default()
        .iterations(12)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    let epry = Epry::default()
        .iterations(12)
        .object_step(0.4)
        .pupil_step(0.02)
        .run(&problem)
        .unwrap();
    let admm = Admm::default()
        .iterations(12)
        .object_step(0.5)
        .run(&problem)
        .unwrap();
    for result in [&ap, &fpie, &epry, &admm] {
        assert_eq!(result.runtime.completed_iterations, 12);
        assert!(
            result.trace.iterations.last().unwrap().objective
                < result.trace.iterations.first().unwrap().objective
        );
    }
    let result = GradientDescent::default()
        .iterations(12)
        .object_step(0.4)
        .batch_size(2)
        .run(&problem)
        .unwrap();
    assert_eq!(result.runtime.completed_iterations, 12);
    assert!(
        result.trace.iterations.last().unwrap().objective
            < result.trace.iterations.first().unwrap().objective
    );
}

#[test]
fn gradient_minibatch_applies_the_mean_of_frame_gradients() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let initial = ReconstructionState::initialize(&problem).unwrap();
    let mut batched = initial.clone();
    let mut algorithm = GradientDescent::default().object_step(0.3);
    algorithm
        .step(&problem, &mut batched, &Batch::new(vec![0, 1], 0), 0)
        .unwrap();

    let mut individual_states = Vec::new();
    for frame in [0, 1] {
        let mut state = initial.clone();
        algorithm
            .step(&problem, &mut state, &Batch::single(frame), 0)
            .unwrap();
        individual_states.push(state);
    }
    let initial_values = initial
        .object_spectrum()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let first_values = individual_states[0]
        .object_spectrum()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let second_values = individual_states[1]
        .object_spectrum()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let batched_values = batched
        .object_spectrum()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for (pixel, &initial_value) in initial_values.iter().enumerate() {
        let expected = initial_value
            + ((first_values[pixel] - initial_value) + (second_values[pixel] - initial_value))
                / 2.0;
        assert_abs_diff_eq!(batched_values[pixel].re, expected.re, epsilon = 1e-12);
        assert_abs_diff_eq!(batched_values[pixel].im, expected.im, epsilon = 1e-12);
    }
}

#[test]
fn parallel_gradient_reduction_matches_sequential_for_multiplexed_pupil_updates() {
    let model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.7), (1, 0.3)],
            vec![(1, 0.4), (2, 0.6)],
            vec![(3, 0.5), (4, 0.5)],
        ])
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let initial = ReconstructionState::initialize(&problem).unwrap();
    let batch = Batch::new(vec![0, 1, 2], 0);

    let mut sequential = initial.clone();
    let mut sequential_algorithm = GradientDescent::default()
        .object_step(0.3)
        .recover_pupil(true)
        .pupil_step(0.02)
        .parallel_workers(1);
    let sequential_diagnostics = sequential_algorithm
        .step(&problem, &mut sequential, &batch, 0)
        .unwrap();

    let mut parallel = initial;
    let mut parallel_algorithm = GradientDescent::default()
        .object_step(0.3)
        .recover_pupil(true)
        .pupil_step(0.02)
        .parallel_workers(3);
    let parallel_diagnostics = parallel_algorithm
        .step(&problem, &mut parallel, &batch, 0)
        .unwrap();

    assert_abs_diff_eq!(
        parallel_diagnostics.summary.mean_objective().unwrap(),
        sequential_diagnostics.summary.mean_objective().unwrap(),
        epsilon = 1e-14
    );
    for (&parallel, &sequential) in parallel
        .object_spectrum()
        .iter()
        .zip(sequential.object_spectrum().iter())
    {
        assert_abs_diff_eq!(parallel.re, sequential.re, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.im, sequential.im, epsilon = 1e-12);
    }
    for (&parallel, &sequential) in parallel
        .pupil()
        .values()
        .iter()
        .zip(sequential.pupil().values().iter())
    {
        assert_abs_diff_eq!(parallel.re, sequential.re, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.im, sequential.im, epsilon = 1e-12);
    }
}

#[test]
fn parallel_illumination_reduction_matches_sequential_with_shared_sources() {
    let reconstruction_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![
            vec![(0, 0.7), (1, 0.3)],
            vec![(1, 0.4), (2, 0.6)],
            vec![(0, 0.5), (4, 0.5)],
        ])
        .unwrap();
    let true_model = reconstruction_model
        .clone()
        .with_subpixel_offsets(vec![
            FourierOffset::new(0.12, -0.08),
            FourierOffset::new(-0.09, 0.11),
            FourierOffset::new(0.07, 0.06),
            FourierOffset::default(),
            FourierOffset::new(-0.10, -0.07),
        ])
        .unwrap();
    let simulation = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let initial = ReconstructionState::from_object(&problem, truth).unwrap();
    let batch = Batch::new(vec![0, 1, 2], 0);

    let algorithm = GradientDescent::default()
        .object_step(0.2)
        .recover_pupil(true)
        .pupil_step(0.02)
        .recover_illumination(true)
        .illumination_step(0.2)
        .illumination_finite_difference(0.02);
    let mut sequential = initial.clone();
    let mut sequential_algorithm = algorithm.clone().parallel_workers(1);
    let sequential_diagnostics = sequential_algorithm
        .step(&problem, &mut sequential, &batch, 0)
        .unwrap();

    let mut parallel = initial.clone();
    let mut parallel_algorithm = algorithm.clone().parallel_workers(3);
    let parallel_diagnostics = parallel_algorithm
        .step(&problem, &mut parallel, &batch, 0)
        .unwrap();
    let mut repeated = initial;
    algorithm
        .parallel_workers(3)
        .step(&problem, &mut repeated, &batch, 0)
        .unwrap();

    assert_abs_diff_eq!(
        parallel_diagnostics.summary.mean_objective().unwrap(),
        sequential_diagnostics.summary.mean_objective().unwrap(),
        epsilon = 1e-14
    );
    for (&parallel, &sequential) in parallel
        .object_spectrum()
        .iter()
        .zip(sequential.object_spectrum().iter())
    {
        assert_abs_diff_eq!(parallel.re, sequential.re, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.im, sequential.im, epsilon = 1e-12);
    }
    for (&parallel, &sequential) in parallel
        .pupil()
        .values()
        .iter()
        .zip(sequential.pupil().values().iter())
    {
        assert_abs_diff_eq!(parallel.re, sequential.re, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.im, sequential.im, epsilon = 1e-12);
    }
    for (&parallel, &sequential) in parallel
        .illumination_corrections()
        .unwrap()
        .iter()
        .zip(sequential.illumination_corrections().unwrap())
    {
        assert_abs_diff_eq!(parallel.0, sequential.0, epsilon = 1e-12);
        assert_abs_diff_eq!(parallel.1, sequential.1, epsilon = 1e-12);
    }
    assert_eq!(parallel.object_spectrum(), repeated.object_spectrum());
    assert_eq!(parallel.pupil().values(), repeated.pupil().values());
    assert_eq!(
        parallel.illumination_corrections(),
        repeated.illumination_corrections()
    );
}

#[test]
fn gradient_solver_supports_configurable_losses() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();

    for (loss_type, step) in [
        (LossType::IntensityMse, 0.05),
        (LossType::PoissonNegativeLogLikelihood, 0.2),
        (LossType::HuberAmplitude, 0.5),
    ] {
        let result = GradientDescent::default()
            .iterations(8)
            .object_step(step)
            .loss_type(loss_type)
            .run(&problem)
            .unwrap();
        let first = result.trace.iterations.first().unwrap().objective;
        let last = result.trace.iterations.last().unwrap().objective;
        assert!(
            last.is_finite() && last < first,
            "expected {loss_type:?} loss to decrease, got {first} -> {last}"
        );
    }
}

#[test]
fn object_and_pupil_gradients_match_finite_differences_for_all_losses_and_masks() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut mask = Array2::from_elem(simulation.measurements.image_shape(), 1_u8);
    for (pixel, value) in mask.iter_mut().enumerate() {
        if pixel % 5 == 0 {
            *value = 0;
        }
    }
    let measurements = simulation.measurements.with_masks(mask).unwrap();
    let problem =
        ReconstructionProblem::new(measurements, simulation.reconstruction_model).unwrap();
    let mut initial_object = simulation.ground_truth_object;
    for (index, value) in initial_object.iter_mut().enumerate() {
        let amplitude = 0.78 + 0.03 * (index % 7) as f64 / 7.0;
        let phase = 0.08 * ((index % 11) as f64 / 11.0 - 0.5);
        *value *= Complex64::from_polar(amplitude, phase);
    }
    let initial = ReconstructionState::from_object(&problem, initial_object).unwrap();
    let frame = 2;
    let object_direction = complex_test_direction(initial.object_spectrum().len(), 0.07);
    let pupil_direction = complex_test_direction(initial.pupil().values().len(), 0.05);

    for loss_type in [
        LossType::AmplitudeMse,
        LossType::IntensityMse,
        LossType::PoissonNegativeLogLikelihood,
        LossType::HuberAmplitude,
    ] {
        let mut state = initial.clone();
        let object_before = state.object_spectrum().to_owned();
        let pupil_before = state.pupil().clone();
        let mut algorithm = GradientDescent::default()
            .object_step(1.0)
            .loss_type(loss_type)
            .recover_pupil(true)
            .pupil_step(1.0)
            .constrain_pupil_support(false)
            .parallel_workers(1);
        algorithm
            .step(&problem, &mut state, &Batch::single(frame), 0)
            .unwrap();

        let object_denominator = pupil_before
            .values()
            .iter()
            .map(|value| value.norm_sqr())
            .fold(0.0, f64::max)
            .max(algorithm.epsilon)
            + algorithm.epsilon;
        let object_gradient: Vec<_> = object_before
            .iter()
            .zip(state.object_spectrum().iter())
            .map(|(&before, &after)| (before - after) * object_denominator)
            .collect();
        let analytical_object = complex_directional_derivative(&object_gradient, &object_direction);
        let numerical_object = finite_difference_object_loss(
            &problem,
            &object_before,
            &pupil_before,
            frame,
            loss_type,
            &object_direction,
            1e-6,
        );
        assert_derivative_close(
            &format!("{loss_type:?} object"),
            analytical_object,
            numerical_object,
            2e-5,
        );

        let mut patch = vec![Complex64::default(); pupil_before.values().len()];
        problem
            .model
            .extract_patch(object_before.view(), frame, &mut patch)
            .unwrap();
        let pupil_denominator = patch
            .iter()
            .map(|value| value.norm_sqr())
            .fold(0.0, f64::max)
            .max(algorithm.epsilon)
            + algorithm.epsilon;
        let pupil_gradient: Vec<_> = pupil_before
            .values()
            .iter()
            .zip(state.pupil().values().iter())
            .map(|(&before, &after)| (before - after) * pupil_denominator)
            .collect();
        let analytical_pupil = complex_directional_derivative(&pupil_gradient, &pupil_direction);
        let numerical_pupil = finite_difference_pupil_loss(
            &problem,
            &object_before,
            &pupil_before,
            frame,
            loss_type,
            &pupil_direction,
            1e-6,
        );
        assert_derivative_close(
            &format!("{loss_type:?} pupil"),
            analytical_pupil,
            numerical_pupil,
            2e-5,
        );
    }
}

#[test]
fn illumination_gradients_match_direct_loss_differences_for_all_losses_and_masks() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut mask = Array2::from_elem(simulation.measurements.image_shape(), 1_u8);
    for (pixel, value) in mask.iter_mut().enumerate() {
        if pixel % 6 == 1 {
            *value = 0;
        }
    }
    let measurements = simulation.measurements.with_masks(mask).unwrap();
    let problem =
        ReconstructionProblem::new(measurements, simulation.reconstruction_model).unwrap();
    let mut initial_object = simulation.ground_truth_object;
    for (index, value) in initial_object.iter_mut().enumerate() {
        *value *= Complex64::from_polar(0.82, 0.04 * (index % 9) as f64 / 9.0);
    }
    let initial = ReconstructionState::from_object(&problem, initial_object).unwrap();
    let frame = 2;
    let distance = 1e-4;

    for loss_type in [
        LossType::AmplitudeMse,
        LossType::IntensityMse,
        LossType::PoissonNegativeLogLikelihood,
        LossType::HuberAmplitude,
    ] {
        let mut state = initial.clone();
        let object = state.object_spectrum().to_owned();
        let pupil = state.pupil().clone();
        let mut algorithm = GradientDescent::default()
            .object_step(1e-12)
            .loss_type(loss_type)
            .recover_illumination(true)
            .illumination_step(1e-12)
            .illumination_finite_difference(distance)
            .parallel_workers(1);
        algorithm
            .step(&problem, &mut state, &Batch::single(frame), 0)
            .unwrap();
        let analytical = state.illumination_gradient()[frame];
        let numerical_row = finite_difference_illumination_loss(
            &problem,
            &object,
            &pupil,
            frame,
            loss_type,
            FourierOffset::new(distance, 0.0),
        );
        let numerical_column = finite_difference_illumination_loss(
            &problem,
            &object,
            &pupil,
            frame,
            loss_type,
            FourierOffset::new(0.0, distance),
        );
        assert_derivative_close(
            &format!("{loss_type:?} illumination row"),
            analytical.0,
            numerical_row,
            5e-3,
        );
        assert_derivative_close(
            &format!("{loss_type:?} illumination column"),
            analytical.1,
            numerical_column,
            5e-3,
        );
    }
}

fn complex_test_direction(len: usize, scale: f64) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let real = (index % 13) as f64 / 13.0 - 0.5;
            let imaginary = (index % 17) as f64 / 17.0 - 0.5;
            Complex64::new(scale * real, scale * imaginary)
        })
        .collect()
}

fn complex_directional_derivative(gradient: &[Complex64], direction: &[Complex64]) -> f64 {
    2.0 * gradient
        .iter()
        .zip(direction)
        .map(|(&gradient, &direction)| (gradient.conj() * direction).re)
        .sum::<f64>()
}

fn finite_difference_object_loss<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    object: &Array2<Complex64>,
    pupil: &Pupil,
    frame: usize,
    loss_type: LossType,
    direction: &[Complex64],
    distance: f64,
) -> f64 {
    let mut plus = object.clone();
    let mut minus = object.clone();
    for ((plus, minus), &direction) in plus.iter_mut().zip(minus.iter_mut()).zip(direction) {
        *plus += distance * direction;
        *minus -= distance * direction;
    }
    (masked_frame_loss(problem, &plus, pupil, frame, loss_type)
        - masked_frame_loss(problem, &minus, pupil, frame, loss_type))
        / (2.0 * distance)
}

fn finite_difference_pupil_loss<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    object: &Array2<Complex64>,
    pupil: &Pupil,
    frame: usize,
    loss_type: LossType,
    direction: &[Complex64],
    distance: f64,
) -> f64 {
    let mut plus = pupil.clone();
    let mut minus = pupil.clone();
    for ((plus, minus), &direction) in plus
        .values_mut()
        .iter_mut()
        .zip(minus.values_mut().iter_mut())
        .zip(direction)
    {
        *plus += distance * direction;
        *minus -= distance * direction;
    }
    (masked_frame_loss(problem, object, &plus, frame, loss_type)
        - masked_frame_loss(problem, object, &minus, frame, loss_type))
        / (2.0 * distance)
}

fn finite_difference_illumination_loss(
    problem: &ReconstructionProblem<fpm_rs::measurements::MeasurementStack>,
    object: &Array2<Complex64>,
    pupil: &Pupil,
    source: usize,
    loss_type: LossType,
    displacement: FourierOffset,
) -> f64 {
    let offsets: Vec<_> = (0..problem.model.source_count())
        .map(|index| problem.model.source_offset(index).unwrap())
        .collect();
    let mut plus_offsets = offsets.clone();
    let mut minus_offsets = offsets;
    plus_offsets[source].row += displacement.row;
    plus_offsets[source].column += displacement.column;
    minus_offsets[source].row -= displacement.row;
    minus_offsets[source].column -= displacement.column;
    let plus_problem = ReconstructionProblem::new(
        problem.measurements.clone(),
        problem
            .model
            .clone()
            .with_subpixel_offsets(plus_offsets)
            .unwrap(),
    )
    .unwrap();
    let minus_problem = ReconstructionProblem::new(
        problem.measurements.clone(),
        problem
            .model
            .clone()
            .with_subpixel_offsets(minus_offsets)
            .unwrap(),
    )
    .unwrap();
    let distance = displacement.row.abs() + displacement.column.abs();
    (masked_frame_loss(&plus_problem, object, pupil, source, loss_type)
        - masked_frame_loss(&minus_problem, object, pupil, source, loss_type))
        / (2.0 * distance)
}

fn masked_frame_loss<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    object: &Array2<Complex64>,
    pupil: &Pupil,
    frame: usize,
    loss_type: LossType,
) -> f64 {
    let predicted = ForwardModel::new(&problem.model)
        .unwrap()
        .forward_intensity(object.view(), pupil, frame)
        .unwrap();
    let measured = problem.measurements.frame(frame).unwrap();
    let mask = problem.measurements.frame_mask(frame).unwrap();
    let mut valid_prediction = Vec::new();
    let mut valid_measurement = Vec::new();
    for pixel in 0..predicted.len() {
        if mask.is_some_and(|mask| mask[pixel] == 0) {
            continue;
        }
        valid_prediction.push(predicted.iter().nth(pixel).copied().unwrap());
        valid_measurement.push(measured[pixel]);
    }
    loss(&valid_prediction, &valid_measurement, loss_type).unwrap()
}

fn assert_derivative_close(label: &str, analytical: f64, numerical: f64, relative: f64) {
    let scale = analytical.abs().max(numerical.abs()).max(1e-10);
    assert!(
        (analytical - numerical).abs() <= relative * scale,
        "{label} derivative mismatch: analytical={analytical:.12e}, numerical={numerical:.12e}"
    );
}

#[test]
fn object_tv_regularization_reduces_complex_variation_across_batch_sizes() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let initial_object = Array2::from_shape_vec(
        (16, 16),
        (0..16)
            .flat_map(|row| {
                (0..16).map(move |column| {
                    let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
                    Complex64::from_polar(1.0 + 0.15 * sign, 0.2 * sign)
                })
            })
            .collect(),
    )
    .unwrap();
    let state = ReconstructionState::from_object(&problem, initial_object.clone()).unwrap();
    let checkpoint = ReconstructionCheckpoint::capture(0, &state, &ReconstructionTrace::default());
    let initial_variation = complex_variation(&initial_object);
    let full_batch = GradientDescent::default()
        .iterations(1)
        .object_step(1e-12)
        .batch_size(problem.model.frame_count())
        .object_tv(0.002)
        .run_from_checkpoint(&problem, checkpoint.clone())
        .unwrap();
    let single_frame_batches = GradientDescent::default()
        .iterations(1)
        .object_step(1e-12)
        .batch_size(1)
        .object_tv(0.002)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    let full_variation = complex_variation(&full_batch.object);
    let single_variation = complex_variation(&single_frame_batches.object);
    assert!(full_variation < initial_variation);
    assert!(single_variation < initial_variation);
    assert!(
        (full_variation - single_variation).abs() < 0.01 * initial_variation,
        "batch-scaled TV diverged: full={full_variation}, single={single_variation}"
    );
}

#[test]
fn pupil_smoothing_reduces_roughness_and_preserves_support() {
    let mut model = common::direct_model().unwrap();
    let shape = model.image_shape();
    let mut pupil_values = model.pupil().values().to_owned();
    let mut pupil_support = model.pupil().support().to_owned();
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
            pupil_values[(row, column)] = Complex64::from_polar(1.0, 0.35 * sign);
        }
    }
    pupil_support[(0, 0)] = 0;
    pupil_values[(0, 0)] = Complex64::default();
    *model.pupil_mut() = Pupil::new(pupil_values, pupil_support).unwrap();
    model.validate().unwrap();
    let initial_roughness = complex_roughness(&model.pupil().values().to_owned());
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = GradientDescent::default()
        .iterations(1)
        .object_step(1e-12)
        .batch_size(problem.model.frame_count())
        .recover_pupil(true)
        .pupil_step(0.0)
        .pupil_smoothing(0.02)
        .run(&problem)
        .unwrap();
    assert!(complex_roughness(&result.recovered_pupil.values().to_owned()) < initial_roughness);
    assert_eq!(
        result.recovered_pupil.values()[(0, 0)],
        Complex64::default()
    );
}

#[test]
fn gradient_loss_is_invariant_to_known_linear_camera_response() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::mixed_test_pattern((16, 16)).unwrap();
    let ideal = Simulator::ideal(model.clone())
        .object(object.clone())
        .simulate()
        .unwrap();
    let camera = Simulator::ideal(model)
        .object(object)
        .camera(
            CameraModel::new()
                .photons_per_pixel(20.0)
                .gain(3.0)
                .dark_current_electrons(2.0)
                .offset_counts(7.0)
                .quantize(false),
        )
        .simulate()
        .unwrap();
    let ideal_problem =
        ReconstructionProblem::new(ideal.measurements, ideal.reconstruction_model).unwrap();
    let camera_problem =
        ReconstructionProblem::new(camera.measurements, camera.reconstruction_model).unwrap();

    let ideal_result = GradientDescent::default()
        .iterations(4)
        .run(&ideal_problem)
        .unwrap();
    let camera_result = GradientDescent::default()
        .iterations(4)
        .run(&camera_problem)
        .unwrap();

    for (ideal, camera) in ideal_result
        .trace
        .iterations
        .iter()
        .zip(&camera_result.trace.iterations)
    {
        assert_abs_diff_eq!(ideal.objective, camera.objective, epsilon = 1e-12);
    }
    for (&ideal, &camera) in ideal_result
        .object_spectrum
        .iter()
        .zip(camera_result.object_spectrum.iter())
    {
        assert_abs_diff_eq!(ideal.re, camera.re, epsilon = 1e-11);
        assert_abs_diff_eq!(ideal.im, camera.im, epsilon = 1e-11);
    }
}

#[test]
fn epry_runs_joint_object_pupil_updates() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::phase_disk((16, 16), 4.5, 0.8).unwrap();
    let simulation = Simulator::ideal(model).object(object).simulate().unwrap();
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Epry::default()
        .iterations(4)
        .pupil_step(0.05)
        .run(&problem)
        .unwrap();
    let metrics = evaluate_reconstruction(&result, truth.view(), None, None).unwrap();
    assert!(metrics.object.amplitude_rmse.is_finite());
    assert!(metrics.object.phase_rmse.is_finite());
    assert_eq!(result.runtime.completed_iterations, 4);
}

#[test]
fn initialization_undoes_known_frame_gain_and_background() {
    let model = common::direct_model()
        .unwrap()
        .with_frame_gains(Some(vec![4.0; 5]))
        .unwrap()
        .with_background(Some(vec![9.0; 64]))
        .unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::constant((16, 16), 2.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = AlternatingProjection::default()
        .iterations(0)
        .run(&problem)
        .unwrap();
    for amplitude in &result.amplitude {
        assert_abs_diff_eq!(*amplitude, 2.0, epsilon = 1e-10);
    }
}

#[test]
fn camera_counts_reconstruct_in_compiled_sensor_units() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::new(model)
        .object(SyntheticObject::constant((16, 16), 2.0, 0.0).unwrap())
        .camera(
            CameraModel::new()
                .photons_per_pixel(100.0)
                .gain(2.0)
                .dark_current_electrons(3.0)
                .offset_counts(5.0)
                .quantize(false),
        )
        .simulate()
        .unwrap();
    assert_eq!(
        simulation.reconstruction_model.frame_gains(),
        Some(&vec![200.0; simulation.reconstruction_model.frame_count()][..])
    );
    assert_eq!(
        simulation.reconstruction_model.background(),
        Some(&vec![11.0; 64][..])
    );
    assert!(
        simulation
            .measurements
            .as_slice()
            .iter()
            .all(|&value| (value - 811.0).abs() < 1e-10)
    );
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = AlternatingProjection::default()
        .iterations(0)
        .run(&problem)
        .unwrap();
    assert!(
        result
            .amplitude
            .iter()
            .all(|&value| (value - 2.0).abs() < 1e-10)
    );
}

#[test]
fn zero_weight_frames_do_not_affect_reconstruction() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut clean = simulation.measurements.clone();
    let mut corrupted = simulation.measurements;
    for frame in 1..clean.frame_count() {
        clean.set_frame_weight(frame, 0.0).unwrap();
        corrupted.set_frame_weight(frame, 0.0).unwrap();
    }
    for frame in 1..corrupted.frame_count() {
        corrupted.frame_mut(frame).unwrap().fill(1.0e12);
    }
    let clean_problem =
        ReconstructionProblem::new(clean, simulation.reconstruction_model.clone()).unwrap();
    let corrupted_problem =
        ReconstructionProblem::new(corrupted, simulation.reconstruction_model).unwrap();
    let clean_result = AlternatingProjection::default()
        .iterations(3)
        .run(&clean_problem)
        .unwrap();
    let corrupted_result = AlternatingProjection::default()
        .iterations(3)
        .run(&corrupted_problem)
        .unwrap();
    for (&clean, &corrupted) in clean_result
        .object_spectrum
        .iter()
        .zip(corrupted_result.object_spectrum.iter())
    {
        assert_abs_diff_eq!(clean.re, corrupted.re, epsilon = 1e-12);
        assert_abs_diff_eq!(clean.im, corrupted.im, epsilon = 1e-12);
    }
}

#[test]
fn masked_pixels_do_not_affect_reconstruction() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut corrupted = simulation.measurements.clone();
    for frame in 0..corrupted.frame_count() {
        corrupted.frame_mut(frame).unwrap()[0] = 1.0e12;
    }
    let mut mask = Array2::from_elem(corrupted.image_shape(), 1_u8);
    mask[(0, 0)] = 0;
    let clean = simulation.measurements.with_masks(mask.clone()).unwrap();
    let corrupted = corrupted.with_masks(mask).unwrap();
    let clean_problem =
        ReconstructionProblem::new(clean, simulation.reconstruction_model.clone()).unwrap();
    let corrupted_problem =
        ReconstructionProblem::new(corrupted, simulation.reconstruction_model).unwrap();
    let clean_result = AlternatingProjection::default()
        .iterations(3)
        .run(&clean_problem)
        .unwrap();
    let corrupted_result = AlternatingProjection::default()
        .iterations(3)
        .run(&corrupted_problem)
        .unwrap();
    assert_eq!(
        clean_result.object_spectrum,
        corrupted_result.object_spectrum
    );
}

#[test]
fn invalid_algorithm_options_fail_before_iteration() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let algorithm = AlternatingProjection {
        object_step: f64::NAN,
        ..AlternatingProjection::default()
    };
    assert!(algorithm.run(&problem).is_err());
    let invalid_gain_options = Epry {
        recover_frame_gains: true,
        minimum_gain: 2.0,
        maximum_gain: 1.0,
        ..Epry::default()
    };
    assert!(invalid_gain_options.run(&problem).is_err());
    let invalid_background_options = Epry {
        recover_background: true,
        background_step: f64::NAN,
        ..Epry::default()
    };
    assert!(invalid_background_options.run(&problem).is_err());
    assert!(
        GradientDescent::default()
            .recover_illumination(true)
            .illumination_finite_difference(0.0)
            .run(&problem)
            .is_err()
    );
    assert!(
        GradientDescent::default()
            .recover_pupil(true)
            .pupil_step(f64::NAN)
            .run(&problem)
            .is_err()
    );
    assert!(
        GradientDescent::default()
            .object_tv(-1.0)
            .run(&problem)
            .is_err()
    );
    assert!(
        GradientDescent::default()
            .pupil_smoothing(0.1)
            .run(&problem)
            .is_err()
    );
    assert!(Admm::default().penalty(0.0).run(&problem).is_err());
}

#[test]
fn epry_reduces_known_pupil_phase_error() {
    let assumed_optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.10,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let true_optics = Optics {
        defocus_distance: Some(-24e-6),
        pupil_aberration: Some(PupilAberration {
            astigmatism: 0.2,
            ..PupilAberration::default()
        }),
        ..assumed_optics.clone()
    };
    let leds = Illumination::from_geometry(PlanarLedArray::new(
        (3, 3),
        (4e-3, 4e-3),
        (1.0, 1.0),
        ArrayPose::from_translation([0.0, 0.0, -90e-3]),
    ))
    .unwrap();
    let reconstruction_model = ImagePlaneModel::from_experiment(
        &assumed_optics,
        &leds,
        (16, 16),
        ReconstructionShape::Exact((32, 32)),
    )
    .unwrap();
    let true_model = ImagePlaneModel::from_experiment(
        &true_optics,
        &leds,
        (16, 16),
        ReconstructionShape::Exact((32, 32)),
    )
    .unwrap();
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::phase_disk((32, 32), 8.0, 0.9).unwrap())
        .reconstruction_model(reconstruction_model.clone())
        .simulate()
        .unwrap();
    let initial_error = pupil_phase_rmse(
        &reconstruction_model
            .pupil()
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &simulation
            .true_model
            .pupil()
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &simulation
            .true_model
            .pupil()
            .support()
            .iter()
            .map(|&value| value != 0)
            .collect::<Vec<_>>(),
    );
    let truth = simulation.ground_truth_object.clone();
    let true_model = simulation.true_model.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Epry::default()
        .iterations(40)
        .pupil_step(0.05)
        .run(&problem)
        .unwrap();
    let recovered_error = evaluate_reconstruction(&result, truth.view(), Some(&true_model), None)
        .unwrap()
        .pupil
        .unwrap()
        .phase_rmse;
    assert!(
        recovered_error < initial_error,
        "expected Epry pupil improvement, got {initial_error} -> {recovered_error}"
    );
}

#[test]
fn pupil_metrics_remove_global_complex_scale() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let truth = simulation.ground_truth_object.clone();
    let true_model = simulation.true_model.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let mut result = AlternatingProjection::default()
        .iterations(0)
        .run(&problem)
        .unwrap();
    let ambiguity = Complex64::from_polar(3.0, 0.7);
    for value in result.recovered_pupil.values_mut() {
        *value *= ambiguity;
    }
    let metrics = evaluate_reconstruction(&result, truth.view(), Some(&true_model), None).unwrap();
    assert!(metrics.pupil.as_ref().unwrap().amplitude_rmse < 1e-12);
    assert!(metrics.pupil.as_ref().unwrap().phase_rmse < 1e-12);
}

#[test]
fn epry_recovers_relative_frame_gain_mismatch() {
    let true_gains = vec![0.5, 1.0, 1.5, 2.0, 0.75];
    let true_model = common::direct_model()
        .unwrap()
        .with_frame_gains(Some(true_gains.clone()))
        .unwrap();
    let reconstruction_model = common::direct_model().unwrap();
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::constant((16, 16), 2.0, 0.0).unwrap())
        .reconstruction_model(reconstruction_model)
        .simulate()
        .unwrap();
    let truth = simulation.ground_truth_object.clone();
    let true_model = simulation.true_model.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Epry::default()
        .iterations(2)
        .object_step(1e-6)
        .recover_pupil(false)
        .recover_frame_gains(true)
        .gain_step(1.0)
        .run(&problem)
        .unwrap();
    let recovered_error = evaluate_reconstruction(&result, truth.view(), Some(&true_model), None)
        .unwrap()
        .frame_gains
        .unwrap()
        .relative_error;
    let mean_gain = true_gains.iter().sum::<f64>() / true_gains.len() as f64;
    let initial_error = (true_gains
        .iter()
        .map(|gain| (mean_gain - gain).powi(2))
        .sum::<f64>()
        / true_gains.iter().map(|gain| gain * gain).sum::<f64>())
    .sqrt();
    assert!(
        recovered_error < 0.1 * initial_error,
        "expected relative gain recovery, got {initial_error} -> {recovered_error}"
    );
}

#[test]
fn epry_recovers_relative_per_frame_background() {
    let true_backgrounds = [0.0, 1.0, 2.0, 3.0, 0.5];
    let image_len = 64;
    let true_model = common::direct_model()
        .unwrap()
        .with_background(Some(
            true_backgrounds
                .iter()
                .flat_map(|&value| vec![value; image_len])
                .collect(),
        ))
        .unwrap();
    let reconstruction_model = common::direct_model().unwrap();
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::constant((16, 16), 2.0, 0.0).unwrap())
        .reconstruction_model(reconstruction_model)
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = Epry::default()
        .iterations(1)
        .object_step(1e-6)
        .recover_pupil(false)
        .recover_background(true)
        .background_step(1.0)
        .background_bounds(-10.0, 10.0)
        .run(&problem)
        .unwrap();
    let recovered = result.recovered_background.unwrap();
    let recovered_frame_means: Vec<f64> = recovered
        .chunks_exact(image_len)
        .map(|frame| frame.iter().sum::<f64>() / image_len as f64)
        .collect();
    let recovered_mean =
        recovered_frame_means.iter().sum::<f64>() / recovered_frame_means.len() as f64;
    let true_mean = true_backgrounds.iter().sum::<f64>() / true_backgrounds.len() as f64;
    for (&recovered, &truth) in recovered_frame_means.iter().zip(&true_backgrounds) {
        assert_abs_diff_eq!(
            recovered - recovered_mean,
            truth - true_mean,
            epsilon = 1e-8
        );
    }
}

fn pupil_phase_rmse(recovered: &[Complex64], truth: &[Complex64], support: &[bool]) -> f64 {
    let cross: Complex64 = recovered
        .iter()
        .zip(truth)
        .zip(support)
        .filter(|&(_, &inside)| inside)
        .map(|((&recovered, &truth), _)| recovered * truth.conj())
        .sum();
    let correction = Complex64::from_polar(1.0, -cross.arg());
    let mut sum = 0.0;
    let mut count = 0;
    for ((&recovered, &truth), &inside) in recovered.iter().zip(truth).zip(support) {
        if inside {
            let difference = ((recovered * correction).arg() - truth.arg() + std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            sum += difference * difference;
            count += 1;
        }
    }
    (sum / count as f64).sqrt()
}

fn complex_variation(values: &Array2<Complex64>) -> f64 {
    let mut variation = 0.0;
    let shape = values.dim();
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let value = values[(row, column)];
            let horizontal = if column + 1 < shape.1 {
                (values[(row, column + 1)] - value).norm_sqr()
            } else {
                0.0
            };
            let vertical = if row + 1 < shape.0 {
                (values[(row + 1, column)] - value).norm_sqr()
            } else {
                0.0
            };
            variation += (horizontal + vertical).sqrt();
        }
    }
    variation
}

fn complex_roughness(values: &Array2<Complex64>) -> f64 {
    let mut roughness = 0.0;
    let shape = values.dim();
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let value = values[(row, column)];
            if column + 1 < shape.1 {
                roughness += (values[(row, column + 1)] - value).norm_sqr();
            }
            if row + 1 < shape.0 {
                roughness += (values[(row + 1, column)] - value).norm_sqr();
            }
        }
    }
    roughness
}
