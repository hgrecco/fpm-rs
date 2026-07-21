mod common;

use approx::assert_abs_diff_eq;
use fpm_rs::algorithms::{
    AlternatingProjection, ReconstructionAlgorithm,
    objective::{LossType, loss},
};
use fpm_rs::callbacks::{Callback, CallbackAction, CallbackHook, StepContext};
use fpm_rs::diagnostics::{
    DiagnosticRecorder, DiagnosticRecorderConfig, DiagnosticRequest, Diagnostics,
    FrameDiagnosticRecord, IterationDiagnostics, RawFrameStatisticsRecord,
    ReconstructionDiagnostics, ReconstructionHistory, compute_fourier_coverage,
};
use fpm_rs::{
    Array2, Complex64, Error, Result,
    metrics::{
        complex_field::{ComplexFieldComparisonMetrics, compare_complex_fields},
        intensity::{
            IntensityComparisonMetrics, IntensityStatistics, compare_intensity,
            compare_intensity_masked, intensity_statistics,
        },
    },
    reconstruction::{ReconstructionProblem, ReconstructionState},
    simulation::{Simulator, SyntheticObject},
};

struct FailingIterationCallback;

impl Callback for FailingIterationCallback {
    fn on_iteration_end(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Err(Error::Unsupported("intentional callback failure".into()))
    }
}

#[test]
fn loss_modes_match_known_scalar_values() {
    let predicted = [4.0];
    let measured = [1.0];
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::AmplitudeMse).unwrap(),
        1.0
    );
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::IntensityMse).unwrap(),
        9.0
    );
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::HuberAmplitude).unwrap(),
        0.5
    );
    assert_abs_diff_eq!(
        loss(
            &predicted,
            &measured,
            LossType::PoissonNegativeLogLikelihood
        )
        .unwrap(),
        4.0 - 4.0_f64.ln()
    );
}

#[test]
fn loss_rejects_empty_mismatched_and_non_finite_inputs() {
    assert!(loss(&[], &[], LossType::AmplitudeMse).is_err());
    assert!(loss(&[1.0], &[1.0, 2.0], LossType::AmplitudeMse).is_err());
    assert!(loss(&[f64::NAN], &[1.0], LossType::AmplitudeMse).is_err());
}

#[test]
fn raw_frame_stats_match_known_values() {
    let stats = intensity_statistics(&[0.0, 1.0, 2.0, 3.0], Some(2.0)).unwrap();
    assert_abs_diff_eq!(stats.mean, 1.5, epsilon = 1e-14);
    assert_abs_diff_eq!(stats.std, (1.25f64).sqrt(), epsilon = 1e-14);
    assert_abs_diff_eq!(stats.min, 0.0, epsilon = 1e-14);
    assert_abs_diff_eq!(stats.max, 3.0, epsilon = 1e-14);
    assert_abs_diff_eq!(stats.sum, 6.0, epsilon = 1e-14);
    assert_eq!(stats.saturated_pixels, 2);
    assert_eq!(stats.zero_pixels, 1);
}

#[test]
fn frame_diagnostics_match_known_values() {
    let measured = [1.0, 2.0];
    let predicted = [2.0, 0.0];
    let diagnostics = compare_intensity(&measured, &predicted, Some(2.0)).unwrap();
    assert_abs_diff_eq!(diagnostics.reference_sum, 3.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.candidate_sum, 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_l1, 3.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_l2, (5.0f64).sqrt(), epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_mean, -0.5, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_std, 1.5, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_max_abs, 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.normalized_l2, 1.0, epsilon = 1e-14);
    assert_eq!(diagnostics.saturated_pixels, Some(1));
}

#[test]
fn frame_diagnostics_handles_zero_measured_frames() {
    let measured = [0.0, 0.0];
    let predicted = [1.0, 2.0];
    let diagnostics = compare_intensity(&measured, &predicted, None).unwrap();
    assert!(diagnostics.normalized_l2.is_finite());
    assert!(diagnostics.normalized_l2 > 0.0);
}

#[test]
fn frame_diagnostics_exclude_masked_pixels() {
    let diagnostics =
        compare_intensity_masked(&[1.0, 100.0], &[1.0, 0.0], Some(&[1, 0]), None).unwrap();
    assert_abs_diff_eq!(diagnostics.reference_sum, 1.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.candidate_sum, 1.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.residual_l2, 0.0, epsilon = 1e-14);
    assert_abs_diff_eq!(diagnostics.normalized_l2, 0.0, epsilon = 1e-14);
}

#[test]
fn ground_truth_metrics_align_global_phase() {
    let truth = Array2::from_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let phase = Complex64::from_polar(1.0, 0.7);
    let reconstruction = Array2::from_vec(
        (2, 2),
        truth
            .as_slice()
            .iter()
            .map(|&value| value * phase)
            .collect(),
    )
    .unwrap();
    let metrics = compare_complex_fields(&truth, &reconstruction).unwrap();
    assert_abs_diff_eq!(metrics.amplitude_rmse, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(metrics.amplitude_nrmse, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(metrics.complex_rmse, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(metrics.complex_nrmse, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(metrics.phase_rmse, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(metrics.phase_mae, 0.0, epsilon = 1e-12);
}

#[test]
fn reconstruction_diagnostics_json_round_trips() {
    let diagnostics = ReconstructionDiagnostics {
        iteration_history: vec![IterationDiagnostics {
            iteration: 1,
            total_loss: Some(1.0),
            data_loss: Some(0.8),
            regularization_loss: Some(0.2),
            object_relative_change: Some(0.1),
            pupil_relative_change: Some(0.2),
            median_frame_loss: Some(0.3),
            worst_frame_loss: Some(0.4),
            elapsed_ms: Some(12.0),
        }],
        frame_diagnostics: vec![FrameDiagnosticRecord {
            iteration: Some(1),
            frame_index: 0,
            illumination_index: 1,
            metrics: IntensityComparisonMetrics {
                reference_sum: 10.0,
                candidate_sum: 9.0,
                residual_l1: 1.0,
                residual_l2: 1.0,
                residual_mean: 0.0,
                residual_std: 1.0,
                residual_max_abs: 1.0,
                normalized_l2: 0.1,
                saturated_pixels: Some(2),
            },
        }],
        raw_frame_stats: vec![RawFrameStatisticsRecord {
            frame_index: 0,
            metrics: IntensityStatistics {
                mean: 1.0,
                std: 0.5,
                min: 0.0,
                max: 2.0,
                sum: 4.0,
                saturated_pixels: 2,
                zero_pixels: 1,
            },
        }],
        coverage: Some(compute_fourier_coverage(&common::direct_model().unwrap()).unwrap()),
        ground_truth_metrics: Some(ComplexFieldComparisonMetrics {
            amplitude_rmse: 0.0,
            amplitude_nrmse: 0.0,
            complex_rmse: 0.0,
            complex_nrmse: 0.0,
            phase_rmse: 0.0,
            phase_mae: 0.0,
            fourier_nrmse: 0.0,
            global_phase_offset: 0.0,
        }),
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("diagnostics.json");
    diagnostics.to_json_file(&path).unwrap();
    let loaded = ReconstructionDiagnostics::from_json_file(&path).unwrap();
    assert_eq!(loaded.iteration_history.len(), 1);
    assert_eq!(loaded.frame_diagnostics.len(), 1);
    assert_eq!(loaded.raw_frame_stats.len(), 1);
    assert!(loaded.coverage.is_some());
    assert!(loaded.ground_truth_metrics.is_some());
}

#[test]
fn diagnostic_recorder_default_disables_snapshots() {
    let config = DiagnosticRecorderConfig::default();
    assert_eq!(config.every, 1);
    assert!(config.record_iteration_history);
    assert!(!config.record_object_snapshots);
    assert!(!config.record_pupil_snapshots);
}

#[test]
fn diagnostic_recorder_respects_every() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let state = ReconstructionState::initialize(&problem).unwrap();
    let history = ReconstructionHistory::default();

    let start_diagnostics = Diagnostics {
        raw_frame_stats: Some(vec![RawFrameStatisticsRecord {
            frame_index: 0,
            metrics: IntensityStatistics {
                mean: 1.0,
                std: 0.0,
                min: 1.0,
                max: 1.0,
                sum: 1.0,
                saturated_pixels: 0,
                zero_pixels: 0,
            },
        }]),
        ..Diagnostics::default()
    };
    let start_context = StepContext {
        iteration: 0,
        frame_index: None,
        batch_index: None,
        state: &state,
        diagnostics: &start_diagnostics,
        history: &history,
        model: &problem.model,
        problem_name: None,
    };

    let mut recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig {
        every: 2,
        ..DiagnosticRecorderConfig::default()
    });
    assert_eq!(
        recorder.on_start(&start_context).unwrap(),
        CallbackAction::Continue
    );
    assert_eq!(recorder.diagnostics().raw_frame_stats.len(), 1);

    let iteration_one = Diagnostics {
        loss: Some(1.0),
        per_frame_error: Some(vec![1.0, 3.0, 2.0]),
        ..Diagnostics::default()
    };
    let iteration_two = Diagnostics {
        loss: Some(2.0),
        per_frame_error: Some(vec![1.0, 3.0, 2.0]),
        ..Diagnostics::default()
    };
    let iteration_context = |iteration, diagnostics| StepContext {
        iteration,
        frame_index: None,
        batch_index: None,
        state: &state,
        diagnostics,
        history: &history,
        model: &problem.model,
        problem_name: None,
    };

    recorder
        .on_iteration_end(&iteration_context(1, &iteration_one))
        .unwrap();
    recorder
        .on_iteration_end(&iteration_context(2, &iteration_two))
        .unwrap();

    assert_eq!(recorder.diagnostics().iteration_history.len(), 1);
    assert_eq!(recorder.diagnostics().iteration_history[0].iteration, 2);
    assert_abs_diff_eq!(
        recorder.diagnostics().iteration_history[0]
            .median_frame_loss
            .unwrap(),
        2.0,
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(
        recorder.diagnostics().iteration_history[0]
            .worst_frame_loss
            .unwrap(),
        3.0,
        epsilon = 1e-14
    );
}

#[test]
fn diagnostic_recorder_snapshot_cadence_is_independent() {
    let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig {
        every: 2,
        record_object_snapshots: true,
        snapshot_every: 3,
        ..DiagnosticRecorderConfig::default()
    });

    let iteration_two = recorder.requires_for(CallbackHook::IterationEnd, 2);
    assert!(iteration_two.contains(&DiagnosticRequest::Loss));
    assert!(!iteration_two.contains(&DiagnosticRequest::ObjectAmplitude));

    let iteration_three = recorder.requires_for(CallbackHook::IterationEnd, 3);
    assert!(!iteration_three.contains(&DiagnosticRequest::Loss));
    assert!(iteration_three.contains(&DiagnosticRequest::ObjectAmplitude));
    assert!(iteration_three.contains(&DiagnosticRequest::ObjectPhase));
}

#[test]
fn diagnostic_recorder_output_remains_accessible_after_runner_consumes_clone() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig {
        record_frame_summaries: true,
        ..DiagnosticRecorderConfig::default()
    });

    AlternatingProjection::default()
        .iterations(2)
        .run_with_callbacks(&problem, vec![Box::new(recorder.clone())])
        .unwrap();

    let diagnostics = recorder.diagnostics();
    assert_eq!(diagnostics.iteration_history.len(), 2);
    assert_eq!(
        diagnostics.frame_diagnostics.len(),
        2 * problem.model.frame_count()
    );
    assert!(
        diagnostics
            .frame_diagnostics
            .iter()
            .all(|summary| summary.iteration.is_some())
    );
}

#[test]
fn diagnostic_recorder_reuse_discards_state_from_a_failed_run() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig::default());

    let failed = AlternatingProjection::default()
        .iterations(2)
        .run_with_callbacks(
            &problem,
            vec![
                Box::new(recorder.clone()),
                Box::new(FailingIterationCallback),
            ],
        );
    assert!(matches!(failed, Err(Error::Unsupported(message)) if message.contains("intentional")));

    AlternatingProjection::default()
        .iterations(1)
        .run_with_callbacks(&problem, vec![Box::new(recorder.clone())])
        .unwrap();
    assert_eq!(recorder.diagnostics().iteration_history.len(), 1);
}

#[test]
fn runner_frame_summaries_ignore_masked_measurements() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    let mut corrupted = simulation.measurements.clone();
    for frame in 0..corrupted.frame_count() {
        corrupted.frame_mut(frame).unwrap()[0] = 1e12;
    }
    let mut mask = vec![1; corrupted.frame_len()];
    mask[0] = 0;
    let clean_problem = ReconstructionProblem::new(
        simulation.measurements.with_masks(mask.clone()).unwrap(),
        simulation.reconstruction_model.clone(),
    )
    .unwrap();
    let corrupted_problem = ReconstructionProblem::new(
        corrupted.with_masks(mask).unwrap(),
        simulation.reconstruction_model,
    )
    .unwrap();
    let config = DiagnosticRecorderConfig {
        record_frame_summaries: true,
        ..DiagnosticRecorderConfig::default()
    };
    let clean_recorder = DiagnosticRecorder::new(config.clone());
    let corrupted_recorder = DiagnosticRecorder::new(config);

    AlternatingProjection::default()
        .iterations(1)
        .run_with_callbacks(&clean_problem, vec![Box::new(clean_recorder.clone())])
        .unwrap();
    AlternatingProjection::default()
        .iterations(1)
        .run_with_callbacks(
            &corrupted_problem,
            vec![Box::new(corrupted_recorder.clone())],
        )
        .unwrap();

    let clean = clean_recorder.diagnostics();
    let corrupted = corrupted_recorder.diagnostics();
    for (clean, corrupted) in clean
        .frame_diagnostics
        .iter()
        .zip(&corrupted.frame_diagnostics)
    {
        assert_abs_diff_eq!(
            clean.metrics.reference_sum,
            corrupted.metrics.reference_sum,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            clean.metrics.residual_l2,
            corrupted.metrics.residual_l2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            clean.metrics.normalized_l2,
            corrupted.metrics.normalized_l2,
            epsilon = 1e-12
        );
    }
}
