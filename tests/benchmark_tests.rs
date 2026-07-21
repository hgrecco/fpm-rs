use std::fs;

use fpm_rs::{
    Array2, Complex64, Result,
    algorithms::{AlternatingProjection, Epry, ReconstructionAlgorithm},
    benchmark::{
        BENCHMARK_PROFILES, BENCHMARK_RECORD_FORMAT_VERSION, CPU_BENCHMARK_PROFILE,
        SMOKE_BENCHMARK_PROFILE, annotate_benchmark_profile, benchmark_profile, run_benchmark_case,
        save_benchmark_outputs, write_benchmark_csv, write_benchmark_json,
    },
    diagnostics::{DiagnosticRecorder, DiagnosticRecorderConfig},
    evaluation::evaluate_reconstruction,
    reconstruction::ReconstructionProblem,
    simulation::presets::{
        ABERRATED_PUPIL_PRESET, NOISELESS_MIXED_PRESET, POISSON_GAUSSIAN_PRESET,
        aberrated_pupil_fpm, noiseless_mixed_fpm, poisson_gaussian_fpm,
    },
};

#[test]
fn named_simulation_presets_are_deterministic_and_well_formed() -> Result<()> {
    let first = noiseless_mixed_fpm(17)?;
    let second = noiseless_mixed_fpm(17)?;
    assert_eq!(first.parameters.frame_count, 9);
    assert_eq!(first.parameters.image_shape, (32, 32));
    assert_eq!(first.ground_truth_object.shape(), (64, 64));
    assert_eq!(
        first.measurements.as_slice(),
        second.measurements.as_slice()
    );

    let aberrated = aberrated_pupil_fpm(17)?;
    assert_eq!(aberrated.measurements.frame_count(), 9);
    assert_ne!(
        aberrated.true_model.pupil.values.as_slice(),
        aberrated.reconstruction_model.pupil.values.as_slice()
    );

    let noisy_first = poisson_gaussian_fpm(17)?;
    let noisy_second = poisson_gaussian_fpm(17)?;
    let noisy_other_seed = poisson_gaussian_fpm(18)?;
    assert_eq!(
        noisy_first.measurements.as_slice(),
        noisy_second.measurements.as_slice()
    );
    assert_ne!(
        noisy_first.measurements.as_slice(),
        noisy_other_seed.measurements.as_slice()
    );
    assert!(noisy_first.camera.is_some());
    Ok(())
}

#[test]
fn benchmark_profiles_have_stable_names_and_metadata() {
    let names = BENCHMARK_PROFILES
        .iter()
        .map(|profile| profile.name)
        .collect::<Vec<_>>();
    assert_eq!(names, [SMOKE_BENCHMARK_PROFILE, CPU_BENCHMARK_PROFILE]);
    let smoke = benchmark_profile(SMOKE_BENCHMARK_PROFILE).unwrap();
    assert_eq!(smoke.output_directory, "target/benchmark-results/smoke");
    assert!(smoke.algorithms.contains(&"AlternatingProjection"));
    assert!(smoke.algorithms.contains(&"GradientDescent"));
    assert!(benchmark_profile("unknown").is_none());
}

#[test]
fn benchmark_runs_ap_and_epry_on_the_same_problem() -> Result<()> {
    let simulation = noiseless_mixed_fpm(123)?;
    let truth = simulation.ground_truth_object;
    let true_model = simulation.true_model;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;

    let (mut ap, ap_result) = run_benchmark_case(
        "synthetic-noiseless",
        "iterations=4,object_step=1.0",
        AlternatingProjection::default().iterations(4),
        &problem,
        Some(&truth),
        Some(&true_model),
        None,
    );
    ap.preset_name = Some(NOISELESS_MIXED_PRESET.into());
    ap.random_seed = Some(123);
    annotate_benchmark_profile(&mut ap, benchmark_profile(SMOKE_BENCHMARK_PROFILE).unwrap());
    let (mut epry, epry_result) = run_benchmark_case(
        "synthetic-noiseless",
        "iterations=4,recover_pupil=true",
        Epry::default().iterations(4).recover_pupil(true),
        &problem,
        Some(&truth),
        Some(&true_model),
        None,
    );
    epry.preset_name = Some(NOISELESS_MIXED_PRESET.into());
    epry.random_seed = Some(123);

    for record in [&ap, &epry] {
        assert!(record.success, "{:?}", record.error);
        assert_eq!(record.format_version, BENCHMARK_RECORD_FORMAT_VERSION);
        assert_eq!(record.frame_count, 9);
        assert_eq!(record.image_shape, [32, 32]);
        assert_eq!(record.reconstruction_shape, [64, 64]);
        assert_eq!(record.completed_iterations, 4);
        assert!(record.runtime_seconds.is_finite());
        assert!(record.final_loss.unwrap().is_finite());
        assert!(record.amplitude_rmse.unwrap().is_finite());
        assert!(record.phase_rmse.unwrap().is_finite());
        assert!(record.per_frame_residual_mean.unwrap().is_finite());
        assert_eq!(
            record.selected_original_frame_indices,
            (0..9).collect::<Vec<_>>()
        );
    }
    assert_eq!(
        ap.metadata.get("benchmark_profile").map(String::as_str),
        Some(SMOKE_BENCHMARK_PROFILE)
    );
    assert_eq!(
        ap.metadata
            .get("benchmark_profile_output_directory")
            .map(String::as_str),
        Some("target/benchmark-results/smoke")
    );
    assert!(ap_result.is_some());
    assert!(epry_result.is_some());
    Ok(())
}

#[test]
fn benchmark_preset_produces_regression_diagnostics() -> Result<()> {
    let simulation = noiseless_mixed_fpm(123)?;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig {
        every: 2,
        record_frame_summaries: true,
        record_coverage: true,
        ..DiagnosticRecorderConfig::default()
    });

    AlternatingProjection::default()
        .iterations(4)
        .run_with_callbacks(&problem, vec![Box::new(recorder.clone())])?;

    let diagnostics = recorder.diagnostics();
    assert_eq!(
        diagnostics
            .iteration_history
            .iter()
            .map(|entry| entry.iteration)
            .collect::<Vec<_>>(),
        [2, 4]
    );
    assert_eq!(
        diagnostics.frame_diagnostics.len(),
        2 * problem.model.frame_count()
    );
    assert!(diagnostics.raw_frame_stats.is_empty());
    let coverage = diagnostics.coverage.as_ref().unwrap();
    assert_eq!(
        coverage.pupil_centers_px.len(),
        problem.model.source_count()
    );
    assert_eq!(coverage.crop_indices.len(), problem.model.source_count());
    let losses = diagnostics
        .iteration_history
        .iter()
        .map(|entry| entry.total_loss.unwrap())
        .collect::<Vec<_>>();
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(losses[1] < losses[0]);
    assert!(
        diagnostics
            .frame_diagnostics
            .iter()
            .all(|entry| entry.metrics.normalized_l2.is_finite())
    );
    Ok(())
}

#[test]
fn benchmark_writes_outputs_csv_json_and_failure_records() -> Result<()> {
    let simulation = noiseless_mixed_fpm(99)?;
    let truth = simulation.ground_truth_object;
    let true_model = simulation.true_model;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let (mut successful, result) = run_benchmark_case(
        "synthetic,\"noiseless\"",
        "iterations=2",
        AlternatingProjection::default().iterations(2),
        &problem,
        Some(&truth),
        Some(&true_model),
        None,
    );
    successful.preset_name = Some(NOISELESS_MIXED_PRESET.into());
    successful.random_seed = Some(99);
    let directory = tempfile::tempdir()?;
    save_benchmark_outputs(&mut successful, result.as_ref().unwrap(), directory.path())?;
    assert_eq!(successful.output_paths.len(), 4);
    assert!(successful.output_paths.iter().all(|path| path.is_file()));
    let mut different_configuration = successful.clone();
    different_configuration.algorithm_configuration = "iterations=3".into();
    different_configuration.output_paths.clear();
    save_benchmark_outputs(
        &mut different_configuration,
        result.as_ref().unwrap(),
        directory.path(),
    )?;
    assert_ne!(
        successful.output_paths,
        different_configuration.output_paths
    );

    let (failed, failed_result) = run_benchmark_case(
        "synthetic,\"noiseless\"",
        "object_step=-1",
        AlternatingProjection::default().object_step(-1.0),
        &problem,
        Some(&truth),
        Some(&true_model),
        None,
    );
    assert!(!failed.success);
    assert!(failed.error.as_deref().unwrap().contains("object_step"));
    assert!(failed_result.is_none());

    let records = [successful, failed];
    let csv_path = directory.path().join("summary.csv");
    let json_path = directory.path().join("summary.json");
    write_benchmark_csv(&records, &csv_path)?;
    write_benchmark_json(&records, &json_path)?;
    let csv = fs::read_to_string(csv_path)?;
    assert!(csv.contains("final_to_initial_loss_ratio"));
    let mut reader = csv::Reader::from_reader(csv.as_bytes());
    let first = reader.records().next().unwrap().unwrap();
    assert_eq!(&first[1], "synthetic,\"noiseless\"");
    let json: serde_json::Value = serde_json::from_slice(&fs::read(json_path)?)?;
    assert_eq!(json["format_version"], BENCHMARK_RECORD_FORMAT_VERSION);
    assert_eq!(json["records"].as_array().unwrap().len(), 2);
    assert_eq!(json["records"][0]["success"], true);
    assert_eq!(json["records"][1]["success"], false);
    Ok(())
}

#[test]
fn preset_names_are_versioned() {
    for name in [
        NOISELESS_MIXED_PRESET,
        ABERRATED_PUPIL_PRESET,
        POISSON_GAUSSIAN_PRESET,
    ] {
        assert!(name.ends_with("_v1"));
    }
}

#[test]
fn object_mask_controls_ground_truth_metrics_and_benchmark_evaluation() -> Result<()> {
    let simulation = noiseless_mixed_fpm(71)?;
    let truth = simulation.ground_truth_object;
    let true_model = simulation.true_model;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let mut mask_values = vec![0_u8; 64 * 64];
    for row in 16..48 {
        for column in 16..48 {
            mask_values[row * 64 + column] = 1;
        }
    }
    let mask = Array2::from_vec((64, 64), mask_values)?;
    let (record, result) = run_benchmark_case(
        "masked-synthetic",
        "iterations=2",
        AlternatingProjection::default().iterations(2),
        &problem,
        Some(&truth),
        Some(&true_model),
        Some(&mask),
    );
    assert!(record.success, "{:?}", record.error);
    let result = result.unwrap();
    let expected = evaluate_reconstruction(&result, &truth, None, Some(&mask))?;
    assert!((record.amplitude_rmse.unwrap() - expected.object.amplitude_rmse).abs() < 1e-12);

    let mut changed_outside = result;
    for (index, value) in changed_outside.object.as_mut_slice().iter_mut().enumerate() {
        if mask.as_slice()[index] == 0 {
            *value = Complex64::new(1e6, -1e6);
        }
    }
    let unchanged = evaluate_reconstruction(&changed_outside, &truth, None, Some(&mask))?;
    assert!((unchanged.object.amplitude_rmse - expected.object.amplitude_rmse).abs() < 1e-12);
    assert!((unchanged.object.phase_rmse - expected.object.phase_rmse).abs() < 1e-12);
    assert!((unchanged.object.complex_nrmse - expected.object.complex_nrmse).abs() < 1e-12);
    assert!((unchanged.object.fourier_nrmse - expected.object.fourier_nrmse).abs() < 1e-12);

    let empty_mask = Array2::filled((64, 64), 0_u8)?;
    assert!(evaluate_reconstruction(&changed_outside, &truth, None, Some(&empty_mask)).is_err());
    Ok(())
}
