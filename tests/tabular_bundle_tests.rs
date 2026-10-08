#![cfg(feature = "parquet")]

mod common;

use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
    sync::Arc,
};

use fpm_rs::{
    Error, Result,
    algorithms::{Admm, ReconstructionAlgorithm},
    benchmark::BenchmarkRecord,
    benchmark_bundle::{
        BenchmarkBundleExportOptions, read_benchmark_bundle, write_benchmark_bundle,
    },
    diagnostics::{FourierCoverageDiagnostics, IterationDiagnostics, ReconstructionDiagnostics},
    reconstruction::{BundleExportOptions, ReconstructionProblem, ReconstructionResult},
    simulation::{Simulator, SyntheticObject},
    tabular::{
        COMMON_RUN_COLUMNS, algorithm_metrics_dataframe, benchmark_frames_dataframe,
        benchmark_runs_dataframe, history_dataframe, metadata_dataframe,
        scalar_diagnostics_dataframe, summary_dataframe,
    },
};
use polars::prelude::DataFrame;

fn column_names(dataframe: &DataFrame) -> Vec<&str> {
    dataframe
        .get_column_names()
        .into_iter()
        .map(|name| name.as_str())
        .collect()
}

fn reconstruction() -> Result<ReconstructionResult> {
    let model = common::direct_model()?;
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::mixed_test_pattern((16, 16))?)
        .simulate()?;
    let frame_count = simulation.measurements.frame_count();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let mut result = Admm::default().iterations(3).run(&problem)?;
    result.metadata.insert("case_id".into(), "case-a".into());
    result
        .metadata
        .insert("dataset_name".into(), "synthetic".into());
    result.metadata.insert("dataset_version".into(), "1".into());
    result
        .metadata
        .insert("algorithm_configuration".into(), "iterations=3".into());
    result.metadata.insert("random_seed".into(), "17".into());
    result
        .metadata
        .insert("frame_count".into(), frame_count.to_string());
    result.metadata.insert("operator".into(), "test".into());
    result.scalar_diagnostics.insert("custom_score".into(), 0.5);
    result.calibrated_illumination = Some(vec![(0.25, -0.5), (0.0, 0.125)]);
    result.recovered_frame_gains = Some(vec![1.0; frame_count]);
    result.recovered_background = Some(vec![0.0; frame_count]);
    Ok(result)
}

#[test]
fn canonical_tables_have_stable_schemas_and_long_form_dynamic_values() -> Result<()> {
    let result = reconstruction()?;
    let history = history_dataframe("run-a", &result.trace)?;
    assert_eq!(
        column_names(&history),
        ["run_id", "iteration", "objective", "elapsed_seconds"]
    );
    assert_eq!(history.height(), 3);

    let metrics = algorithm_metrics_dataframe("run-a", &result.trace)?;
    assert_eq!(
        column_names(&metrics),
        ["run_id", "iteration", "namespace", "metric", "value"]
    );
    assert_eq!(metrics.height(), 6);
    assert!(
        metrics
            .column("namespace")?
            .str()?
            .iter()
            .flatten()
            .all(|value| value == "admm")
    );

    let scalar = scalar_diagnostics_dataframe("run-a", &result.scalar_diagnostics)?;
    assert_eq!(column_names(&scalar), ["run_id", "key", "value"]);
    assert_eq!(scalar.height(), 2);

    let metadata = metadata_dataframe("run-a", &result.metadata)?;
    assert_eq!(column_names(&metadata), ["run_id", "key", "value"]);
    assert_eq!(metadata.height(), 1);
    assert_eq!(metadata.column("key")?.str()?.get(0), Some("operator"));

    let summary = summary_dataframe("run-a", &result)?;
    let record = BenchmarkRecord::from_result("case-a", "synthetic", "iterations=3", &result);
    let runs = benchmark_runs_dataframe(std::slice::from_ref(&record))?;
    for column in ["crop_row", "crop_column", "crop_height", "crop_width"] {
        assert_eq!(runs.column(column)?.u64()?.get(0), None);
    }
    for &column in COMMON_RUN_COLUMNS {
        assert_eq!(
            summary.column(column)?.dtype(),
            runs.column(column)?.dtype(),
            "common run column {column} changed dtype"
        );
    }

    let mut failed = record.clone();
    failed.run_id = "failed-run".into();
    failed.success = false;
    failed.error = Some("deliberate failure".into());
    failed.final_objective = Some(123.0);
    let mut nan = record;
    nan.run_id = "nan-run".into();
    nan.final_objective = Some(f64::NAN);
    let runs = benchmark_runs_dataframe(&[failed.clone(), nan])?;
    assert_eq!(runs.column("completed_iterations")?.u64()?.get(0), None);
    assert_eq!(runs.column("final_objective")?.f64()?.get(0), None);
    assert!(
        runs.column("final_objective")?
            .f64()?
            .get(1)
            .is_some_and(f64::is_nan)
    );
    assert_eq!(
        runs.column("error")?.str()?.get(0),
        Some("deliberate failure")
    );
    let frames = benchmark_frames_dataframe(&[failed])?;
    assert_eq!(
        frames.column("frame_index")?.dtype(),
        &polars::prelude::DataType::UInt64
    );
    assert_eq!(frames.column("frame_index")?.u64()?.get(0), Some(0));
    Ok(())
}

#[test]
fn result_bundle_is_lazy_cached_immutable_on_disk_and_uniquely_finalized() -> Result<()> {
    let result = reconstruction()?;
    let directory = tempfile::tempdir()?;
    let requested = directory.path().join("result");
    let first = result.write_bundle(
        &requested,
        BundleExportOptions {
            run_id: Some("run-a".into()),
            label: Some("first".into()),
            include_previews: true,
        },
    )?;
    let second = result.write_bundle(
        &requested,
        BundleExportOptions {
            run_id: Some("run-b".into()),
            label: None,
            include_previews: false,
        },
    )?;

    assert_eq!(first.path, requested);
    assert_ne!(first.path, second.path);
    assert!(first.manifest_path.is_file());
    assert!(!first.path.join("run-state.json").exists());
    assert!(!first.path.join("checkpoints").exists());
    assert!(first.previews.object_amplitude.is_some());
    assert!(second.previews.object_amplitude.is_none());
    assert!(!directory.path().join("result.inprogress").exists());

    let object_one = first.object()?;
    let object_two = first.object()?;
    assert!(Arc::ptr_eq(&object_one, &object_two));
    let loaded_result = first.result()?;
    assert_eq!(loaded_result.object, result.object);
    assert_eq!(first.verify()?.artifact_count, 18);
    assert!(first.tables.illumination_calibration.is_some());
    assert!(first.tables.frame_calibration.is_some());
    assert!(first.arrays.illumination_calibration.is_some());
    assert!(first.arrays.frame_gains.is_some());
    assert!(first.arrays.background.is_some());
    assert_eq!(
        loaded_result.calibrated_illumination,
        result.calibrated_illumination
    );
    assert_eq!(
        loaded_result.recovered_frame_gains,
        result.recovered_frame_gains
    );
    assert_eq!(
        loaded_result.recovered_background,
        result.recovered_background
    );
    assert_eq!(loaded_result.metadata, result.metadata);

    first.clear_cache();
    let object_three = first.object()?;
    assert!(!Arc::ptr_eq(&object_one, &object_three));
    assert_eq!(*object_one, *object_three);
    Ok(())
}

#[test]
fn result_bundle_rejects_incomplete_corrupt_and_invalid_manifests() -> Result<()> {
    let result = reconstruction()?;
    let directory = tempfile::tempdir()?;
    let bundle = result.write_bundle(
        directory.path().join("result"),
        BundleExportOptions {
            run_id: Some("run-corrupt".into()),
            ..BundleExportOptions::default()
        },
    )?;
    let manifest_bytes = std::fs::read(&bundle.manifest_path)?;

    let mut manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    manifest["artifacts"][0]["role"] = serde_json::json!("tables.unknown");
    std::fs::write(&bundle.manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    assert!(matches!(
        fpm_rs::read_bundle(&bundle.path),
        Err(Error::UnsupportedArtifactRole(_))
    ));

    let mut manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    manifest["artifacts"][0]["relative_path"] = serde_json::json!("../escape.parquet");
    std::fs::write(&bundle.manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    assert!(matches!(
        fpm_rs::read_bundle(&bundle.path),
        Err(Error::InvalidRelativePath(_))
    ));

    let mut manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    let object = manifest["artifacts"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|artifact| artifact["role"] == "arrays.object")
        .unwrap();
    object["dtype"] = serde_json::json!("<f8");
    std::fs::write(&bundle.manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    assert!(matches!(
        fpm_rs::read_bundle(&bundle.path),
        Err(Error::InvalidArrayDtype { .. })
    ));

    std::fs::write(&bundle.manifest_path, &manifest_bytes)?;
    let reopened = fpm_rs::read_bundle(&bundle.path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&reopened.arrays.object.path)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(b"X")?;
    file.sync_all()?;
    assert!(matches!(
        reopened.object(),
        Err(Error::ArtifactHashMismatch { .. })
    ));

    let incomplete = directory.path().join("unfinished.inprogress");
    std::fs::create_dir(&incomplete)?;
    assert!(matches!(
        fpm_rs::read_bundle(incomplete),
        Err(Error::IncompleteBundle(_))
    ));
    Ok(())
}

#[test]
fn result_bundle_round_trips_optional_diagnostics_and_coverage_preview() -> Result<()> {
    let result = reconstruction()?;
    let diagnostics = ReconstructionDiagnostics {
        iteration_diagnostics: vec![IterationDiagnostics {
            iteration: 2,
            total_objective: result
                .trace
                .iterations
                .get(1)
                .map(|record| record.objective),
            data_objective: None,
            regularization_objective: None,
            object_relative_change: Some(0.1),
            pupil_relative_change: None,
            median_frame_objective: None,
            worst_frame_objective: None,
            elapsed_seconds: result
                .trace
                .iterations
                .get(1)
                .map(|record| record.elapsed_seconds),
        }],
        coverage: Some(FourierCoverageDiagnostics {
            synthetic_na: Some(0.2),
            pupil_radius_px: 2.0,
            pupil_centers_px: vec![[8.0, 8.0]],
            illumination_na: vec![0.1],
            crop_indices: Vec::new(),
            overlap_shape: Some([16, 16]),
        }),
        ..ReconstructionDiagnostics::default()
    };
    let directory = tempfile::tempdir()?;
    let bundle = result.write_bundle_with_context(
        directory.path().join("diagnostics"),
        BundleExportOptions {
            run_id: Some("run-diagnostics".into()),
            include_previews: true,
            ..BundleExportOptions::default()
        },
        Some(&diagnostics),
        None,
    )?;

    assert!(bundle.tables.iteration_diagnostics.is_some());
    assert!(bundle.previews.fourier_coverage.is_some());
    let first = bundle.diagnostics()?.expect("diagnostics artifact");
    let second = bundle.diagnostics()?.expect("diagnostics artifact");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.iteration_diagnostics[0].iteration, 2);
    bundle.verify()?;

    let mut invalid_diagnostics = diagnostics;
    invalid_diagnostics
        .coverage
        .as_mut()
        .expect("coverage")
        .pupil_radius_px = -1.0;
    let failed_path = directory.path().join("failed");
    assert!(
        result
            .write_bundle_with_context(
                &failed_path,
                BundleExportOptions {
                    include_previews: true,
                    ..BundleExportOptions::default()
                },
                Some(&invalid_diagnostics),
                None,
            )
            .is_err()
    );
    let retained = directory.path().join("failed.inprogress");
    assert!(retained.join("run-state.json").is_file());
    assert!(retained.join("checkpoints").is_dir());
    assert!(!failed_path.exists());
    Ok(())
}

#[test]
fn benchmark_bundle_normalizes_runs_and_reuses_nested_result_bundles() -> Result<()> {
    let result = reconstruction()?;
    let first = BenchmarkRecord::from_result("same-case", "synthetic", "iterations=3", &result);
    let second = BenchmarkRecord::from_result("same-case", "synthetic", "iterations=3", &result);
    assert_eq!(first.case_id, second.case_id);
    assert_ne!(first.run_id, second.run_id);

    let records = vec![first.clone(), second.clone()];
    let results = records
        .iter()
        .map(|record| (record.run_id.clone(), result.clone()))
        .collect::<BTreeMap<_, _>>();
    let directory = tempfile::tempdir()?;
    let written = write_benchmark_bundle(
        directory.path().join("benchmark"),
        "comparison",
        &records,
        &results,
        BenchmarkBundleExportOptions {
            label: Some("two repeats".into()),
        },
    )?;
    assert!(written.tables.runs.path.is_file());
    assert!(written.tables.frames.path.is_file());
    assert_eq!(written.results.len(), 2);
    assert!(!written.path.join("arrays").exists());
    for record in &records {
        let nested = &written.results[&record.run_id];
        assert_eq!(nested.run_id, record.run_id);
        assert_eq!(nested.result()?.trace.iterations.len(), 3);
    }

    let reopened = read_benchmark_bundle(&written.path)?;
    assert_eq!(reopened.name, "comparison");
    assert_eq!(reopened.label.as_deref(), Some("two repeats"));
    assert_eq!(reopened.results.len(), 2);

    let mut duplicate_records = records.clone();
    duplicate_records[1].run_id = duplicate_records[0].run_id.clone();
    assert!(matches!(
        write_benchmark_bundle(
            directory.path().join("duplicate"),
            "invalid",
            &duplicate_records,
            &results,
            BenchmarkBundleExportOptions::default(),
        ),
        Err(Error::InvalidParameter {
            name: "benchmark records",
            ..
        })
    ));
    assert!(matches!(
        write_benchmark_bundle(
            directory.path().join("missing-result"),
            "invalid",
            &records,
            &BTreeMap::new(),
            BenchmarkBundleExportOptions::default(),
        ),
        Err(Error::InvalidParameter {
            name: "benchmark results",
            ..
        })
    ));

    let manifest_bytes = std::fs::read(&written.manifest_path)?;
    let mut manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    manifest["benchmark_bundle_format_version"] = serde_json::json!(999);
    std::fs::write(
        &written.manifest_path,
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    assert!(matches!(
        read_benchmark_bundle(&written.path),
        Err(Error::UnsupportedBundleVersion { .. })
    ));
    std::fs::write(&written.manifest_path, manifest_bytes)?;

    let nested_path = written.results[&records[0].run_id].path.clone();
    let displaced = directory.path().join("displaced-result");
    std::fs::rename(&nested_path, &displaced)?;
    assert!(matches!(
        read_benchmark_bundle(&written.path),
        Err(Error::MissingArtifact { .. })
    ));
    Ok(())
}
