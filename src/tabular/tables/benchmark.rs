use polars::prelude::*;

use crate::{
    Result,
    benchmark::{BenchmarkFrameRecord, BenchmarkRecord},
};

pub fn benchmark_runs_dataframe(records: &[BenchmarkRecord]) -> Result<DataFrame> {
    let length = records.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut case_ids = Vec::with_capacity(length);
    let mut algorithms = Vec::with_capacity(length);
    let mut algorithm_configurations = Vec::with_capacity(length);
    let mut dataset_names = Vec::with_capacity(length);
    let mut dataset_versions = Vec::with_capacity(length);
    let mut random_seeds = Vec::with_capacity(length);
    let mut frame_counts = Vec::with_capacity(length);
    let mut completed_iterations = Vec::with_capacity(length);
    let mut elapsed_seconds = Vec::with_capacity(length);
    let mut final_objectives = Vec::with_capacity(length);
    let mut successes = Vec::with_capacity(length);
    let mut errors = Vec::with_capacity(length);
    let mut repetitions = Vec::with_capacity(length);
    let mut warmups = Vec::with_capacity(length);
    let mut groups = Vec::with_capacity(length);
    let mut reference_run_ids = Vec::with_capacity(length);
    for record in records {
        run_ids.push(record.run_id.as_str());
        case_ids.push(record.case_id.as_str());
        algorithms.push(record.algorithm.as_str());
        algorithm_configurations.push(record.algorithm_configuration.as_str());
        dataset_names.push(record.dataset_name.as_str());
        dataset_versions.push(record.dataset_version.as_deref());
        random_seeds.push(record.random_seed);
        frame_counts.push(record.frame_count as u64);
        completed_iterations.push(record.success.then_some(record.completed_iterations as u64));
        elapsed_seconds.push(record.elapsed_seconds);
        final_objectives.push(record.success.then_some(record.final_objective).flatten());
        successes.push(record.success);
        errors.push(
            (!record.success)
                .then_some(record.error.as_deref())
                .flatten(),
        );
        repetitions.push(
            record
                .metadata
                .get("repetition")
                .and_then(|value| value.parse::<u64>().ok()),
        );
        warmups.push(
            record
                .metadata
                .get("warmup")
                .and_then(|value| value.parse::<bool>().ok()),
        );
        groups.push(record.metadata.get("benchmark_group").map(String::as_str));
        reference_run_ids.push(record.metadata.get("reference_run_id").map(String::as_str));
    }
    Ok(df!(
        "run_id" => run_ids,
        "case_id" => case_ids,
        "algorithm" => algorithms,
        "algorithm_configuration" => algorithm_configurations,
        "dataset_name" => dataset_names,
        "dataset_version" => dataset_versions,
        "random_seed" => random_seeds,
        "frame_count" => frame_counts,
        "completed_iterations" => completed_iterations,
        "elapsed_seconds" => elapsed_seconds,
        "final_objective" => final_objectives,
        "success" => successes,
        "error" => errors,
        "repetition" => repetitions,
        "warmup" => warmups,
        "benchmark_group" => groups,
        "reference_run_id" => reference_run_ids,
    )?)
}

pub fn benchmark_frames_dataframe(records: &[BenchmarkRecord]) -> Result<DataFrame> {
    let length = records.iter().map(|record| record.frames.len()).sum();
    let mut run_ids = Vec::with_capacity(length);
    let mut frame_indices = Vec::with_capacity(length);
    let mut original_frame_indices = Vec::with_capacity(length);
    let mut original_illumination_indices = Vec::with_capacity(length);
    let mut normalized_l2 = Vec::with_capacity(length);
    for record in records {
        for BenchmarkFrameRecord {
            frame_index,
            original_frame_index,
            original_illumination_index,
            normalized_l2: residual,
        } in &record.frames
        {
            run_ids.push(record.run_id.as_str());
            frame_indices.push(*frame_index as u64);
            original_frame_indices.push(*original_frame_index as u64);
            original_illumination_indices
                .push(original_illumination_index.map(|value| value as u64));
            normalized_l2.push(*residual);
        }
    }
    Ok(df!(
        "run_id" => run_ids,
        "frame_index" => frame_indices,
        "original_frame_index" => original_frame_indices,
        "original_illumination_index" => original_illumination_indices,
        "normalized_l2" => normalized_l2,
    )?)
}

pub fn benchmark_artifacts_dataframe(
    records: &[BenchmarkRecord],
    relative_result_paths: &std::collections::BTreeMap<String, String>,
) -> Result<DataFrame> {
    let length = records.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut roles = Vec::with_capacity(length);
    let mut relative_paths = Vec::with_capacity(length);
    for record in records {
        if let Some(path) = relative_result_paths.get(&record.run_id) {
            run_ids.push(record.run_id.as_str());
            roles.push("result_bundle");
            relative_paths.push(path.as_str());
        }
    }
    Ok(df!(
        "run_id" => run_ids,
        "role" => roles,
        "relative_path" => relative_paths,
    )?)
}

pub fn benchmark_metadata_dataframe(records: &[BenchmarkRecord]) -> Result<DataFrame> {
    const PROMOTED: &[&str] = &[
        "repetition",
        "warmup",
        "benchmark_group",
        "reference_run_id",
    ];
    let length = records
        .iter()
        .map(|record| record.metadata.len())
        .sum::<usize>();
    let mut run_ids = Vec::with_capacity(length);
    let mut keys = Vec::with_capacity(length);
    let mut values = Vec::with_capacity(length);
    for record in records {
        for (key, value) in &record.metadata {
            if PROMOTED.contains(&key.as_str()) {
                continue;
            }
            run_ids.push(record.run_id.as_str());
            keys.push(key.as_str());
            values.push(value.as_str());
        }
    }
    Ok(df!(
        "run_id" => run_ids,
        "key" => keys,
        "value" => values,
    )?)
}
