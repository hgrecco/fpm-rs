use std::collections::BTreeMap;

use polars::prelude::*;

use crate::{Result, diagnostics::ReconstructionDiagnostics};

pub fn iteration_diagnostics_dataframe(
    run_id: &str,
    diagnostics: &ReconstructionDiagnostics,
) -> Result<DataFrame> {
    let length = diagnostics.iteration_diagnostics.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut iterations = Vec::with_capacity(length);
    let mut total_objectives = Vec::with_capacity(length);
    let mut data_objectives = Vec::with_capacity(length);
    let mut regularization_objectives = Vec::with_capacity(length);
    let mut object_changes = Vec::with_capacity(length);
    let mut pupil_changes = Vec::with_capacity(length);
    let mut median_frame_objectives = Vec::with_capacity(length);
    let mut worst_frame_objectives = Vec::with_capacity(length);
    let mut elapsed_seconds = Vec::with_capacity(length);
    for record in &diagnostics.iteration_diagnostics {
        run_ids.push(run_id);
        iterations.push(record.iteration as u64);
        total_objectives.push(record.total_objective);
        data_objectives.push(record.data_objective);
        regularization_objectives.push(record.regularization_objective);
        object_changes.push(record.object_relative_change);
        pupil_changes.push(record.pupil_relative_change);
        median_frame_objectives.push(record.median_frame_objective);
        worst_frame_objectives.push(record.worst_frame_objective);
        elapsed_seconds.push(record.elapsed_seconds);
    }
    Ok(df!(
        "run_id" => run_ids,
        "iteration" => iterations,
        "total_objective" => total_objectives,
        "data_objective" => data_objectives,
        "regularization_objective" => regularization_objectives,
        "object_relative_change" => object_changes,
        "pupil_relative_change" => pupil_changes,
        "median_frame_objective" => median_frame_objectives,
        "worst_frame_objective" => worst_frame_objectives,
        "elapsed_seconds" => elapsed_seconds,
    )?)
}

pub fn frame_diagnostics_dataframe(
    run_id: &str,
    diagnostics: &ReconstructionDiagnostics,
) -> Result<DataFrame> {
    let length = diagnostics.frame_diagnostics.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut iterations = Vec::with_capacity(length);
    let mut frame_indices = Vec::with_capacity(length);
    let mut illumination_indices = Vec::with_capacity(length);
    let mut reference_sums = Vec::with_capacity(length);
    let mut estimate_sums = Vec::with_capacity(length);
    let mut residual_l1 = Vec::with_capacity(length);
    let mut residual_l2 = Vec::with_capacity(length);
    let mut residual_mean = Vec::with_capacity(length);
    let mut residual_std = Vec::with_capacity(length);
    let mut residual_max_abs = Vec::with_capacity(length);
    let mut normalized_l2 = Vec::with_capacity(length);
    let mut saturated_pixels = Vec::with_capacity(length);
    for record in &diagnostics.frame_diagnostics {
        run_ids.push(run_id);
        iterations.push(record.iteration.map(|value| value as u64));
        frame_indices.push(record.frame_index as u64);
        illumination_indices.push(record.illumination_index as u64);
        reference_sums.push(record.metrics.reference_sum);
        estimate_sums.push(record.metrics.estimate_sum);
        residual_l1.push(record.metrics.residual_l1);
        residual_l2.push(record.metrics.residual_l2);
        residual_mean.push(record.metrics.residual_mean);
        residual_std.push(record.metrics.residual_std);
        residual_max_abs.push(record.metrics.residual_max_abs);
        normalized_l2.push(record.metrics.normalized_l2);
        saturated_pixels.push(record.metrics.saturated_pixels.map(|value| value as u64));
    }
    Ok(df!(
        "run_id" => run_ids,
        "iteration" => iterations,
        "frame_index" => frame_indices,
        "illumination_index" => illumination_indices,
        "reference_sum" => reference_sums,
        "estimate_sum" => estimate_sums,
        "residual_l1" => residual_l1,
        "residual_l2" => residual_l2,
        "residual_mean" => residual_mean,
        "residual_std" => residual_std,
        "residual_max_abs" => residual_max_abs,
        "normalized_l2" => normalized_l2,
        "saturated_pixels" => saturated_pixels,
    )?)
}

pub fn raw_frame_statistics_dataframe(
    run_id: &str,
    diagnostics: &ReconstructionDiagnostics,
) -> Result<DataFrame> {
    let length = diagnostics.raw_frame_statistics.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut frame_indices = Vec::with_capacity(length);
    let mut means = Vec::with_capacity(length);
    let mut stds = Vec::with_capacity(length);
    let mut minima = Vec::with_capacity(length);
    let mut maxima = Vec::with_capacity(length);
    let mut sums = Vec::with_capacity(length);
    let mut saturated = Vec::with_capacity(length);
    let mut zeros = Vec::with_capacity(length);
    for record in &diagnostics.raw_frame_statistics {
        run_ids.push(run_id);
        frame_indices.push(record.frame_index as u64);
        means.push(record.metrics.mean);
        stds.push(record.metrics.std);
        minima.push(record.metrics.min);
        maxima.push(record.metrics.max);
        sums.push(record.metrics.sum);
        saturated.push(record.metrics.saturated_pixels as u64);
        zeros.push(record.metrics.zero_pixels as u64);
    }
    Ok(df!(
        "run_id" => run_ids,
        "frame_index" => frame_indices,
        "mean" => means,
        "std" => stds,
        "min" => minima,
        "max" => maxima,
        "sum" => sums,
        "saturated_pixels" => saturated,
        "zero_pixels" => zeros,
    )?)
}

pub fn scalar_diagnostics_dataframe(
    run_id: &str,
    values: &BTreeMap<String, f64>,
) -> Result<DataFrame> {
    let length = values.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut keys = Vec::with_capacity(length);
    let mut scalar_values = Vec::with_capacity(length);
    for (key, &value) in values {
        run_ids.push(run_id);
        keys.push(key.as_str());
        scalar_values.push(value);
    }
    Ok(df!(
        "run_id" => run_ids,
        "key" => keys,
        "value" => scalar_values,
    )?)
}

pub fn metadata_dataframe(run_id: &str, values: &BTreeMap<String, String>) -> Result<DataFrame> {
    const PROMOTED: &[&str] = &[
        "case_id",
        "dataset_name",
        "dataset_version",
        "preset_name",
        "algorithm_configuration",
        "random_seed",
        "frame_count",
    ];
    let length = values.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut keys = Vec::with_capacity(length);
    let mut metadata_values = Vec::with_capacity(length);
    for (key, value) in values {
        if PROMOTED.contains(&key.as_str()) {
            continue;
        }
        run_ids.push(run_id);
        keys.push(key.as_str());
        metadata_values.push(value.as_str());
    }
    Ok(df!(
        "run_id" => run_ids,
        "key" => keys,
        "value" => metadata_values,
    )?)
}
