use polars::prelude::*;

use crate::{Result, reconstruction::ReconstructionResult};

pub const COMMON_RUN_COLUMNS: &[&str] = &[
    "run_id",
    "case_id",
    "algorithm",
    "algorithm_configuration",
    "dataset_name",
    "dataset_version",
    "random_seed",
    "frame_count",
    "completed_iterations",
    "elapsed_seconds",
    "final_objective",
    "success",
    "error",
];

pub fn summary_dataframe(run_id: &str, result: &ReconstructionResult) -> Result<DataFrame> {
    let (reconstruction_height, reconstruction_width) = result.object.dim();
    let (image_height, image_width) = result.recovered_pupil.shape();
    let metadata = &result.metadata;
    let optional = |key: &str| metadata.get(key).map(String::as_str);
    let parse_u64 = |key: &str| metadata.get(key).and_then(|value| value.parse().ok());
    let frame_count = parse_u64("frame_count").or_else(|| {
        result
            .recovered_frame_gains
            .as_ref()
            .map(|values| values.len() as u64)
    });
    Ok(df!(
        "run_id" => [run_id],
        "case_id" => [optional("case_id")],
        "crate_version" => [env!("CARGO_PKG_VERSION")],
        "dataset_name" => [optional("dataset_name")],
        "dataset_version" => [optional("dataset_version")],
        "preset_name" => [optional("preset_name")],
        "algorithm" => [result.runtime.algorithm.as_str()],
        "algorithm_configuration" => [optional("algorithm_configuration")],
        "random_seed" => [parse_u64("random_seed")],
        "frame_count" => [frame_count],
        "image_width" => [image_width as u64],
        "image_height" => [image_height as u64],
        "reconstruction_width" => [reconstruction_width as u64],
        "reconstruction_height" => [reconstruction_height as u64],
        "completed_iterations" => [result.runtime.completed_iterations as u64],
        "elapsed_seconds" => [result.runtime.elapsed_seconds],
        "stopped_early" => [result.runtime.stopped_early],
        "final_objective" => [result.trace.final_objective()],
        "success" => [true],
        "error" => [None::<&str>],
    )?)
}
