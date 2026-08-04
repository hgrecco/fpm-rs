use polars::prelude::*;

use crate::{Result, evaluation::FrameIntensityEvaluation};

/// Builds one predicted-versus-measured intensity comparison row per acquisition frame.
pub fn frame_evaluation_dataframe(
    run_id: &str,
    evaluation: &FrameIntensityEvaluation,
) -> Result<DataFrame> {
    let length = evaluation.per_frame.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut frame_indices = Vec::with_capacity(length);
    let mut reference_sums = Vec::with_capacity(length);
    let mut estimate_sums = Vec::with_capacity(length);
    let mut residual_l1 = Vec::with_capacity(length);
    let mut residual_l2 = Vec::with_capacity(length);
    let mut residual_mean = Vec::with_capacity(length);
    let mut residual_std = Vec::with_capacity(length);
    let mut residual_max_abs = Vec::with_capacity(length);
    let mut normalized_l2 = Vec::with_capacity(length);
    let mut saturated_pixels = Vec::with_capacity(length);
    for (frame, metrics) in evaluation.per_frame.iter().enumerate() {
        run_ids.push(run_id);
        frame_indices.push(frame as u64);
        reference_sums.push(metrics.reference_sum);
        estimate_sums.push(metrics.estimate_sum);
        residual_l1.push(metrics.residual_l1);
        residual_l2.push(metrics.residual_l2);
        residual_mean.push(metrics.residual_mean);
        residual_std.push(metrics.residual_std);
        residual_max_abs.push(metrics.residual_max_abs);
        normalized_l2.push(metrics.normalized_l2);
        saturated_pixels.push(metrics.saturated_pixels.map(|value| value as u64));
    }
    Ok(df!(
        "run_id" => run_ids,
        "frame_index" => frame_indices,
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
