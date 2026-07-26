use polars::prelude::*;

use crate::{Result, reconstruction::ReconstructionResult};

pub fn illumination_calibration_dataframe(
    run_id: &str,
    result: &ReconstructionResult,
) -> Result<DataFrame> {
    let corrections = result.calibrated_illumination.as_deref().unwrap_or(&[]);
    let length = corrections.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut source_indices = Vec::with_capacity(length);
    let mut row_corrections = Vec::with_capacity(length);
    let mut column_corrections = Vec::with_capacity(length);
    for (source, &(row, column)) in corrections.iter().enumerate() {
        run_ids.push(run_id);
        source_indices.push(source as u64);
        row_corrections.push(row);
        column_corrections.push(column);
    }
    Ok(df!(
        "run_id" => run_ids,
        "source_index" => source_indices,
        "row_correction_pixels" => row_corrections,
        "column_correction_pixels" => column_corrections,
    )?)
}

pub fn frame_calibration_dataframe(
    run_id: &str,
    result: &ReconstructionResult,
) -> Result<DataFrame> {
    let length = result
        .recovered_frame_gains
        .as_ref()
        .map_or(0, Vec::len)
        .max(result.recovered_background.as_ref().map_or(0, Vec::len));
    let mut run_ids = Vec::with_capacity(length);
    let mut frame_indices = Vec::with_capacity(length);
    let mut gains = Vec::with_capacity(length);
    let mut backgrounds = Vec::with_capacity(length);
    for frame in 0..length {
        run_ids.push(run_id);
        frame_indices.push(frame as u64);
        gains.push(
            result
                .recovered_frame_gains
                .as_ref()
                .and_then(|values| values.get(frame).copied()),
        );
        backgrounds.push(
            result
                .recovered_background
                .as_ref()
                .and_then(|values| values.get(frame).copied()),
        );
    }
    Ok(df!(
        "run_id" => run_ids,
        "frame_index" => frame_indices,
        "gain" => gains,
        "background" => backgrounds,
    )?)
}
