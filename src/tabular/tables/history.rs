use polars::prelude::*;

use crate::{Result, reconstruction::ReconstructionTrace};

/// Builds one run-ID/objective/elapsed-time row per completed iteration.
pub fn history_dataframe(run_id: &str, trace: &ReconstructionTrace) -> Result<DataFrame> {
    let length = trace.iterations.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut iterations = Vec::with_capacity(length);
    let mut objectives = Vec::with_capacity(length);
    let mut elapsed_seconds = Vec::with_capacity(length);
    for record in &trace.iterations {
        run_ids.push(run_id);
        iterations.push(record.iteration as u64);
        objectives.push(record.objective);
        elapsed_seconds.push(record.elapsed_seconds);
    }
    Ok(df!(
        "run_id" => run_ids,
        "iteration" => iterations,
        "objective" => objectives,
        "elapsed_seconds" => elapsed_seconds,
    )?)
}
