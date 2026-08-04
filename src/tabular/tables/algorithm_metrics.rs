use polars::prelude::*;

use crate::{Result, reconstruction::ReconstructionTrace};

/// Builds one row per algorithm-specific trace metric with stable identity columns.
pub fn algorithm_metrics_dataframe(run_id: &str, trace: &ReconstructionTrace) -> Result<DataFrame> {
    let length = trace.algorithm_metrics.len();
    let mut run_ids = Vec::with_capacity(length);
    let mut iterations = Vec::with_capacity(length);
    let mut namespaces = Vec::with_capacity(length);
    let mut metrics = Vec::with_capacity(length);
    let mut values = Vec::with_capacity(length);
    for record in &trace.algorithm_metrics {
        run_ids.push(run_id);
        iterations.push(record.iteration as u64);
        namespaces.push(record.namespace.as_str());
        metrics.push(record.metric.as_str());
        values.push(record.value);
    }
    Ok(df!(
        "run_id" => run_ids,
        "iteration" => iterations,
        "namespace" => namespaces,
        "metric" => metrics,
        "value" => values,
    )?)
}
