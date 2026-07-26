//! Algorithm-neutral execution history for every reconstruction.

use serde::{Deserialize, Serialize};

/// Universal information recorded after one complete reconstruction iteration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IterationRecord {
    /// One-based iteration number.
    pub iteration: usize,
    /// Universal scalar optimized or reported by the algorithm.
    pub objective: f64,
    /// Seconds elapsed from the start of this reconstruction, including any
    /// elapsed time restored from a checkpoint.
    pub elapsed_seconds: f64,
}

/// One scalar emitted by an algorithm-specific iteration metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmMetricRecord {
    /// One-based iteration number.
    pub iteration: usize,
    /// Stable algorithm-owned namespace, for example `"admm"`.
    pub namespace: String,
    /// Stable metric name within `namespace`.
    pub metric: String,
    pub value: f64,
}

/// Execution trace recorded for every reconstruction, independently of
/// diagnostic callbacks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionTrace {
    pub iterations: Vec<IterationRecord>,
    pub algorithm_metrics: Vec<AlgorithmMetricRecord>,
}

impl ReconstructionTrace {
    pub fn final_objective(&self) -> Option<f64> {
        self.iterations.last().map(|record| record.objective)
    }
}
