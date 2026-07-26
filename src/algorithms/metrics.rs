//! Typed algorithm-step output and deterministic metric reduction.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::reconstruction::AlgorithmMetricRecord;

/// Universal, algorithm-neutral information produced by one batch step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepSummary {
    /// Sum of per-frame objectives multiplied by frame weights.
    pub objective_sum: f64,
    /// Number of frames visited, including zero-weight frames.
    pub frame_count: usize,
    pub weight_sum: f64,
    pub per_frame_objective: BTreeMap<usize, f64>,
}

impl StepSummary {
    pub fn push_frame(&mut self, frame: usize, objective: f64, weight: f64) {
        self.objective_sum += weight * objective;
        self.frame_count += 1;
        self.weight_sum += weight;
        self.per_frame_objective.insert(frame, objective);
    }

    pub fn mean_objective(&self) -> Option<f64> {
        (self.weight_sum > 0.0).then(|| self.objective_sum / self.weight_sum)
    }

    pub fn merge(&mut self, other: Self) {
        self.objective_sum += other.objective_sum;
        self.frame_count += other.frame_count;
        self.weight_sum += other.weight_sum;
        self.per_frame_objective.extend(other.per_frame_objective);
    }
}

/// Algorithm-owned iteration metrics.
///
/// Implementations define both deterministic batch merging and conversion to
/// generic persisted records. This deliberately avoids a central algorithm or
/// metric enum.
pub trait AlgorithmIterationMetrics: Default {
    fn merge(&mut self, other: Self);

    fn append_records(&self, iteration: usize, output: &mut Vec<AlgorithmMetricRecord>);
}

/// Metrics type for algorithms that have no stable algorithm-specific scalar.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoIterationMetrics;

impl AlgorithmIterationMetrics for NoIterationMetrics {
    fn merge(&mut self, _other: Self) {}

    fn append_records(&self, _iteration: usize, _output: &mut Vec<AlgorithmMetricRecord>) {}
}

/// Typed output from one algorithm batch.
#[derive(Clone, Debug)]
pub struct StepOutput<M> {
    pub summary: StepSummary,
    pub metrics: M,
}

impl<M: Default> From<StepSummary> for StepOutput<M> {
    fn from(summary: StepSummary) -> Self {
        Self {
            summary,
            metrics: M::default(),
        }
    }
}

impl<M: AlgorithmIterationMetrics> Default for StepOutput<M> {
    fn default() -> Self {
        Self {
            summary: StepSummary::default(),
            metrics: M::default(),
        }
    }
}

impl<M: AlgorithmIterationMetrics> StepOutput<M> {
    pub fn merge(&mut self, other: Self) {
        self.summary.merge(other.summary);
        self.metrics.merge(other.metrics);
    }
}
