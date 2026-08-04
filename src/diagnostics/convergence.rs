//! Iteration-level convergence records assembled by diagnostic recorders.

use serde::{Deserialize, Serialize};

/// Optional convergence scalars recorded for one completed iteration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationDiagnostics {
    /// One-based completed iteration number.
    pub iteration: usize,

    /// Full reported objective, when available.
    pub total_objective: Option<f64>,
    /// Data-fidelity contribution to the objective, when separated by the solver.
    pub data_objective: Option<f64>,
    /// Regularization contribution to the objective, when separated by the solver.
    pub regularization_objective: Option<f64>,

    /// Relative change in the complex object field since the previous snapshot.
    pub object_relative_change: Option<f64>,
    /// Relative change in the complex pupil since the previous snapshot.
    pub pupil_relative_change: Option<f64>,

    /// Median acquisition-frame objective for this iteration.
    pub median_frame_objective: Option<f64>,
    /// Maximum acquisition-frame objective for this iteration.
    pub worst_frame_objective: Option<f64>,

    /// Wall-clock seconds elapsed since reconstruction start.
    pub elapsed_seconds: Option<f64>,
}
