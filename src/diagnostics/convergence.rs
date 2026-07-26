//! Iteration-level convergence records assembled by diagnostic recorders.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationDiagnostics {
    pub iteration: usize,

    pub total_objective: Option<f64>,
    pub data_objective: Option<f64>,
    pub regularization_objective: Option<f64>,

    pub object_relative_change: Option<f64>,
    pub pupil_relative_change: Option<f64>,

    pub median_frame_objective: Option<f64>,
    pub worst_frame_objective: Option<f64>,

    pub elapsed_seconds: Option<f64>,
}
