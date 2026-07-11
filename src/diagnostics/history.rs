use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationRecord {
    pub iteration: usize,
    pub loss: f64,
    pub elapsed_seconds: f64,
    /// RMS ADMM consensus residual `A x - z`; absent for other algorithms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admm_primal_residual_rms: Option<f64>,
    /// RMS ADMM dual residual `rho (z_k - z_{k-1})`; absent for other algorithms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admm_dual_residual_rms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconstructionHistory {
    pub iterations: Vec<IterationRecord>,
}

impl ReconstructionHistory {
    pub fn final_loss(&self) -> Option<f64> {
        self.iterations.last().map(|record| record.loss)
    }
}
