use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationRecord {
    pub iteration: usize,
    pub loss: f64,
    pub elapsed_seconds: f64,
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
