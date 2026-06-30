use crate::{
    Result,
    diagnostics::{LossType, StepDiagnostics},
    error::Error,
    measurements::MeasurementRead,
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

use super::{
    ReconstructionAlgorithm,
    common::{ObjectDenominator, UpdateConfiguration, projection_update},
};

#[derive(Clone, Debug)]
pub struct Fpie {
    pub iterations: usize,
    pub object_step: f64,
    pub stability: f64,
    pub batch_size: usize,
    pub epsilon: f64,
    pub loss_type: LossType,
}

impl Default for Fpie {
    fn default() -> Self {
        Self {
            iterations: 50,
            object_step: 0.8,
            stability: 0.1,
            batch_size: 1,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
        }
    }
}

impl Fpie {
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn object_step(mut self, step: f64) -> Self {
        self.object_step = step;
        self
    }

    pub fn stability(mut self, stability: f64) -> Self {
        self.stability = stability.clamp(0.0, 1.0);
        self
    }
}

impl ReconstructionAlgorithm for Fpie {
    fn validate(&self) -> Result<()> {
        if !self.object_step.is_finite() || self.object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.stability.is_finite() || !(0.0..=1.0).contains(&self.stability) {
            return Err(Error::InvalidParameter {
                name: "stability",
                reason: "must be finite and between zero and one".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 || self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "epsilon/batch_size",
                reason: "epsilon must be positive and batch size non-zero".into(),
            });
        }
        Ok(())
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepDiagnostics> {
        projection_update(
            problem,
            state,
            batch,
            UpdateConfiguration {
                object_step: self.object_step,
                pupil_step: None,
                epsilon: self.epsilon,
                loss_type: self.loss_type,
                object_denominator: ObjectDenominator::Rpie(self.stability),
                constrain_pupil: true,
                gain_update: None,
                background_update: None,
            },
        )
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}
