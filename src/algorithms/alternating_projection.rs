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
pub struct AlternatingProjection {
    pub iterations: usize,
    pub object_step: f64,
    pub batch_size: usize,
    pub epsilon: f64,
    pub loss_type: LossType,
}

impl Default for AlternatingProjection {
    fn default() -> Self {
        Self {
            iterations: 50,
            object_step: 1.0,
            batch_size: 1,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
        }
    }
}

impl AlternatingProjection {
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn object_step(mut self, object_step: f64) -> Self {
        self.object_step = object_step;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl ReconstructionAlgorithm for AlternatingProjection {
    fn validate(&self) -> Result<()> {
        if !self.object_step.is_finite() || self.object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        if self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                reason: "must be greater than zero".into(),
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
                object_denominator: ObjectDenominator::Local,
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
