use crate::{
    Result,
    algorithms::{NoIterationMetrics, StepOutput, objective::LossType},
    error::Error,
    measurements::MeasurementRead,
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

use super::{
    ReconstructionAlgorithm,
    common::{ObjectDenominator, UpdateConfiguration, projection_update},
};

/// Alternating-projection reconstruction for Fourier ptychographic microscopy.
///
/// # Method
///
/// For each measured frame, the algorithm extracts the corresponding patch of
/// the current high-resolution object spectrum, multiplies it by the pupil, and
/// propagates the resulting field to the detector plane. It replaces the
/// predicted detector amplitude with the measured amplitude while retaining
/// the predicted phase, propagates the corrected field back to Fourier space,
/// and inserts the resulting correction into the object spectrum. Repeating
/// this operation over overlapping Fourier patches makes them converge toward
/// a mutually consistent complex object.
///
/// For incoherently multiplexed frames, one measured-to-predicted amplitude
/// ratio is applied jointly to every source mode before their corrections are
/// back-projected.
///
/// # Reference
///
/// G. Zheng, R. Horstmeyer, and C. Yang, “Wide-field, high-resolution Fourier
/// ptychographic microscopy,” *Nature Photonics* **7**, 739–745 (2013),
/// [doi:10.1038/nphoton.2013.187](https://doi.org/10.1038/nphoton.2013.187).
#[derive(Clone, Debug)]
pub struct AlternatingProjection {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Relaxation factor applied to each object-spectrum correction.
    pub object_step: f64,
    /// Number of measured frames supplied to each reconstruction step.
    pub batch_size: usize,
    /// Positive numerical floor used in divisions and dark-field handling.
    pub epsilon: f64,
    /// Loss used for diagnostics; the projection itself always enforces the
    /// measured amplitude.
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
    type IterationMetrics = NoIterationMetrics;

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
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        Ok(projection_update(
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
        )?
        .into())
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}
