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

/// Regularized ptychographic iterative-engine reconstruction adapted to FPM.
///
/// # Method
///
/// `Fpie` performs the same detector-amplitude projection as
/// [`super::AlternatingProjection`], but preconditions each object correction
/// with the rPIE denominator
/// `(1 - stability) * |pupil|^2 + stability * max(|pupil|^2)`. This blends
/// local inverse-pupil weighting with a global power bound, reducing unstable
/// updates where the pupil transfer is weak. The corrected low-resolution
/// spectrum is inserted into the corresponding overlapping patch of the
/// high-resolution object spectrum.
///
/// This crate adapts the rPIE update, originally formulated for scanned
/// ptychography, to image-plane Fourier ptychography.
///
/// # Reference
///
/// [A. Maiden, D. Johnson, and P. Li, “Further improvements to the
/// ptychographical iterative engine” (2017)](https://doi.org/10.1364/OPTICA.4.000736),
/// *Optica* **4**(7), 736–745.
#[derive(Clone, Debug)]
pub struct Fpie {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Relaxation factor applied to each object-spectrum correction.
    pub object_step: f64,
    /// Blend between local pupil power (`0`) and maximum pupil power (`1`) in
    /// the rPIE denominator.
    pub stability: f64,
    /// Number of measured frames supplied to each reconstruction step.
    pub batch_size: usize,
    /// Positive numerical floor added to the rPIE denominator.
    pub epsilon: f64,
    /// Loss used for diagnostics; the projection itself always enforces the
    /// measured amplitude.
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
    /// Sets the number of complete acquisition-schedule passes; validation requires non-zero.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the finite positive relaxation applied to object-spectrum corrections.
    pub fn object_step(mut self, step: f64) -> Self {
        self.object_step = step;
        self
    }

    /// Sets the rPIE pupil-power blend after clamping the value to `[0, 1]`.
    pub fn stability(mut self, stability: f64) -> Self {
        self.stability = stability.clamp(0.0, 1.0);
        self
    }
}

impl ReconstructionAlgorithm for Fpie {
    type IterationMetrics = NoIterationMetrics;

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
                object_denominator: ObjectDenominator::Rpie(self.stability),
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
