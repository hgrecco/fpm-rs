use crate::{
    Result,
    algorithms::objective::LossType,
    diagnostics::StepDiagnostics,
    error::Error,
    measurements::MeasurementRead,
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

use super::{
    ReconstructionAlgorithm,
    common::{ObjectDenominator, UpdateConfiguration, projection_update},
};

/// Embedded pupil-recovery reconstruction for Fourier ptychographic microscopy.
///
/// # Method
///
/// EPRY alternates detector-amplitude projection with two normalized updates:
/// the exit-wave error is divided by pupil power to update the overlapping
/// object-spectrum patch and by object-patch power to update the pupil. Jointly
/// recovering these two complex functions lets the pupil absorb aberrations
/// that would otherwise be imprinted on the reconstructed object. The pupil
/// can be projected back onto the known aperture support after every update.
///
/// This implementation also optionally estimates a relative gain and an
/// additive, spatially uniform background for each frame. Those calibration
/// updates and incoherent multiplexing support are crate extensions to the
/// reference EPRY method.
///
/// # Reference
///
/// X. Ou, G. Zheng, and C. Yang, “Embedded pupil function recovery for Fourier
/// ptychographic microscopy,” *Optics Express* **22**(5), 4960–4972 (2014),
/// [doi:10.1364/OE.22.004960](https://doi.org/10.1364/OE.22.004960).
#[derive(Clone, Debug)]
pub struct Epry {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Relaxation factor applied to each object-spectrum correction.
    pub object_step: f64,
    /// Relaxation factor applied to each pupil correction.
    pub pupil_step: f64,
    /// Number of measured frames supplied to each reconstruction step.
    pub batch_size: usize,
    /// Whether to update the complex pupil alongside the object.
    pub recover_pupil: bool,
    /// Whether to zero recovered pupil values outside the compiled aperture.
    pub constrain_pupil_support: bool,
    /// Whether to estimate one multiplicative intensity gain per frame.
    pub recover_frame_gains: bool,
    /// Fraction of each least-squares frame-gain estimate applied per update.
    pub gain_step: f64,
    /// Lower bound for recovered frame gains; must be positive.
    pub minimum_gain: f64,
    /// Upper bound for recovered frame gains.
    pub maximum_gain: f64,
    /// Whether to estimate one additive, spatially uniform background per frame.
    pub recover_background: bool,
    /// Fraction of the mean frame residual added to the background per update.
    pub background_step: f64,
    /// Lower bound for recovered background intensities.
    pub minimum_background: f64,
    /// Upper bound for recovered background intensities.
    pub maximum_background: f64,
    /// Positive numerical floor used in normalized updates.
    pub epsilon: f64,
    /// Loss used for diagnostics; the projection itself always enforces the
    /// measured amplitude.
    pub loss_type: LossType,
}

impl Default for Epry {
    fn default() -> Self {
        Self {
            iterations: 100,
            object_step: 0.8,
            pupil_step: 0.1,
            batch_size: 1,
            recover_pupil: true,
            constrain_pupil_support: true,
            recover_frame_gains: false,
            gain_step: 0.2,
            minimum_gain: 1e-6,
            maximum_gain: 1e6,
            recover_background: false,
            background_step: 0.2,
            minimum_background: 0.0,
            maximum_background: 1e12,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
        }
    }
}

impl Epry {
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn object_step(mut self, step: f64) -> Self {
        self.object_step = step;
        self
    }

    pub fn pupil_step(mut self, step: f64) -> Self {
        self.pupil_step = step;
        self
    }

    pub fn recover_pupil(mut self, recover: bool) -> Self {
        self.recover_pupil = recover;
        self
    }

    pub fn constrain_pupil_support(mut self, constrain: bool) -> Self {
        self.constrain_pupil_support = constrain;
        self
    }

    pub fn recover_frame_gains(mut self, recover: bool) -> Self {
        self.recover_frame_gains = recover;
        self
    }

    pub fn gain_step(mut self, step: f64) -> Self {
        self.gain_step = step;
        self
    }

    pub fn gain_bounds(mut self, minimum: f64, maximum: f64) -> Self {
        self.minimum_gain = minimum;
        self.maximum_gain = maximum;
        self
    }

    pub fn recover_background(mut self, recover: bool) -> Self {
        self.recover_background = recover;
        self
    }

    pub fn background_step(mut self, step: f64) -> Self {
        self.background_step = step;
        self
    }

    pub fn background_bounds(mut self, minimum: f64, maximum: f64) -> Self {
        self.minimum_background = minimum;
        self.maximum_background = maximum;
        self
    }
}

impl ReconstructionAlgorithm for Epry {
    fn validate(&self) -> Result<()> {
        if !self.object_step.is_finite() || self.object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.pupil_step.is_finite() || self.pupil_step < 0.0 {
            return Err(Error::InvalidParameter {
                name: "pupil_step",
                reason: "must be finite and non-negative".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 || self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "epsilon/batch_size",
                reason: "epsilon must be positive and batch size non-zero".into(),
            });
        }
        if !self.gain_step.is_finite() || !(0.0..=1.0).contains(&self.gain_step) {
            return Err(Error::InvalidParameter {
                name: "gain_step",
                reason: "must be finite and between zero and one".into(),
            });
        }
        if !self.minimum_gain.is_finite()
            || !self.maximum_gain.is_finite()
            || self.minimum_gain <= 0.0
            || self.maximum_gain <= self.minimum_gain
        {
            return Err(Error::InvalidParameter {
                name: "gain_bounds",
                reason: "must be finite, positive, and strictly increasing".into(),
            });
        }
        if !self.background_step.is_finite() || !(0.0..=1.0).contains(&self.background_step) {
            return Err(Error::InvalidParameter {
                name: "background_step",
                reason: "must be finite and between zero and one".into(),
            });
        }
        if !self.minimum_background.is_finite()
            || !self.maximum_background.is_finite()
            || self.maximum_background <= self.minimum_background
        {
            return Err(Error::InvalidParameter {
                name: "background_bounds",
                reason: "must be finite and strictly increasing".into(),
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
                pupil_step: self.recover_pupil.then_some(self.pupil_step),
                epsilon: self.epsilon,
                loss_type: self.loss_type,
                object_denominator: ObjectDenominator::Global,
                constrain_pupil: self.constrain_pupil_support,
                gain_update: self.recover_frame_gains.then_some(
                    super::common::GainUpdateConfiguration {
                        step: self.gain_step,
                        minimum: self.minimum_gain,
                        maximum: self.maximum_gain,
                    },
                ),
                background_update: self.recover_background.then_some(
                    super::common::BackgroundUpdateConfiguration {
                        step: self.background_step,
                        minimum: self.minimum_background,
                        maximum: self.maximum_background,
                    },
                ),
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
