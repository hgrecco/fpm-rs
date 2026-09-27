use std::sync::Arc;

use num_complex::Complex64;

use crate::{
    Result,
    algorithms::{NoIterationMetrics, StepOutput, objective::LossType},
    backend::Backend,
    error::Error,
    measurements::MeasurementRead,
    reconstruction::{
        AlgorithmAuxiliaryState, Batch, MpieAuxiliaryState, ReconstructionProblem,
        ReconstructionState,
    },
};

use super::{
    ReconstructionAlgorithm,
    common::{MomentumConfiguration, ObjectDenominator, UpdateConfiguration, projection_update},
};

/// Momentum-accelerated regularized PIE adapted to image-plane FPM.
///
/// # Method
///
/// `Mpie` applies the same sequential, object-only rPIE projection as
/// [`super::Fpie`], then periodically accelerates the centered high-resolution
/// object spectrum. After `momentum_interval` positive-weight measured-frame
/// updates, it updates the complex velocity and object as
///
/// `V <- momentum_friction * V + (O_rpie - O_anchor)`
///
/// `O <- O_rpie + momentum_feedback * V`.
///
/// A multiplexed measurement counts once after all of its source modes are
/// inserted. Zero-weight frames do not advance the interval. The counter,
/// velocity, anchor, and defining parameters are checkpointed, so changing
/// `batch_size` does not change the numerical path and a matching checkpoint
/// resumes exactly.
///
/// # Adaptation and assumptions
///
/// The cited method was formulated and tested for scanned ptychography and
/// applies momentum to both object and probe. This implementation accelerates
/// only the fixed-pupil Fourier-ptychographic object spectrum. It separates the
/// paper's single `eta_obj` into friction and feedback controls; setting them
/// equal reproduces its Eqs. (19) and (21). `object_step` corresponds to the
/// paper's `gamma_obj` in Eq. (22). Physical joint calibration is unsupported
/// because recompiling the model would require an explicit rule for resetting
/// or transporting momentum.
///
/// # Reference
///
/// [A. Maiden, D. Johnson, and P. Li, “Further improvements to the
/// ptychographical iterative engine” (2017)](https://doi.org/10.1364/OPTICA.4.000736),
/// *Optica* **4**(7), 736–745.
#[derive(Clone, Debug)]
pub struct Mpie {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Relaxation factor applied to each rPIE object-spectrum correction.
    pub object_step: f64,
    /// Blend between local pupil power (`0`) and maximum pupil power (`1`) in
    /// the rPIE denominator.
    pub stability: f64,
    /// Positive-weight measured-frame updates between momentum events.
    pub momentum_interval: usize,
    /// Fraction of the previous velocity retained at each momentum event.
    pub momentum_friction: f64,
    /// Fraction of the updated velocity added to the object spectrum.
    pub momentum_feedback: f64,
    /// Number of measured frames supplied to each reconstruction step.
    pub batch_size: usize,
    /// Positive numerical floor added to the rPIE denominator.
    pub epsilon: f64,
    /// Loss used for diagnostics; the projection itself always enforces the
    /// measured amplitude.
    pub loss_type: LossType,
}

impl Default for Mpie {
    fn default() -> Self {
        Self {
            iterations: 50,
            object_step: 0.2,
            stability: 0.05,
            momentum_interval: 30,
            momentum_friction: 0.9,
            momentum_feedback: 0.9,
            batch_size: 1,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
        }
    }
}

impl Mpie {
    /// Sets the number of complete acquisition-schedule passes.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the finite positive relaxation applied to rPIE corrections.
    pub fn object_step(mut self, object_step: f64) -> Self {
        self.object_step = object_step;
        self
    }

    /// Sets the rPIE pupil-power blend; validation requires `[0, 1]`.
    pub fn stability(mut self, stability: f64) -> Self {
        self.stability = stability;
        self
    }

    /// Sets the positive number of effective frame updates per momentum event.
    pub fn momentum_interval(mut self, momentum_interval: usize) -> Self {
        self.momentum_interval = momentum_interval;
        self
    }

    /// Sets the retained-velocity fraction; validation requires `[0, 1)`.
    pub fn momentum_friction(mut self, momentum_friction: f64) -> Self {
        self.momentum_friction = momentum_friction;
        self
    }

    /// Sets the velocity-feedback fraction; validation requires `[0, 1]`.
    pub fn momentum_feedback(mut self, momentum_feedback: f64) -> Self {
        self.momentum_feedback = momentum_feedback;
        self
    }

    /// Sets the positive number of acquisition frames supplied per step.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the finite positive numerical floor used by the rPIE denominator.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets the loss reported by diagnostics.
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    fn auxiliary_from_state(&self, state: &ReconstructionState) -> MpieAuxiliaryState {
        MpieAuxiliaryState {
            velocity: vec![Complex64::default(); state.object_spectrum.len()],
            anchor: state.object_spectrum.as_slice().to_vec(),
            effective_frames_since_momentum: 0,
            object_step: self.object_step,
            stability: self.stability,
            epsilon: self.epsilon,
            momentum_interval: self.momentum_interval,
            momentum_friction: self.momentum_friction,
            momentum_feedback: self.momentum_feedback,
        }
    }

    fn prepare_auxiliary(&self, state: &mut ReconstructionState) -> Result<()> {
        if state.algorithm_auxiliary.is_none() {
            state.algorithm_auxiliary = Some(AlgorithmAuxiliaryState::Mpie(
                self.auxiliary_from_state(state),
            ));
            return Ok(());
        }
        let auxiliary = match state.algorithm_auxiliary.as_ref() {
            Some(AlgorithmAuxiliaryState::Mpie(auxiliary)) => auxiliary,
            Some(_) => {
                return Err(Error::InvalidModel(
                    "mPIE cannot resume auxiliary state owned by another algorithm".into(),
                ));
            }
            None => unreachable!("missing state was initialized above"),
        };
        if auxiliary.velocity.len() != state.object_spectrum.len()
            || auxiliary.anchor.len() != state.object_spectrum.len()
        {
            return Err(Error::InvalidModel(
                "mPIE auxiliary dimensions do not match the object spectrum".into(),
            ));
        }
        for (name, current, stored) in [
            ("object_step", self.object_step, auxiliary.object_step),
            ("stability", self.stability, auxiliary.stability),
            ("epsilon", self.epsilon, auxiliary.epsilon),
            (
                "momentum_friction",
                self.momentum_friction,
                auxiliary.momentum_friction,
            ),
            (
                "momentum_feedback",
                self.momentum_feedback,
                auxiliary.momentum_feedback,
            ),
        ] {
            if current.to_bits() != stored.to_bits() {
                return Err(Error::InvalidParameter {
                    name,
                    reason: format!(
                        "value {current} differs from checkpointed mPIE value {stored}"
                    ),
                });
            }
        }
        if self.momentum_interval != auxiliary.momentum_interval {
            return Err(Error::InvalidParameter {
                name: "momentum_interval",
                reason: format!(
                    "value {} differs from checkpointed mPIE value {}",
                    self.momentum_interval, auxiliary.momentum_interval
                ),
            });
        }
        Ok(())
    }
}

impl ReconstructionAlgorithm for Mpie {
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
        if self.momentum_interval == 0 {
            return Err(Error::InvalidParameter {
                name: "momentum_interval",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.momentum_friction.is_finite() || !(0.0..1.0).contains(&self.momentum_friction) {
            return Err(Error::InvalidParameter {
                name: "momentum_friction",
                reason: "must be finite, at least zero, and less than one".into(),
            });
        }
        if !self.momentum_feedback.is_finite() || !(0.0..=1.0).contains(&self.momentum_feedback) {
            return Err(Error::InvalidParameter {
                name: "momentum_feedback",
                reason: "must be finite and between zero and one".into(),
            });
        }
        if self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        Ok(())
    }

    fn initialize<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionState> {
        let mut state = ReconstructionState::initialize(problem)?;
        self.prepare_auxiliary(&mut state)?;
        Ok(state)
    }

    fn initialize_with_backend<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        backend: Arc<dyn Backend>,
    ) -> Result<ReconstructionState> {
        let mut state = ReconstructionState::initialize_with_backend(problem, backend)?;
        self.prepare_auxiliary(&mut state)?;
        Ok(state)
    }

    fn supports_joint_reconstruction(&self) -> bool {
        false
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        self.prepare_auxiliary(state)?;
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
                momentum: Some(MomentumConfiguration {
                    interval: self.momentum_interval,
                    friction: self.momentum_friction,
                    feedback: self.momentum_feedback,
                }),
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
