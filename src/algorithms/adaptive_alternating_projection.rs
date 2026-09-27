use std::sync::Arc;

use crate::{
    Result,
    algorithms::{AlgorithmIterationMetrics, StepOutput, objective::LossType},
    backend::Backend,
    error::Error,
    measurements::MeasurementRead,
    reconstruction::{
        AdaptiveAlternatingProjectionAuxiliaryState, AlgorithmAuxiliaryState, Batch,
        ReconstructionProblem, ReconstructionState,
    },
};

use super::{
    ReconstructionAlgorithm,
    common::{ObjectDenominator, UpdateConfiguration, projection_update},
};

/// Effective object step used by one adaptive-projection iteration.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveAlternatingProjectionIterationMetrics {
    object_step: Option<f64>,
}

impl AdaptiveAlternatingProjectionIterationMetrics {
    /// Returns the object step shared by every frame update in this iteration.
    pub const fn object_step(&self) -> Option<f64> {
        self.object_step
    }
}

impl AlgorithmIterationMetrics for AdaptiveAlternatingProjectionIterationMetrics {
    fn merge(&mut self, other: Self) {
        if self.object_step.is_none() {
            self.object_step = other.object_step;
        }
    }

    fn append_records(
        &self,
        iteration: usize,
        output: &mut Vec<crate::reconstruction::AlgorithmMetricRecord>,
    ) {
        if let Some(value) = self.object_step {
            output.push(crate::reconstruction::AlgorithmMetricRecord {
                iteration,
                namespace: "adaptive_alternating_projection".into(),
                metric: "object_step".into(),
                value,
            });
        }
    }
}

/// Noise-robust alternating projection with a pass-adaptive object step.
///
/// # Method
///
/// The per-frame update is the same fixed-pupil amplitude projection used by
/// [`super::AlternatingProjection`]. One relaxation factor is shared by every
/// frame in a complete acquisition-schedule pass. The algorithm accumulates
/// the mask-aware, frame-weighted amplitude-MSE objective already evaluated by
/// those projections. After two completed passes establish consecutive
/// objectives, it retains the step when relative progress is greater than
/// `progress_threshold`; otherwise it multiplies the step by
/// `reduction_factor`, without going below `minimum_object_step`.
///
/// This feedback rule does not retry or roll back an iteration. Its objective
/// is the inexpensive incremental approximation described by Zuo et al., not
/// an additional exact full-data evaluation. The current step, preceding
/// objective, partial objective sums, and controller parameters are stored in
/// [`ReconstructionState`] so a matching checkpoint resumes exactly. Batching
/// cannot change the numerical path, while changing the acquisition schedule
/// intentionally can.
///
/// # Assumptions and limitations
///
/// The algorithm recovers only the object and keeps the compiled pupil fixed.
/// Its feedback objective is always amplitude MSE so a reporting option cannot
/// silently change controller behavior. It cannot be nested in physical joint
/// calibration because recompiling the forward model changes the meaning of
/// its objective history. The convergence analysis in the cited work assumes
/// convex component objectives; Fourier-ptychographic phase retrieval is
/// non-convex, so the adaptive rule is a practical robustness strategy rather
/// than a global-convergence guarantee.
///
/// # References
///
/// [C. Zuo, J. Sun, and Q. Chen, “Adaptive step-size strategy for noise-robust
/// Fourier ptychographic microscopy”
/// (2016)](https://doi.org/10.1364/OE.24.020724), *Optics Express* **24**(18),
/// 20724–20744.
#[derive(Clone, Debug)]
pub struct AdaptiveAlternatingProjection {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Object relaxation used until the feedback rule first reduces it.
    pub initial_object_step: f64,
    /// Minimum relative objective decrease required to retain the current step.
    pub progress_threshold: f64,
    /// Multiplicative step reduction used when progress is insufficient.
    pub reduction_factor: f64,
    /// Positive lower bound on the adaptive object step.
    pub minimum_object_step: f64,
    /// Number of measured frames supplied to each reconstruction step.
    pub batch_size: usize,
    /// Positive numerical floor used in projection divisions and relative progress.
    pub epsilon: f64,
}

impl Default for AdaptiveAlternatingProjection {
    fn default() -> Self {
        Self {
            iterations: 50,
            initial_object_step: 1.0,
            progress_threshold: 0.01,
            reduction_factor: 0.5,
            minimum_object_step: 0.001,
            batch_size: 1,
            epsilon: 1e-10,
        }
    }
}

impl AdaptiveAlternatingProjection {
    /// Sets the number of complete acquisition-schedule passes; validation requires non-zero.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the finite positive initial object-projection relaxation.
    pub fn initial_object_step(mut self, initial_object_step: f64) -> Self {
        self.initial_object_step = initial_object_step;
        self
    }

    /// Sets the required relative progress; validation requires `[0, 1)`.
    pub fn progress_threshold(mut self, progress_threshold: f64) -> Self {
        self.progress_threshold = progress_threshold;
        self
    }

    /// Sets the multiplicative reduction; validation requires `(0, 1)`.
    pub fn reduction_factor(mut self, reduction_factor: f64) -> Self {
        self.reduction_factor = reduction_factor;
        self
    }

    /// Sets the finite positive step floor, no greater than the initial step.
    pub fn minimum_object_step(mut self, minimum_object_step: f64) -> Self {
        self.minimum_object_step = minimum_object_step;
        self
    }

    /// Sets the positive number of acquisition frames supplied per step.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the finite positive numerical floor used by projection and feedback.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    fn new_auxiliary(
        &self,
        active_iteration: usize,
    ) -> AdaptiveAlternatingProjectionAuxiliaryState {
        AdaptiveAlternatingProjectionAuxiliaryState {
            active_iteration,
            current_object_step: self.initial_object_step,
            previous_objective: None,
            objective_sum: 0.0,
            weight_sum: 0.0,
            frames_accumulated: 0,
            initial_object_step: self.initial_object_step,
            progress_threshold: self.progress_threshold,
            reduction_factor: self.reduction_factor,
            minimum_object_step: self.minimum_object_step,
            epsilon: self.epsilon,
        }
    }

    fn prepare_auxiliary(
        &self,
        state: &mut ReconstructionState,
        active_iteration: usize,
    ) -> Result<()> {
        if state.algorithm_auxiliary.is_none() {
            state.algorithm_auxiliary =
                Some(AlgorithmAuxiliaryState::AdaptiveAlternatingProjection(
                    self.new_auxiliary(active_iteration),
                ));
            return Ok(());
        }
        let auxiliary = match state.algorithm_auxiliary.as_ref() {
            Some(AlgorithmAuxiliaryState::AdaptiveAlternatingProjection(auxiliary)) => auxiliary,
            Some(_) => {
                return Err(Error::InvalidModel(
                    "adaptive alternating projection cannot resume auxiliary state owned by another algorithm"
                        .into(),
                ));
            }
            None => unreachable!("missing state was initialized above"),
        };
        for (name, current, stored) in [
            (
                "initial_object_step",
                self.initial_object_step,
                auxiliary.initial_object_step,
            ),
            (
                "progress_threshold",
                self.progress_threshold,
                auxiliary.progress_threshold,
            ),
            (
                "reduction_factor",
                self.reduction_factor,
                auxiliary.reduction_factor,
            ),
            (
                "minimum_object_step",
                self.minimum_object_step,
                auxiliary.minimum_object_step,
            ),
            ("epsilon", self.epsilon, auxiliary.epsilon),
        ] {
            if current.to_bits() != stored.to_bits() {
                return Err(Error::InvalidParameter {
                    name,
                    reason: format!(
                        "value {current} differs from checkpointed adaptive-projection value {stored}"
                    ),
                });
            }
        }
        Ok(())
    }

    fn begin_iteration<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        iteration: usize,
    ) -> Result<f64> {
        self.prepare_auxiliary(state, iteration)?;
        let auxiliary = match state.algorithm_auxiliary.as_mut() {
            Some(AlgorithmAuxiliaryState::AdaptiveAlternatingProjection(auxiliary)) => auxiliary,
            _ => unreachable!("adaptive auxiliary was prepared above"),
        };
        if iteration == auxiliary.active_iteration {
            return Ok(auxiliary.current_object_step);
        }
        if iteration != auxiliary.active_iteration.saturating_add(1) {
            return Err(Error::InvalidModel(format!(
                "adaptive alternating projection expected iteration {} or {}, got {iteration}",
                auxiliary.active_iteration,
                auxiliary.active_iteration.saturating_add(1)
            )));
        }
        if auxiliary.frames_accumulated != problem.model.frame_count() {
            return Err(Error::InvalidModel(format!(
                "adaptive alternating projection accumulated {} of {} frames in iteration {}",
                auxiliary.frames_accumulated,
                problem.model.frame_count(),
                auxiliary.active_iteration
            )));
        }
        if !auxiliary.objective_sum.is_finite()
            || !auxiliary.weight_sum.is_finite()
            || auxiliary.weight_sum <= 0.0
        {
            return Err(Error::Numerical(
                "adaptive alternating projection has a non-finite or empty pass objective".into(),
            ));
        }
        let objective = auxiliary.objective_sum / auxiliary.weight_sum;
        if !objective.is_finite() || objective < 0.0 {
            return Err(Error::Numerical(
                "adaptive alternating projection produced an invalid pass objective".into(),
            ));
        }
        if let Some(previous) = auxiliary.previous_objective {
            let relative_progress = (previous - objective) / previous.max(auxiliary.epsilon);
            if !relative_progress.is_finite() {
                return Err(Error::Numerical(
                    "adaptive alternating projection produced non-finite relative progress".into(),
                ));
            }
            if relative_progress <= auxiliary.progress_threshold {
                auxiliary.current_object_step = (auxiliary.current_object_step
                    * auxiliary.reduction_factor)
                    .max(auxiliary.minimum_object_step);
            }
        }
        auxiliary.previous_objective = Some(objective);
        auxiliary.objective_sum = 0.0;
        auxiliary.weight_sum = 0.0;
        auxiliary.frames_accumulated = 0;
        auxiliary.active_iteration = iteration;
        Ok(auxiliary.current_object_step)
    }

    fn accumulate_batch(
        &self,
        state: &mut ReconstructionState,
        output: &StepOutput<AdaptiveAlternatingProjectionIterationMetrics>,
    ) -> Result<()> {
        if !output.summary.objective_sum.is_finite()
            || !output.summary.weight_sum.is_finite()
            || output.summary.weight_sum < 0.0
        {
            return Err(Error::Numerical(
                "adaptive alternating projection produced a non-finite batch objective".into(),
            ));
        }
        let auxiliary = match state.algorithm_auxiliary.as_mut() {
            Some(AlgorithmAuxiliaryState::AdaptiveAlternatingProjection(auxiliary)) => auxiliary,
            _ => {
                return Err(Error::InvalidModel(
                    "adaptive alternating projection auxiliary state is missing".into(),
                ));
            }
        };
        auxiliary.objective_sum += output.summary.objective_sum;
        auxiliary.weight_sum += output.summary.weight_sum;
        auxiliary.frames_accumulated = auxiliary
            .frames_accumulated
            .checked_add(output.summary.frame_count)
            .ok_or_else(|| {
                Error::Numerical("adaptive alternating projection frame counter overflowed".into())
            })?;
        if !auxiliary.objective_sum.is_finite() || !auxiliary.weight_sum.is_finite() {
            return Err(Error::Numerical(
                "adaptive alternating projection pass objective overflowed".into(),
            ));
        }
        Ok(())
    }
}

impl ReconstructionAlgorithm for AdaptiveAlternatingProjection {
    type IterationMetrics = AdaptiveAlternatingProjectionIterationMetrics;

    fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "iterations",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.initial_object_step.is_finite() || self.initial_object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "initial_object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.progress_threshold.is_finite() || !(0.0..1.0).contains(&self.progress_threshold) {
            return Err(Error::InvalidParameter {
                name: "progress_threshold",
                reason: "must be finite, at least zero, and less than one".into(),
            });
        }
        if !self.reduction_factor.is_finite()
            || !(0.0..1.0).contains(&self.reduction_factor)
            || self.reduction_factor == 0.0
        {
            return Err(Error::InvalidParameter {
                name: "reduction_factor",
                reason: "must be finite, greater than zero, and less than one".into(),
            });
        }
        if !self.minimum_object_step.is_finite() || self.minimum_object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "minimum_object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if self.minimum_object_step > self.initial_object_step {
            return Err(Error::InvalidParameter {
                name: "minimum_object_step",
                reason: "must not exceed initial_object_step".into(),
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
        self.prepare_auxiliary(&mut state, 0)?;
        Ok(state)
    }

    fn initialize_with_backend<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        backend: Arc<dyn Backend>,
    ) -> Result<ReconstructionState> {
        let mut state = ReconstructionState::initialize_with_backend(problem, backend)?;
        self.prepare_auxiliary(&mut state, 0)?;
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
        iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        let object_step = self.begin_iteration(problem, state, iteration)?;
        let summary = projection_update(
            problem,
            state,
            batch,
            UpdateConfiguration {
                object_step,
                pupil_step: None,
                epsilon: self.epsilon,
                loss_type: LossType::AmplitudeMse,
                object_denominator: ObjectDenominator::Local,
                constrain_pupil: true,
                gain_update: None,
                background_update: None,
                momentum: None,
            },
        )?;
        if state
            .object_spectrum()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::Numerical(
                "adaptive alternating projection produced a non-finite object spectrum".into(),
            ));
        }
        let output = StepOutput {
            summary,
            metrics: AdaptiveAlternatingProjectionIterationMetrics {
                object_step: Some(object_step),
            },
        };
        self.accumulate_batch(state, &output)?;
        Ok(output)
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}
