//! Iterative reconstruction algorithms for compiled image-plane models.
//!
//! Choose a concrete solver such as [`crate::algorithms::AlternatingProjection`],
//! [`crate::algorithms::Fpie`], [`crate::algorithms::Epry`],
//! [`crate::algorithms::Admm`], or [`crate::algorithms::GradientDescent`]. All
//! implement [`crate::algorithms::ReconstructionAlgorithm`] and consume a validated
//! [`crate::reconstruction::ReconstructionProblem`]; illumination geometry is compiled
//! beforehand into the problem's [`crate::model::ImagePlaneModel`].

mod admm;
mod alternating_projection;
mod common;
mod epry;
mod fpie;
mod gauge;
mod gradient_descent;
mod joint_reconstruction;
mod metrics;
pub mod objective;
mod regularization;

pub use admm::{Admm, AdmmIterationMetrics};
pub use alternating_projection::AlternatingProjection;
pub use epry::Epry;
pub use fpie::Fpie;
pub use gradient_descent::GradientDescent;
pub use joint_reconstruction::{
    JointIterationMetrics, JointReconstruction, JointReconstructionResult,
};
pub use metrics::{AlgorithmIterationMetrics, NoIterationMetrics, StepOutput, StepSummary};

use crate::{
    Result,
    backend::Backend,
    callbacks::Callback,
    measurements::MeasurementRead,
    reconstruction::{
        Batch, ReconstructionCheckpoint, ReconstructionProblem, ReconstructionResult,
        ReconstructionState, RunOptions, Runner,
    },
};
use std::sync::Arc;

/// Contract implemented by iterative image-plane reconstruction solvers.
///
/// Implementors validate their configuration, initialize a [`ReconstructionState`],
/// and update one scheduled [`Batch`] at a time. The trait's convenience methods own
/// the algorithm, borrow the problem for the duration of the run, and return an owned
/// [`ReconstructionResult`].
pub trait ReconstructionAlgorithm {
    /// Algorithm-specific metrics emitted by each step and appended to the trace.
    type IterationMetrics: AlgorithmIterationMetrics;

    /// Validates solver parameters independently of a reconstruction problem.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Validates solver requirements that depend on `problem`.
    fn validate_problem<M: MeasurementRead>(
        &self,
        _problem: &ReconstructionProblem<M>,
    ) -> Result<()> {
        Ok(())
    }

    /// Creates the default CPU-backed state for `problem`.
    fn initialize<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionState> {
        ReconstructionState::initialize(problem)
    }

    /// Creates reconstruction state using the supplied execution `backend`.
    fn initialize_with_backend<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        backend: Arc<dyn Backend>,
    ) -> Result<ReconstructionState> {
        ReconstructionState::initialize_with_backend(problem, backend)
    }

    /// Projects algorithm-owned ambiguities into a stable reported convention.
    ///
    /// [`Runner`] calls this after initialization or checkpoint restoration and
    /// after every completed iteration, before iteration diagnostics and
    /// callbacks, checkpoints, and final result construction. The default
    /// implementation leaves state unchanged. Algorithms that jointly recover
    /// coupled fields can override it without requiring external implementations
    /// to add a lifecycle method. Implementations must preserve the represented
    /// forward prediction and make repeated projection numerically idempotent.
    fn canonicalize_state<M: MeasurementRead>(
        &self,
        _problem: &ReconstructionProblem<M>,
        _state: &mut ReconstructionState,
    ) -> Result<()> {
        Ok(())
    }

    /// Updates `state` for one scheduled batch in zero-based `iteration`.
    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>>;

    /// Returns the requested number of complete schedule passes.
    fn iterations(&self) -> usize;

    /// Returns the number of measured frames combined into one step.
    fn batch_size(&self) -> usize {
        1
    }

    /// Runs the algorithm with default sequential scheduling and no callbacks.
    fn run<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionResult>
    where
        Self: Sized,
    {
        let options = RunOptions {
            max_iterations: self.iterations(),
            batch_size: self.batch_size(),
            ..RunOptions::default()
        };
        Runner::new(self, options).run(problem)
    }

    /// Runs the algorithm and invokes `callbacks` at their declared hooks.
    fn run_with_callbacks<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
        callbacks: Vec<Box<dyn Callback>>,
    ) -> Result<ReconstructionResult>
    where
        Self: Sized,
    {
        let options = RunOptions {
            max_iterations: self.iterations(),
            batch_size: self.batch_size(),
            ..RunOptions::default()
        };
        Runner::new(self, options)
            .with_callbacks(callbacks)
            .run(problem)
    }

    /// Resumes a run from a checkpoint after validating it against `problem`.
    fn run_from_checkpoint<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
        checkpoint: ReconstructionCheckpoint,
    ) -> Result<ReconstructionResult>
    where
        Self: Sized,
    {
        let options = RunOptions {
            max_iterations: self.iterations(),
            batch_size: self.batch_size(),
            ..RunOptions::default()
        };
        Runner::new(self, options)
            .resume_from(checkpoint)
            .run(problem)
    }
}
