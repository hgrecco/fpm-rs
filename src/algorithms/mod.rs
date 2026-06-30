mod admm;
mod alternating_projection;
mod common;
mod epry;
mod fpie;
mod gradient_descent;
mod regularization;

pub use admm::Admm;
pub use alternating_projection::AlternatingProjection;
pub use epry::Epry;
pub use fpie::Fpie;
pub use gradient_descent::GradientDescent;

use crate::{
    Result,
    backend::Backend,
    callbacks::Callback,
    diagnostics::StepDiagnostics,
    measurements::MeasurementRead,
    reconstruction::{
        Batch, ReconstructionCheckpoint, ReconstructionProblem, ReconstructionResult,
        ReconstructionState, RunOptions, Runner,
    },
};
use std::sync::Arc;

pub trait ReconstructionAlgorithm {
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    fn validate_problem<M: MeasurementRead>(
        &self,
        _problem: &ReconstructionProblem<M>,
    ) -> Result<()> {
        Ok(())
    }

    fn initialize<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionState> {
        ReconstructionState::initialize(problem)
    }

    fn initialize_with_backend<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        backend: Arc<dyn Backend>,
    ) -> Result<ReconstructionState> {
        ReconstructionState::initialize_with_backend(problem, backend)
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepDiagnostics>;

    fn iterations(&self) -> usize;

    fn batch_size(&self) -> usize {
        1
    }

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
