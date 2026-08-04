use crate::{
    Result,
    diagnostics::{DiagnosticRequest, Diagnostics},
    model::ImagePlaneModel,
    reconstruction::{
        AlgorithmMetricRecord, ReconstructionResult, ReconstructionState, ReconstructionTrace,
    },
};

/// Control decision returned by a callback hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue the current reconstruction.
    Continue,
    /// Stop cleanly after the current hook and return a partial result.
    Stop,
}

/// Point in runner execution at which a callback can be invoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackHook {
    /// Once after initialization and before the first scheduled frame.
    Start,
    /// After a processed acquisition frame when frame callbacks are enabled.
    FrameEnd,
    /// After every complete acquisition schedule pass.
    IterationEnd,
    /// Once after a [`crate::reconstruction::ReconstructionResult`] is assembled.
    Finish,
}

/// Borrowed reconstruction snapshot supplied to non-finish callback hooks.
pub struct StepContext<'a> {
    /// One-based current iteration, or zero at the start hook.
    pub iteration: usize,
    /// Current zero-based acquisition-frame index at frame-end hooks.
    pub frame_index: Option<usize>,
    /// Current zero-based batch index at frame-end hooks.
    pub batch_index: Option<usize>,
    /// Current mutable-run state exposed immutably to callbacks.
    pub state: &'a ReconstructionState,
    /// Diagnostics computed because active callbacks requested them.
    pub diagnostics: &'a Diagnostics,
    /// Trace of iterations completed before or at this hook.
    pub trace: &'a ReconstructionTrace,
    /// Algorithm-specific scalar metrics emitted for the current completed iteration.
    pub current_algorithm_metrics: &'a [AlgorithmMetricRecord],
    /// Compiled image-plane model used by the run.
    pub model: &'a ImagePlaneModel,
    /// Optional user-facing reconstruction-problem name.
    pub problem_name: Option<&'a str>,
}

/// Thread-sendable observer or early-stop controller for reconstruction execution.
///
/// Implementors declare diagnostics before hooks so the runner computes only requested
/// snapshots. Hooks receive borrowed state and cannot mutate numerical reconstruction data.
pub trait Callback: Send {
    /// Declares diagnostics required at every hook by legacy or non-periodic callbacks.
    fn requires(&self) -> Vec<DiagnosticRequest> {
        Vec::new()
    }

    /// Requests diagnostics for a specific hook. Built-in periodic callbacks
    /// override this to avoid computing snapshots on inactive iterations.
    fn requires_for(&self, _hook: CallbackHook, _iteration: usize) -> Vec<DiagnosticRequest> {
        self.requires()
    }

    /// Runs once after state initialization and may stop before processing frames.
    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    /// Runs once for every completed frame when frame callbacks are enabled.
    /// For multi-frame algorithms, all frames in a batch observe the same
    /// post-batch state while `frame_index` and natural loss remain per-frame.
    fn on_frame_end(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    /// Runs after a complete iteration and may request clean termination.
    fn on_iteration_end(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    /// Runs once with the final owned result; errors are propagated by the runner.
    fn on_finish(&mut self, _result: &ReconstructionResult) -> Result<()> {
        Ok(())
    }
}
