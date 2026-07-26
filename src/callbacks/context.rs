use crate::{
    Result,
    diagnostics::{DiagnosticRequest, Diagnostics},
    model::ImagePlaneModel,
    reconstruction::{
        AlgorithmMetricRecord, ReconstructionResult, ReconstructionState, ReconstructionTrace,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackAction {
    Continue,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackHook {
    Start,
    FrameEnd,
    IterationEnd,
    Finish,
}

pub struct StepContext<'a> {
    pub iteration: usize,
    pub frame_index: Option<usize>,
    pub batch_index: Option<usize>,
    pub state: &'a ReconstructionState,
    pub diagnostics: &'a Diagnostics,
    pub trace: &'a ReconstructionTrace,
    pub current_algorithm_metrics: &'a [AlgorithmMetricRecord],
    pub model: &'a ImagePlaneModel,
    pub problem_name: Option<&'a str>,
}

pub trait Callback: Send {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        Vec::new()
    }

    /// Requests diagnostics for a specific hook. Built-in periodic callbacks
    /// override this to avoid computing snapshots on inactive iterations.
    fn requires_for(&self, _hook: CallbackHook, _iteration: usize) -> Vec<DiagnosticRequest> {
        self.requires()
    }

    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    /// Runs once for every completed frame when frame callbacks are enabled.
    /// For multi-frame algorithms, all frames in a batch observe the same
    /// post-batch state while `frame_index` and natural loss remain per-frame.
    fn on_frame_end(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    fn on_finish(&mut self, _result: &ReconstructionResult) -> Result<()> {
        Ok(())
    }
}
