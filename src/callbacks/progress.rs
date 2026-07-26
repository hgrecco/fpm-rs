use crate::{Result, diagnostics::DiagnosticRequest};

use super::{Callback, CallbackAction, CallbackHook, StepContext};

pub struct ProgressLogger {
    frequency: usize,
}

impl ProgressLogger {
    pub fn new(frequency: usize) -> Self {
        Self {
            frequency: frequency.max(1),
        }
    }
}

impl Default for ProgressLogger {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Callback for ProgressLogger {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::Objective]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(self.frequency) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if context.iteration.is_multiple_of(self.frequency)
            && let Some(objective) = context.diagnostics.objective
        {
            eprintln!(
                "iteration {:>5}: objective {objective:.6e}",
                context.iteration
            );
        }
        Ok(CallbackAction::Continue)
    }
}
