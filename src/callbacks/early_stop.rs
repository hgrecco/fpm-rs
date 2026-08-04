use crate::{Result, diagnostics::DiagnosticRequest};

use super::{Callback, CallbackAction, CallbackHook, StepContext};

/// Stops after a configured number of iterations without sufficient objective improvement.
pub struct StopOnPlateau {
    patience: usize,
    minimum_improvement: f64,
    best: f64,
    stale_iterations: usize,
}

impl StopOnPlateau {
    /// Creates a stopper with at least one iteration of patience and a non-negative
    /// absolute `minimum_improvement`.
    pub fn new(patience: usize, minimum_improvement: f64) -> Self {
        Self {
            patience: patience.max(1),
            minimum_improvement: minimum_improvement.max(0.0),
            best: f64::INFINITY,
            stale_iterations: 0,
        }
    }
}

impl Callback for StopOnPlateau {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::Objective]
    }

    fn requires_for(&self, hook: CallbackHook, _iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_start(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        self.best = f64::INFINITY;
        self.stale_iterations = 0;
        for record in &context.trace.iterations {
            if self.best - record.objective > self.minimum_improvement {
                self.best = record.objective;
                self.stale_iterations = 0;
            } else {
                self.stale_iterations += 1;
            }
        }
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if let Some(objective) = context.diagnostics.objective {
            if self.best - objective > self.minimum_improvement {
                self.best = objective;
                self.stale_iterations = 0;
            } else {
                self.stale_iterations += 1;
            }
        }
        Ok(if self.stale_iterations >= self.patience {
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        })
    }
}
