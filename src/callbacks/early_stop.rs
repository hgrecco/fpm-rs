use crate::{Result, diagnostics::DiagnosticRequest};

use super::{Callback, CallbackAction, CallbackHook, StepContext};

pub struct StopOnPlateau {
    patience: usize,
    minimum_improvement: f64,
    best: f64,
    stale_iterations: usize,
}

impl StopOnPlateau {
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
        vec![DiagnosticRequest::Loss]
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
        for record in &context.history.iterations {
            if self.best - record.loss > self.minimum_improvement {
                self.best = record.loss;
                self.stale_iterations = 0;
            } else {
                self.stale_iterations += 1;
            }
        }
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if let Some(loss) = context.diagnostics.loss {
            if self.best - loss > self.minimum_improvement {
                self.best = loss;
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
