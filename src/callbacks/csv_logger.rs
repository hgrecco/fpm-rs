use std::{fs::File, path::PathBuf};

use crate::{Result, diagnostics::DiagnosticRequest};

use super::{Callback, CallbackAction, CallbackHook, StepContext};

pub struct CsvLogger {
    path: PathBuf,
    writer: Option<csv::Writer<File>>,
}

impl CsvLogger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            writer: None,
        }
    }
}

impl Callback for CsvLogger {
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

    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        let mut writer = csv::Writer::from_path(&self.path)?;
        writer.write_record(["iteration", "loss"])?;
        self.writer = Some(writer);
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if let (Some(writer), Some(loss)) = (&mut self.writer, context.diagnostics.loss) {
            writer.serialize((context.iteration, loss))?;
            writer.flush()?;
        }
        Ok(CallbackAction::Continue)
    }
}
