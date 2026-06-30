use std::{fs, path::PathBuf};

use crate::{Result, reconstruction::ReconstructionCheckpoint};

use super::{Callback, CallbackAction, StepContext};

pub struct CheckpointEvery {
    frequency: usize,
    directory: PathBuf,
}

impl CheckpointEvery {
    pub fn new(frequency: usize, directory: impl Into<PathBuf>) -> Self {
        Self {
            frequency: frequency.max(1),
            directory: directory.into(),
        }
    }
}

impl Callback for CheckpointEvery {
    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        fs::create_dir_all(&self.directory)?;
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if context.iteration.is_multiple_of(self.frequency) {
            ReconstructionCheckpoint::capture(context.iteration, context.state, context.history)
                .save(
                    self.directory
                        .join(format!("checkpoint_{:05}.json", context.iteration)),
                )?;
        }
        Ok(CallbackAction::Continue)
    }
}
