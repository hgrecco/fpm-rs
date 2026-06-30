use std::{fs, path::PathBuf};

use crate::{
    Result,
    diagnostics::DiagnosticRequest,
    reconstruction::{save_grayscale, save_signed_grayscale},
};

use super::{Callback, CallbackAction, CallbackHook, StepContext};

pub struct SaveImageEvery {
    frequency: usize,
    directory: PathBuf,
}

impl SaveImageEvery {
    pub fn new(frequency: usize, directory: impl Into<PathBuf>) -> Self {
        Self {
            frequency: frequency.max(1),
            directory: directory.into(),
        }
    }
}

impl Callback for SaveImageEvery {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![
            DiagnosticRequest::ObjectAmplitude,
            DiagnosticRequest::ObjectPhase,
        ]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(self.frequency) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        fs::create_dir_all(&self.directory)?;
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if context.iteration.is_multiple_of(self.frequency) {
            if let Some(amplitude) = &context.diagnostics.object_amplitude {
                save_grayscale(
                    amplitude,
                    self.directory
                        .join(format!("amplitude_{:05}.png", context.iteration)),
                    false,
                )?;
            }
            if let Some(phase) = &context.diagnostics.object_phase {
                save_grayscale(
                    phase,
                    self.directory
                        .join(format!("phase_{:05}.png", context.iteration)),
                    true,
                )?;
            }
        }
        Ok(CallbackAction::Continue)
    }
}

pub struct SavePupilEvery {
    frequency: usize,
    directory: PathBuf,
}

impl SavePupilEvery {
    pub fn new(frequency: usize, directory: impl Into<PathBuf>) -> Self {
        Self {
            frequency: frequency.max(1),
            directory: directory.into(),
        }
    }
}

impl Callback for SavePupilEvery {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::Pupil]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(self.frequency) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        fs::create_dir_all(&self.directory)?;
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if context.iteration.is_multiple_of(self.frequency) {
            if let Some(amplitude) = &context.diagnostics.pupil_amplitude {
                save_grayscale(
                    amplitude,
                    self.directory
                        .join(format!("pupil_amplitude_{:05}.png", context.iteration)),
                    false,
                )?;
            }
            if let Some(phase) = &context.diagnostics.pupil_phase {
                save_grayscale(
                    phase,
                    self.directory
                        .join(format!("pupil_phase_{:05}.png", context.iteration)),
                    true,
                )?;
            }
        }
        Ok(CallbackAction::Continue)
    }
}

/// Periodically saves one signed intensity-residual PNG per measured frame.
///
/// Each image is scaled symmetrically around zero so mid-gray represents zero,
/// dark pixels are negative residuals, and bright pixels are positive residuals.
pub struct SaveResidualsEvery {
    frequency: usize,
    directory: PathBuf,
}

impl SaveResidualsEvery {
    pub fn new(frequency: usize, directory: impl Into<PathBuf>) -> Self {
        Self {
            frequency: frequency.max(1),
            directory: directory.into(),
        }
    }
}

impl Callback for SaveResidualsEvery {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::ResidualImages]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(self.frequency) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        fs::create_dir_all(&self.directory)?;
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        if context.iteration.is_multiple_of(self.frequency)
            && let Some(images) = &context.diagnostics.residual_images
        {
            for (frame, residual) in images.iter().enumerate() {
                save_signed_grayscale(
                    residual,
                    self.directory.join(format!(
                        "residual_{:05}_frame_{frame:05}.png",
                        context.iteration
                    )),
                )?;
            }
        }
        Ok(CallbackAction::Continue)
    }
}
