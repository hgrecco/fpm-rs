use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

use ndarray::{Array2, ArrayView2};

use crate::{
    Result,
    callbacks::{Callback, CallbackAction, StepContext},
    reconstruction::ReconstructionResult,
};

use super::{
    DiagnosticRequest, IterationDiagnostics, ReconstructionDiagnostics, compute_fourier_coverage,
};

#[derive(Clone, Debug)]
pub struct DiagnosticRecorderConfig {
    pub every: usize,

    pub record_iteration_history: bool,
    pub record_frame_summaries: bool,
    pub record_raw_stack_stats: bool,
    pub record_coverage: bool,

    pub record_object_snapshots: bool,
    pub record_pupil_snapshots: bool,
    pub snapshot_every: usize,
}

impl Default for DiagnosticRecorderConfig {
    fn default() -> Self {
        Self {
            every: 1,
            record_iteration_history: true,
            record_frame_summaries: false,
            record_raw_stack_stats: false,
            record_coverage: false,
            record_object_snapshots: false,
            record_pupil_snapshots: false,
            snapshot_every: 10,
        }
    }
}

#[derive(Default)]
struct DiagnosticRecorderState {
    diagnostics: ReconstructionDiagnostics,
    started_at: Option<Instant>,
    previous_object: Option<Array2<Complex64Proxy>>,
    previous_pupil: Option<Array2<Complex64Proxy>>,
    object_snapshots: Vec<(usize, Array2<f64>, Array2<f64>)>,
    pupil_snapshots: Vec<(usize, Array2<f64>, Array2<f64>)>,
}

/// A cloneable callback whose recorded output remains accessible after a run.
///
/// Pass a clone to a runner and retain the original handle:
///
/// ```no_run
/// # use fpm_rs::diagnostics::{DiagnosticRecorder, DiagnosticRecorderConfig};
/// let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig::default());
/// let callback = Box::new(recorder.clone());
/// # let _ = callback;
/// // Run with `callback`, then read `recorder.diagnostics()`.
/// ```
///
/// Clones share one recorder state and may be read from another thread while a
/// run is active, but they must not be installed in multiple concurrent runs:
/// each run resets the shared state at its start. Recorder state is
/// best-effort diagnostics, so a poisoned mutex is recovered and reset on the
/// next run rather than preventing later diagnostic reads or reuse.
#[derive(Clone)]
pub struct DiagnosticRecorder {
    config: DiagnosticRecorderConfig,
    state: Arc<Mutex<DiagnosticRecorderState>>,
}

type Complex64Proxy = num_complex::Complex64;

impl DiagnosticRecorder {
    pub fn new(config: DiagnosticRecorderConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(DiagnosticRecorderState::default())),
        }
    }

    pub fn diagnostics(&self) -> ReconstructionDiagnostics {
        self.lock_state().diagnostics.clone()
    }

    pub fn into_diagnostics(self) -> ReconstructionDiagnostics {
        self.diagnostics()
    }

    pub fn object_snapshots(&self) -> Vec<(usize, Array2<f64>, Array2<f64>)> {
        self.lock_state().object_snapshots.clone()
    }

    pub fn pupil_snapshots(&self) -> Vec<(usize, Array2<f64>, Array2<f64>)> {
        self.lock_state().pupil_snapshots.clone()
    }

    /// Clears all values collected by this recorder and its clones.
    pub fn reset(&self) {
        *self.lock_state() = DiagnosticRecorderState::default();
    }

    fn lock_state(&self) -> MutexGuard<'_, DiagnosticRecorderState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Callback for DiagnosticRecorder {
    fn requires_for(
        &self,
        hook: crate::callbacks::CallbackHook,
        iteration: usize,
    ) -> Vec<DiagnosticRequest> {
        let should_record = cadence_matches(iteration, self.config.every);
        let should_snapshot = cadence_matches(iteration, self.config.snapshot_every);
        match hook {
            crate::callbacks::CallbackHook::Start => {
                let mut requests = Vec::new();
                if self.config.record_raw_stack_stats {
                    requests.push(DiagnosticRequest::RawFrameStats);
                }
                requests
            }
            crate::callbacks::CallbackHook::IterationEnd if should_record || should_snapshot => {
                let mut requests = Vec::new();
                if should_record && self.config.record_iteration_history {
                    requests.push(DiagnosticRequest::Loss);
                    requests.push(DiagnosticRequest::PerFrameError);
                }
                if should_record && self.config.record_frame_summaries {
                    requests.push(DiagnosticRequest::FrameSummaries);
                }
                if should_snapshot && self.config.record_object_snapshots {
                    requests.push(DiagnosticRequest::ObjectAmplitude);
                    requests.push(DiagnosticRequest::ObjectPhase);
                }
                if should_snapshot && self.config.record_pupil_snapshots {
                    requests.push(DiagnosticRequest::Pupil);
                }
                requests
            }
            _ => Vec::new(),
        }
    }

    fn on_start(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        let mut state = self.lock_state();
        *state = DiagnosticRecorderState::default();
        state.started_at = Some(Instant::now());
        if self.config.record_coverage {
            state.diagnostics.coverage = Some(compute_fourier_coverage(context.model)?);
        }
        if let Some(raw_frame_stats) = &context.diagnostics.raw_frame_stats {
            state
                .diagnostics
                .raw_frame_stats
                .extend(raw_frame_stats.iter().cloned());
        }
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        let should_record = cadence_matches(context.iteration, self.config.every);
        let should_snapshot = cadence_matches(context.iteration, self.config.snapshot_every);
        let mut state = self.lock_state();

        if should_record && self.config.record_iteration_history {
            let elapsed_ms = state
                .started_at
                .map(|started| started.elapsed().as_secs_f64() * 1e3);
            let object_relative_change = relative_change(
                state.previous_object.as_ref(),
                context.state.object_spectrum.ndarray_view(),
            );
            let pupil_relative_change =
                relative_change(state.previous_pupil.as_ref(), context.state.pupil.values());
            state
                .diagnostics
                .iteration_history
                .push(IterationDiagnostics {
                    iteration: context.iteration,
                    total_loss: context.diagnostics.loss,
                    data_loss: context.diagnostics.loss,
                    regularization_loss: None,
                    object_relative_change,
                    pupil_relative_change,
                    median_frame_loss: context
                        .diagnostics
                        .per_frame_error
                        .as_ref()
                        .and_then(|values| median(values)),
                    worst_frame_loss: context
                        .diagnostics
                        .per_frame_error
                        .as_ref()
                        .and_then(|values| values.iter().copied().reduce(f64::max)),
                    elapsed_ms,
                });
        }
        if should_record {
            state.previous_object = Some(context.state.object_spectrum.clone().into_inner());
            state.previous_pupil = Some(context.state.pupil.values.clone().into_inner());
        }

        if should_record && let Some(frame_diagnostics) = &context.diagnostics.frame_diagnostics {
            state
                .diagnostics
                .frame_diagnostics
                .extend(frame_diagnostics.iter().cloned());
        }
        if self.config.record_object_snapshots
            && should_snapshot
            && let (Some(amplitude), Some(phase)) = (
                context.diagnostics.object_amplitude.clone(),
                context.diagnostics.object_phase.clone(),
            )
        {
            state
                .object_snapshots
                .push((context.iteration, amplitude, phase));
        }
        if self.config.record_pupil_snapshots
            && should_snapshot
            && let (Some(amplitude), Some(phase)) = (
                context.diagnostics.pupil_amplitude.clone(),
                context.diagnostics.pupil_phase.clone(),
            )
        {
            state
                .pupil_snapshots
                .push((context.iteration, amplitude, phase));
        }
        Ok(CallbackAction::Continue)
    }

    fn on_finish(&mut self, _result: &ReconstructionResult) -> Result<()> {
        Ok(())
    }
}

fn cadence_matches(iteration: usize, every: usize) -> bool {
    every > 0 && iteration.is_multiple_of(every)
}

fn relative_change(
    previous: Option<&Array2<Complex64Proxy>>,
    current: ArrayView2<'_, Complex64Proxy>,
) -> Option<f64> {
    let previous = previous?;
    if previous.dim() != current.dim() {
        return None;
    }
    let mut difference: f64 = 0.0;
    let mut reference: f64 = 0.0;
    for (&previous, &current) in previous.iter().zip(current.iter()) {
        difference += (current - previous).norm_sqr();
        reference += previous.norm_sqr();
    }
    Some(difference.sqrt() / reference.sqrt().max(f64::EPSILON))
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    })
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    #[test]
    fn poisoned_recorder_state_remains_resettable_and_readable() {
        let recorder = DiagnosticRecorder::new(DiagnosticRecorderConfig::default());
        let state = recorder.state.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = state.lock().unwrap();
            panic!("intentional recorder-lock poison for test");
        }));
        assert!(result.is_err());

        recorder.reset();
        assert!(recorder.diagnostics().iteration_history.is_empty());
        assert!(recorder.object_snapshots().is_empty());
        assert!(recorder.pupil_snapshots().is_empty());
    }
}
