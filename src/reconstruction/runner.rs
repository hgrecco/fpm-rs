use ndarray::Array2;
use std::{collections::BTreeSet, sync::Arc, time::Instant};

use crate::{
    Result,
    algorithms::{AlgorithmIterationMetrics, ReconstructionAlgorithm, StepOutput, StepSummary},
    backend::Backend,
    callbacks::{Callback, CallbackAction, CallbackHook, StepContext},
    complex,
    diagnostics::{
        DiagnosticRequest, Diagnostics, FrameDiagnosticRecord, RawFrameStatisticsRecord,
    },
    error::Error,
    measurements::MeasurementRead,
    metrics::intensity::{compare_intensity_u8_masked, stats},
    model::ForwardModel,
};

use super::{
    Batch, ReconstructionCheckpoint, ReconstructionProblem, ReconstructionResult,
    ReconstructionState, ReconstructionTrace, RunOptions, RuntimeInfo, state_object,
};

/// Configurable executor for one reconstruction algorithm.
///
/// A runner owns the algorithm, run options, callbacks, optional checkpoint, and optional
/// backend. [`Self::run`] consumes the runner and borrows a validated problem.
pub struct Runner<A> {
    algorithm: A,
    options: RunOptions,
    callbacks: Vec<Box<dyn Callback>>,
    initial_checkpoint: Option<ReconstructionCheckpoint>,
    backend: Option<Arc<dyn Backend>>,
}

impl<A: ReconstructionAlgorithm> Runner<A> {
    /// Creates a runner from an algorithm and algorithm-independent options.
    pub fn new(algorithm: A, options: RunOptions) -> Self {
        Self {
            algorithm,
            options,
            callbacks: Vec::new(),
            initial_checkpoint: None,
            backend: None,
        }
    }

    /// Appends one callback in invocation order.
    pub fn with_callback(mut self, callback: Box<dyn Callback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Appends callbacks in their vector order.
    pub fn with_callbacks(mut self, callbacks: Vec<Box<dyn Callback>>) -> Self {
        self.callbacks.extend(callbacks);
        self
    }

    /// Sets a validated checkpoint from which state and elapsed trace are restored.
    pub fn resume_from(mut self, checkpoint: ReconstructionCheckpoint) -> Self {
        self.initial_checkpoint = Some(checkpoint);
        self
    }

    /// Selects an execution backend used to initialize or restore state.
    pub fn with_backend(mut self, backend: Arc<dyn Backend>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Validates inputs, executes scheduled batches and callbacks, and returns owned results.
    ///
    /// The run stops at `max_iterations` or when a callback returns a stop action. A supplied
    /// checkpoint must match the problem and the algorithm's required auxiliary state.
    pub fn run<M: MeasurementRead>(
        mut self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionResult> {
        self.algorithm.validate()?;
        problem.validate()?;
        self.algorithm.validate_problem(problem)?;
        if self.options.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                reason: "must be greater than zero".into(),
            });
        }
        let started = Instant::now();
        let algorithm_name = std::any::type_name::<A>()
            .rsplit("::")
            .next()
            .unwrap_or("reconstruction algorithm")
            .to_owned();
        let (mut state, mut trace, starting_iteration) =
            if let Some(checkpoint) = &self.initial_checkpoint {
                (
                    if let Some(backend) = &self.backend {
                        ReconstructionState::from_checkpoint_with_backend(
                            problem,
                            checkpoint,
                            backend.clone(),
                        )?
                    } else {
                        ReconstructionState::from_checkpoint(problem, checkpoint)?
                    },
                    checkpoint.trace.clone(),
                    checkpoint.completed_iterations,
                )
            } else {
                (
                    if let Some(backend) = &self.backend {
                        self.algorithm
                            .initialize_with_backend(problem, backend.clone())?
                    } else {
                        self.algorithm.initialize(problem)?
                    },
                    ReconstructionTrace::default(),
                    0,
                )
            };
        self.algorithm.canonicalize_state(problem, &mut state)?;
        state.object_real_space_cache = None;
        let previous_elapsed = trace
            .iterations
            .last()
            .map_or(0.0, |record| record.elapsed_seconds);
        let start_requests =
            callback_requests(&self.callbacks, CallbackHook::Start, starting_iteration);
        let start_diagnostics = build_diagnostics(
            problem,
            &mut state,
            &start_requests,
            None,
            None,
            starting_iteration,
        )?;
        let start_context = StepContext {
            iteration: starting_iteration,
            frame_index: None,
            batch_index: None,
            state: &state,
            diagnostics: &start_diagnostics,
            trace: &trace,
            current_algorithm_metrics: &[],
            model: state.calibrated_model.as_ref().unwrap_or(&problem.model),
            problem_name: problem.name.as_deref(),
        };
        let mut stopped_early = false;
        for callback in &mut self.callbacks {
            if callback.on_start(&start_context)? == CallbackAction::Stop {
                stopped_early = true;
            }
        }
        if !stopped_early {
            for zero_based_iteration in starting_iteration..self.options.max_iterations {
                let current_iteration = zero_based_iteration + 1;
                let frame_requests =
                    callback_requests(&self.callbacks, CallbackHook::FrameEnd, current_iteration);
                let order = self
                    .options
                    .schedule
                    .order_for_problem(problem, zero_based_iteration)?;
                let mut iteration_step = StepOutput::<A::IterationMetrics>::default();
                for (batch_index, indices) in order.chunks(self.options.batch_size).enumerate() {
                    let batch = Batch::new(indices.to_vec(), batch_index);
                    let batch_step =
                        self.algorithm
                            .step(problem, &mut state, &batch, zero_based_iteration)?;
                    state.object_real_space_cache = None;
                    if self.options.enable_frame_callbacks {
                        let batch_objective = batch_step.summary.mean_objective();
                        let mut batch_metric_records = Vec::new();
                        batch_step
                            .metrics
                            .append_records(current_iteration, &mut batch_metric_records);
                        let mut diagnostics = build_diagnostics(
                            problem,
                            &mut state,
                            &frame_requests,
                            batch_objective,
                            Some(&batch_step.summary),
                            current_iteration,
                        )?;
                        // A batch update completes every frame in the batch at
                        // once. Emit one frame hook per completed frame, with
                        // the frame's own natural loss and the shared post-batch
                        // state. Expensive diagnostics are computed only once.
                        for &frame in &batch.indices {
                            if frame_requests.contains(&DiagnosticRequest::Objective) {
                                diagnostics.objective =
                                    batch_step.summary.per_frame_objective.get(&frame).copied();
                            }
                            let context = StepContext {
                                iteration: current_iteration,
                                frame_index: Some(frame),
                                batch_index: Some(batch_index),
                                state: &state,
                                diagnostics: &diagnostics,
                                trace: &trace,
                                current_algorithm_metrics: &batch_metric_records,
                                model: state.calibrated_model.as_ref().unwrap_or(&problem.model),
                                problem_name: problem.name.as_deref(),
                            };
                            for callback in &mut self.callbacks {
                                if callback.on_frame_end(&context)? == CallbackAction::Stop {
                                    stopped_early = true;
                                }
                            }
                            if stopped_early {
                                break;
                            }
                        }
                    }
                    iteration_step.merge(batch_step);
                    if stopped_early {
                        break;
                    }
                }
                self.algorithm.canonicalize_state(problem, &mut state)?;
                state.object_real_space_cache = None;
                let objective = iteration_step.summary.mean_objective().unwrap_or(f64::NAN);
                trace.iterations.push(super::IterationRecord {
                    iteration: current_iteration,
                    objective,
                    elapsed_seconds: previous_elapsed + started.elapsed().as_secs_f64(),
                });
                let metric_start = trace.algorithm_metrics.len();
                iteration_step
                    .metrics
                    .append_records(current_iteration, &mut trace.algorithm_metrics);
                let iteration_requests = callback_requests(
                    &self.callbacks,
                    CallbackHook::IterationEnd,
                    current_iteration,
                );
                let diagnostics = build_diagnostics(
                    problem,
                    &mut state,
                    &iteration_requests,
                    Some(objective),
                    Some(&iteration_step.summary),
                    current_iteration,
                )?;
                let context = StepContext {
                    iteration: current_iteration,
                    frame_index: None,
                    batch_index: None,
                    state: &state,
                    diagnostics: &diagnostics,
                    trace: &trace,
                    current_algorithm_metrics: &trace.algorithm_metrics[metric_start..],
                    model: state.calibrated_model.as_ref().unwrap_or(&problem.model),
                    problem_name: problem.name.as_deref(),
                };
                for callback in &mut self.callbacks {
                    if callback.on_iteration_end(&context)? == CallbackAction::Stop {
                        stopped_early = true;
                    }
                }
                if stopped_early {
                    break;
                }
            }
        }

        let runtime = RuntimeInfo {
            elapsed_seconds: previous_elapsed + started.elapsed().as_secs_f64(),
            completed_iterations: trace
                .iterations
                .last()
                .map_or(starting_iteration, |record| record.iteration),
            stopped_early,
            algorithm: algorithm_name,
        };
        let mut result = ReconstructionResult::from_state(&mut state, trace, runtime)?;
        if let Some(name) = &problem.name {
            result.metadata.insert("problem_name".into(), name.clone());
        }
        for callback in &mut self.callbacks {
            callback.on_finish(&result)?;
        }
        Ok(result)
    }
}

fn callback_requests(
    callbacks: &[Box<dyn Callback>],
    hook: CallbackHook,
    iteration: usize,
) -> BTreeSet<DiagnosticRequest> {
    callbacks
        .iter()
        .flat_map(|callback| callback.requires_for(hook, iteration))
        .collect()
}

fn build_diagnostics<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
    requests: &BTreeSet<DiagnosticRequest>,
    natural_objective: Option<f64>,
    step: Option<&StepSummary>,
    iteration: usize,
) -> Result<Diagnostics> {
    let mut diagnostics = Diagnostics::default();
    if requests.contains(&DiagnosticRequest::Objective) {
        diagnostics.objective = natural_objective;
    }
    if requests.contains(&DiagnosticRequest::RawFrameStats) {
        let mut values = Vec::with_capacity(problem.model.frame_count());
        for frame in 0..problem.model.frame_count() {
            let measured = problem.measurements.frame(frame)?;
            values.push(RawFrameStatisticsRecord {
                frame_index: frame,
                metrics: stats(&measured, None)?,
            });
        }
        diagnostics.raw_frame_statistics = Some(values);
    }
    if requests.contains(&DiagnosticRequest::PerFrameError)
        && let Some(step) = step
    {
        let mut values = vec![f64::NAN; problem.model.frame_count()];
        for (&frame, &objective) in &step.per_frame_objective {
            values[frame] = objective;
        }
        diagnostics.per_frame_objective = Some(values);
    }
    if requests.contains(&DiagnosticRequest::FrameSummaries)
        || requests.contains(&DiagnosticRequest::ResidualImages)
        || (requests.contains(&DiagnosticRequest::PerFrameError) && step.is_none())
    {
        let diagnostic_model = model_with_state_calibration(problem, state)?;
        let forward = ForwardModel::with_backend(&diagnostic_model, state.backend.clone())?;
        let mut workspace = forward.workspace()?;
        let mut predicted =
            vec![0.0; crate::array_layout::checked_len_2d(problem.model.image_shape)?];
        let mut frame_diagnostics = Vec::with_capacity(problem.model.frame_count());
        let mut residual_images = if requests.contains(&DiagnosticRequest::ResidualImages) {
            Some(Vec::with_capacity(problem.model.frame_count()))
        } else {
            None
        };
        let mut per_frame_objective =
            if requests.contains(&DiagnosticRequest::PerFrameError) && step.is_none() {
                Some(Vec::with_capacity(problem.model.frame_count()))
            } else {
                None
            };
        for frame in 0..problem.model.frame_count() {
            forward.forward_intensity_standard_into(
                state.object_spectrum_standard_view(),
                &state.pupil,
                frame,
                &mut workspace,
                &mut predicted,
            )?;
            let measured = problem.measurements.frame(frame)?;
            let mask = problem.measurements.frame_mask(frame)?;
            let metadata = problem
                .measurements
                .frame_metadata()
                .get(frame)
                .cloned()
                .unwrap_or_else(|| crate::measurements::FrameMetadata::new(frame));
            if let Some(values) = per_frame_objective.as_mut() {
                let mut loss_sum = 0.0;
                let mut valid_pixels = 0;
                for pixel in 0..predicted.len() {
                    if mask.is_none_or(|values| values[pixel] != 0) {
                        let residual =
                            predicted[pixel].max(0.0).sqrt() - measured[pixel].max(0.0).sqrt();
                        loss_sum += residual * residual;
                        valid_pixels += 1;
                    }
                }
                values.push(if valid_pixels == 0 {
                    0.0
                } else {
                    loss_sum / valid_pixels as f64
                });
            }
            if let Some(images) = residual_images.as_mut() {
                let mut residual: Vec<_> = predicted
                    .iter()
                    .zip(measured.iter())
                    .map(|(&predicted, &measured)| predicted - measured)
                    .collect();
                if let Some(mask) = mask {
                    for (value, &valid) in residual.iter_mut().zip(mask) {
                        if valid == 0 {
                            *value = 0.0;
                        }
                    }
                }
                images.push(Array2::from_shape_vec(problem.model.image_shape, residual)?);
            }
            if requests.contains(&DiagnosticRequest::FrameSummaries) {
                let metrics = compare_intensity_u8_masked(&measured, &predicted, mask, None)?;
                frame_diagnostics.push(FrameDiagnosticRecord {
                    iteration: Some(iteration),
                    frame_index: frame,
                    illumination_index: metadata.illumination_index.unwrap_or(frame),
                    metrics,
                });
            }
        }
        if diagnostics.per_frame_objective.is_none() {
            diagnostics.per_frame_objective = per_frame_objective;
        }
        diagnostics.frame_diagnostics = Some(frame_diagnostics);
        if let Some(images) = residual_images {
            diagnostics.residual_images = Some(images);
        }
    }
    if requests.contains(&DiagnosticRequest::ObjectAmplitude)
        || requests.contains(&DiagnosticRequest::ObjectPhase)
    {
        let object = state_object(state)?;
        if requests.contains(&DiagnosticRequest::ObjectAmplitude) {
            diagnostics.object_amplitude = Some(complex::amplitude(object.view()));
        }
        if requests.contains(&DiagnosticRequest::ObjectPhase) {
            diagnostics.object_phase = Some(complex::phase(object.view()));
        }
    }
    if requests.contains(&DiagnosticRequest::Pupil) {
        diagnostics.pupil_amplitude = Some(complex::amplitude(state.pupil.values()));
        diagnostics.pupil_phase = Some(complex::phase(state.pupil.values()));
    }
    Ok(diagnostics)
}

fn model_with_state_calibration<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
) -> Result<crate::model::ImagePlaneModel> {
    let mut model = state
        .calibrated_model
        .clone()
        .unwrap_or_else(|| problem.model.clone());
    model.frame_gains = state.frame_gains.clone();
    model.background = state.background.clone();
    if state.illumination_corrections.is_some() {
        model.subpixel_offsets = Some(
            (0..model.source_count())
                .map(|source| state.effective_source_offset(&model, source))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    model.validate()?;
    Ok(model)
}
