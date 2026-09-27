use num_complex::Complex64;
use std::thread;

use crate::{
    Result,
    algorithms::{
        NoIterationMetrics, StepOutput, StepSummary,
        objective::{LossType, point_loss},
    },
    array_layout::checked_len_2d,
    backend::FftDirection,
    error::Error,
    measurements::MeasurementRead,
    model::{FourierOffset, fftshift_copy, ifftshift_copy},
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

use super::{
    ReconstructionAlgorithm,
    gauge::canonicalize_object_pupil,
    regularization::{apply_complex_tv_step, apply_quadratic_smoothing_step},
};

/// Wirtinger-style loss-gradient reconstruction for Fourier ptychography.
///
/// # Method
///
/// The solver differentiates a selected real-valued data loss through the
/// complex FPM forward model, accumulates gradients from a mini-batch, and
/// applies a pupil-power-preconditioned update to the shared object spectrum.
/// This is the Fourier-ptychographic Wirtinger-flow viewpoint: phase retrieval
/// is treated as direct optimization rather than alternating hard projections.
/// Losses are evaluated after accounting for known linear gain and background,
/// so detector count scaling does not change the intrinsic update scale.
///
/// Optional extensions recover the pupil with an analogous normalized
/// gradient, estimate illumination offsets with finite-difference derivatives
/// and diagonal Gauss–Newton scaling, and regularize the complex object or
/// pupil. Incoherent multiplexing, these calibration updates, and the selectable
/// losses extend the reference formulation.
///
/// # Gauge convention
///
/// With pupil recovery enabled, the compiled pupil fixes the reported joint
/// object/pupil gauge at iteration boundaries. The projection matches supported
/// pupil energy and piston, fixes the remaining object piston, and removes an
/// affine pupil phase ramp only on axes with zero effective subpixel offsets.
/// Reciprocal object corrections preserve predicted intensities. Fractional
/// axes retain their affine phase because bilinear crop interpolation does not
/// commute exactly with a discrete phase ramp. Canonicalization runs before
/// iteration callbacks, checkpoints, and final results, including after resume.
///
/// # References
///
/// [L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai, “Fourier
/// ptychographic reconstruction using Wirtinger flow optimization”
/// (2015)](https://doi.org/10.1364/OE.23.004856), *Optics Express* **23**(4),
/// 4856–4866.
///
/// The blind object/pupil ambiguities follow [A. Fannjiang and P. Chen, “Blind
/// ptychography: uniqueness and ambiguities” (2020)](https://doi.org/10.1088/1361-6420/ab6504),
/// *Inverse Problems* **36**, 045005; this implementation uses a Fourier-domain
/// object and retains affine phase on fractionally interpolated axes.
#[derive(Clone, Debug)]
pub struct GradientDescent {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Step size of the pupil-power-preconditioned object update.
    pub object_step: f64,
    /// Number of frame gradients averaged into one update.
    pub batch_size: usize,
    /// Positive numerical floor used by losses and preconditioners.
    pub epsilon: f64,
    /// Data-fidelity objective to differentiate and report.
    pub loss_type: LossType,
    /// Whether to estimate a Fourier-grid offset for every illumination source.
    pub recover_illumination: bool,
    /// Step size of the diagonally scaled illumination-offset update.
    pub illumination_step: f64,
    /// Central finite-difference spacing in Fourier-grid pixels.
    pub illumination_finite_difference: f64,
    /// Maximum absolute row or column correction, in Fourier-grid pixels.
    pub maximum_illumination_correction: f64,
    /// Whether to update the complex pupil alongside the object.
    pub recover_pupil: bool,
    /// Step size of the normalized pupil update.
    pub pupil_step: f64,
    /// Whether to zero recovered pupil values outside the compiled aperture.
    pub constrain_pupil_support: bool,
    /// Weight of the isotropic total-variation step on the complex object;
    /// `0` disables it.
    pub object_tv_weight: f64,
    /// Positive smoothing constant in the differentiable TV norm.
    pub object_tv_epsilon: f64,
    /// Weight of quadratic nearest-neighbor pupil smoothing; `0` disables it
    /// and positive values require pupil recovery.
    pub pupil_smoothing_weight: f64,
    /// Maximum number of frame-gradient worker threads.
    pub parallel_workers: usize,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            iterations: 100,
            object_step: 0.5,
            batch_size: 1,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
            recover_illumination: false,
            illumination_step: 0.1,
            illumination_finite_difference: 0.05,
            maximum_illumination_correction: 1.0,
            recover_pupil: false,
            pupil_step: 0.05,
            constrain_pupil_support: true,
            object_tv_weight: 0.0,
            object_tv_epsilon: 1e-6,
            pupil_smoothing_weight: 0.0,
            parallel_workers: std::thread::available_parallelism().map_or(1, |count| count.get()),
        }
    }
}

impl GradientDescent {
    /// Sets the number of complete acquisition-schedule passes; validation requires non-zero.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the finite positive step size of the preconditioned object update.
    pub fn object_step(mut self, step: f64) -> Self {
        self.object_step = step;
        self
    }

    /// Sets the positive number of frame gradients averaged into one update.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Selects the differentiable data-fidelity objective used for updates and reporting.
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Enables or disables per-source Fourier-grid offset recovery.
    pub fn recover_illumination(mut self, recover: bool) -> Self {
        self.recover_illumination = recover;
        self
    }

    /// Sets the finite positive illumination-offset step size.
    pub fn illumination_step(mut self, step: f64) -> Self {
        self.illumination_step = step;
        self
    }

    /// Sets positive central finite-difference spacing in Fourier-grid pixels.
    pub fn illumination_finite_difference(mut self, distance: f64) -> Self {
        self.illumination_finite_difference = distance;
        self
    }

    /// Sets the non-negative maximum absolute row or column correction in grid pixels.
    pub fn illumination_bounds(mut self, maximum_absolute_correction: f64) -> Self {
        self.maximum_illumination_correction = maximum_absolute_correction;
        self
    }

    /// Enables or disables simultaneous complex-pupil recovery.
    pub fn recover_pupil(mut self, recover: bool) -> Self {
        self.recover_pupil = recover;
        self
    }

    /// Sets the finite positive normalized pupil-update step size.
    pub fn pupil_step(mut self, step: f64) -> Self {
        self.pupil_step = step;
        self
    }

    /// Selects whether pupil values outside the compiled binary support are forced to zero.
    pub fn constrain_pupil_support(mut self, constrain: bool) -> Self {
        self.constrain_pupil_support = constrain;
        self
    }

    /// Sets a non-negative complex-object isotropic total-variation weight.
    pub fn object_tv(mut self, weight: f64) -> Self {
        self.object_tv_weight = weight;
        self
    }

    /// Sets the finite positive smoothing constant in the differentiable TV norm.
    pub fn object_tv_epsilon(mut self, epsilon: f64) -> Self {
        self.object_tv_epsilon = epsilon;
        self
    }

    /// Sets a non-negative quadratic nearest-neighbor pupil-smoothing weight.
    pub fn pupil_smoothing(mut self, weight: f64) -> Self {
        self.pupil_smoothing_weight = weight;
        self
    }

    /// Sets the maximum number of frame-gradient workers. Object, pupil, and
    /// illumination-calibration contributions are reduced deterministically.
    pub fn parallel_workers(mut self, workers: usize) -> Self {
        self.parallel_workers = workers;
        self
    }
}

impl ReconstructionAlgorithm for GradientDescent {
    type IterationMetrics = NoIterationMetrics;

    fn validate(&self) -> Result<()> {
        if !self.object_step.is_finite() || self.object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        if self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.illumination_step.is_finite() || self.illumination_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "illumination_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.illumination_finite_difference.is_finite()
            || self.illumination_finite_difference <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "illumination_finite_difference",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.maximum_illumination_correction.is_finite()
            || self.maximum_illumination_correction <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "maximum_illumination_correction",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.pupil_step.is_finite() || self.pupil_step < 0.0 {
            return Err(Error::InvalidParameter {
                name: "pupil_step",
                reason: "must be finite and non-negative".into(),
            });
        }
        if !self.object_tv_weight.is_finite() || self.object_tv_weight < 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_tv_weight",
                reason: "must be finite and non-negative".into(),
            });
        }
        if !self.object_tv_epsilon.is_finite() || self.object_tv_epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_tv_epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.pupil_smoothing_weight.is_finite() || self.pupil_smoothing_weight < 0.0 {
            return Err(Error::InvalidParameter {
                name: "pupil_smoothing_weight",
                reason: "must be finite and non-negative".into(),
            });
        }
        if self.pupil_smoothing_weight > 0.0 && !self.recover_pupil {
            return Err(Error::InvalidParameter {
                name: "pupil_smoothing_weight",
                reason: "requires pupil recovery to be enabled".into(),
            });
        }
        if self.parallel_workers == 0 {
            return Err(Error::InvalidParameter {
                name: "parallel_workers",
                reason: "must be greater than zero".into(),
            });
        }
        Ok(())
    }

    fn canonicalize_state<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
    ) -> Result<()> {
        if self.recover_pupil {
            canonicalize_object_pupil(problem, state)?;
        }
        Ok(())
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        if self.parallel_workers > 1 && batch.indices.len() > 1 {
            return self.parallel_step(problem, state, batch, iteration);
        }
        let model = &problem.model;
        let shape = model.image_shape;
        let image_len = checked_len_2d(shape)?;
        let maximum_pupil_power = state
            .pupil
            .values
            .as_slice()
            .iter()
            .map(|value| value.norm_sqr())
            .fold(0.0, f64::max)
            .max(self.epsilon);
        let mut diagnostics = StepSummary::default();
        let mut active_frames = 0;
        if self.recover_illumination {
            prepare_illumination_accumulators(model, state)?;
        }
        state
            .scratch
            .object_gradient
            .resize(state.object_spectrum.len(), Complex64::default());
        state.scratch.object_gradient.fill(Complex64::default());
        if self.recover_pupil {
            state.scratch.pupil_gradient.fill(Complex64::default());
        }

        for &frame in &batch.indices {
            let frame_weight = problem.measurements.frame_weight(frame)?;
            if frame_weight == 0.0 {
                diagnostics.push_frame(frame, 0.0, 0.0);
                continue;
            }
            active_frames += 1;
            let single_source = [(frame, 1.0)];
            let sources = model
                .multiplexing_matrix
                .as_ref()
                .map_or(single_source.as_slice(), |matrix| matrix[frame].as_slice());

            state.scratch.projected_field.fill(Complex64::default());
            for &(source, source_weight) in sources {
                let offset = state.effective_source_offset(model, source)?;
                compute_source_field(problem, state, source, offset)?;
                for (predicted, field) in state
                    .scratch
                    .projected_field
                    .iter_mut()
                    .zip(&state.scratch.field)
                {
                    predicted.re += source_weight * field.norm_sqr();
                }
            }

            let measured = problem.measurements.frame(frame)?;
            let mask = problem.measurements.frame_mask(frame)?;
            let gain = state
                .frame_gains
                .as_ref()
                .map_or(1.0, |values| values[frame]);
            if !gain.is_finite() || gain <= 0.0 {
                return Err(Error::InvalidModel(format!(
                    "state frame {frame} has invalid gain {gain}"
                )));
            }
            let mut frame_loss = 0.0;
            let mut valid_pixels = 0;
            for pixel in 0..image_len {
                if mask.is_some_and(|values| values[pixel] == 0) {
                    state.scratch.projected_field[pixel] = Complex64::default();
                    continue;
                }
                valid_pixels += 1;
                let background = background_value(state, frame, pixel, image_len);
                let intrinsic_prediction = state.scratch.projected_field[pixel].re.max(0.0);
                let target_intensity = ((measured[pixel] - background) / gain).max(0.0);
                frame_loss += point_loss(intrinsic_prediction, target_intensity, self.loss_type);
                state.scratch.projected_field[pixel] = Complex64::new(
                    descent_factor(
                        intrinsic_prediction,
                        target_intensity,
                        self.loss_type,
                        self.epsilon,
                    ),
                    intrinsic_prediction,
                );
            }
            if valid_pixels == 0 {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {frame} has no unmasked pixels"
                )));
            }
            // The FFT adjoint contributes a 1/image_len normalization. Match
            // the reported per-frame mean when masked pixels reduce its divisor.
            let valid_pixel_scale = image_len as f64 / valid_pixels as f64;
            diagnostics.push_frame(frame, frame_loss / valid_pixels as f64, frame_weight);

            for &(source, source_weight) in sources {
                let offset = state.effective_source_offset(model, source)?;
                compute_source_field(problem, state, source, offset)?;
                if self.recover_illumination {
                    for (reference, field) in state
                        .scratch
                        .calibration_reference
                        .iter_mut()
                        .zip(&state.scratch.field)
                    {
                        *reference = field.norm_sqr();
                    }
                }
                for pixel in 0..image_len {
                    state.scratch.field[pixel] *=
                        source_weight * state.scratch.projected_field[pixel].re;
                }
                state.backend.fft2(
                    &mut state.scratch.field,
                    shape,
                    FftDirection::Forward,
                    &mut state.scratch.column,
                )?;
                fftshift_copy(
                    &state.scratch.field,
                    &mut state.scratch.projected_spectrum,
                    shape,
                );
                if self.recover_pupil {
                    let maximum_object_power = state
                        .scratch
                        .patch
                        .iter()
                        .map(|value| value.norm_sqr())
                        .fold(0.0, f64::max)
                        .max(self.epsilon);
                    for pixel in 0..image_len {
                        state.scratch.pupil_gradient[pixel] += frame_weight
                            * valid_pixel_scale
                            * state.scratch.patch[pixel].conj()
                            * state.scratch.projected_spectrum[pixel]
                            / (maximum_object_power + self.epsilon);
                    }
                }
                for pixel in 0..image_len {
                    state.scratch.difference[pixel] = state.pupil.values.as_slice()[pixel].conj()
                        * state.scratch.projected_spectrum[pixel]
                        / (maximum_pupil_power + self.epsilon);
                }
                model.insert_patch_adjoint_slice_at_offset(
                    &mut state.scratch.object_gradient,
                    source,
                    &state.scratch.difference,
                    frame_weight * valid_pixel_scale,
                    offset,
                )?;
                if self.recover_illumination {
                    let gradient = illumination_gradient(
                        problem,
                        state,
                        source,
                        offset,
                        IlluminationGradientConfiguration {
                            source_weight,
                            valid_pixels,
                            distance: self.illumination_finite_difference,
                            loss_type: self.loss_type,
                            epsilon: self.epsilon,
                        },
                    )?;
                    state.scratch.illumination_gradient[source].0 +=
                        frame_weight * gradient.row.gradient;
                    state.scratch.illumination_gradient[source].1 +=
                        frame_weight * gradient.column.gradient;
                    state.scratch.illumination_curvature[source].0 +=
                        frame_weight * gradient.row.curvature;
                    state.scratch.illumination_curvature[source].1 +=
                        frame_weight * gradient.column.curvature;
                    state.scratch.illumination_weight[source] += frame_weight;
                }
            }
        }
        if active_frames > 0 {
            let step = self.object_step / active_frames as f64;
            for (object, &gradient) in state
                .object_spectrum
                .as_slice_mut()
                .iter_mut()
                .zip(&state.scratch.object_gradient)
            {
                *object -= step * gradient;
            }
            if self.recover_pupil {
                let pupil_step = self.pupil_step / active_frames as f64;
                for (pupil, &gradient) in state
                    .pupil
                    .values
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&state.scratch.pupil_gradient)
                {
                    *pupil -= pupil_step * gradient;
                    if !pupil.re.is_finite() || !pupil.im.is_finite() {
                        return Err(Error::Numerical(
                            "pupil update produced a non-finite value".into(),
                        ));
                    }
                }
            }
        }
        let batch_fraction = batch.indices.len() as f64 / model.frame_count() as f64;
        if self.object_tv_weight > 0.0 {
            apply_object_tv(
                state,
                model.reconstruction_shape,
                batch_fraction * self.object_tv_weight,
                self.object_tv_epsilon,
            )?;
        }
        if self.recover_pupil && self.pupil_smoothing_weight > 0.0 {
            apply_quadratic_smoothing_step(
                state.pupil.values.as_slice_mut(),
                model.image_shape,
                batch_fraction * self.pupil_smoothing_weight,
                &mut state.scratch.pupil_gradient,
            )?;
        }
        if self.recover_pupil && self.constrain_pupil_support {
            state.pupil.apply_support();
        }
        if self.recover_illumination {
            self.apply_illumination_update(model, state)?;
        }
        Ok(diagnostics.into())
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

struct ParallelWorkerResult {
    position: usize,
    object_delta: Vec<Complex64>,
    pupil_delta: Vec<Complex64>,
    illumination_gradient: Vec<(f64, f64)>,
    illumination_curvature: Vec<(f64, f64)>,
    illumination_weight: Vec<f64>,
    diagnostics: StepSummary,
    active_frames: usize,
}

impl GradientDescent {
    fn parallel_step<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepOutput<NoIterationMetrics>> {
        let worker_count = self.parallel_workers.min(batch.indices.len());
        if self.recover_illumination {
            prepare_illumination_accumulators(&problem.model, state)?;
        }
        let base_state = state.clone();
        let mut results = thread::scope(|scope| -> Result<Vec<ParallelWorkerResult>> {
            let base_chunk_len = batch.indices.len() / worker_count;
            let remainder = batch.indices.len() % worker_count;
            let handles: Vec<_> = (0..worker_count)
                .map(|worker| {
                    let start = worker * base_chunk_len + worker.min(remainder);
                    let length = base_chunk_len + usize::from(worker < remainder);
                    let frames = &batch.indices[start..start + length];
                    let base_state = &base_state;
                    scope.spawn(move || -> Result<ParallelWorkerResult> {
                        let mut local_state = base_state.clone();
                        let mut local_algorithm = self.clone();
                        local_algorithm.parallel_workers = 1;
                        local_algorithm.object_tv_weight = 0.0;
                        local_algorithm.pupil_smoothing_weight = 0.0;
                        local_algorithm.constrain_pupil_support = false;
                        let mut output = ParallelWorkerResult {
                            position: worker,
                            object_delta: vec![
                                Complex64::default();
                                base_state.object_spectrum.len()
                            ],
                            pupil_delta: if self.recover_pupil {
                                vec![Complex64::default(); base_state.pupil.values.len()]
                            } else {
                                Vec::new()
                            },
                            illumination_gradient: if self.recover_illumination {
                                vec![(0.0, 0.0); problem.model.source_count()]
                            } else {
                                Vec::new()
                            },
                            illumination_curvature: if self.recover_illumination {
                                vec![(0.0, 0.0); problem.model.source_count()]
                            } else {
                                Vec::new()
                            },
                            illumination_weight: if self.recover_illumination {
                                vec![0.0; problem.model.source_count()]
                            } else {
                                Vec::new()
                            },
                            diagnostics: StepSummary::default(),
                            active_frames: 0,
                        };
                        for &frame in frames {
                            local_state
                                .object_spectrum
                                .as_slice_mut()
                                .copy_from_slice(base_state.object_spectrum.as_slice());
                            if self.recover_pupil {
                                local_state
                                    .pupil
                                    .values
                                    .as_slice_mut()
                                    .copy_from_slice(base_state.pupil.values.as_slice());
                            }
                            if self.recover_illumination {
                                local_state
                                    .illumination_corrections
                                    .clone_from(&base_state.illumination_corrections);
                            }
                            let diagnostics = local_algorithm.step(
                                problem,
                                &mut local_state,
                                &Batch::single(frame),
                                iteration,
                            )?;
                            if diagnostics.summary.weight_sum > 0.0 {
                                output.active_frames += 1;
                                for ((sum, &value), &initial) in output
                                    .object_delta
                                    .iter_mut()
                                    .zip(local_state.object_spectrum.as_slice())
                                    .zip(base_state.object_spectrum.as_slice())
                                {
                                    *sum += value - initial;
                                }
                                if self.recover_pupil {
                                    for ((sum, &value), &initial) in output
                                        .pupil_delta
                                        .iter_mut()
                                        .zip(local_state.pupil.values.as_slice())
                                        .zip(base_state.pupil.values.as_slice())
                                    {
                                        *sum += value - initial;
                                    }
                                }
                                if self.recover_illumination {
                                    for source in 0..problem.model.source_count() {
                                        output.illumination_gradient[source].0 +=
                                            local_state.scratch.illumination_gradient[source].0;
                                        output.illumination_gradient[source].1 +=
                                            local_state.scratch.illumination_gradient[source].1;
                                        output.illumination_curvature[source].0 +=
                                            local_state.scratch.illumination_curvature[source].0;
                                        output.illumination_curvature[source].1 +=
                                            local_state.scratch.illumination_curvature[source].1;
                                        output.illumination_weight[source] +=
                                            local_state.scratch.illumination_weight[source];
                                    }
                                }
                            }
                            output.diagnostics.merge(diagnostics.summary);
                        }
                        Ok(output)
                    })
                })
                .collect();
            let mut output = Vec::with_capacity(handles.len());
            for handle in handles {
                output.push(
                    handle.join().map_err(|_| {
                        Error::Numerical("parallel gradient worker panicked".into())
                    })??,
                );
            }
            Ok(output)
        })?;
        results.sort_by_key(|result| result.position);

        state
            .scratch
            .object_gradient
            .resize(state.object_spectrum.len(), Complex64::default());
        state.scratch.object_gradient.fill(Complex64::default());
        if self.recover_pupil {
            state.scratch.pupil_gradient.fill(Complex64::default());
        }
        if self.recover_illumination {
            state.scratch.illumination_gradient.fill((0.0, 0.0));
            state.scratch.illumination_curvature.fill((0.0, 0.0));
            state.scratch.illumination_weight.fill(0.0);
        }
        let mut diagnostics = StepSummary::default();
        let mut active_frames = 0;
        for result in results {
            active_frames += result.active_frames;
            for (sum, &value) in state
                .scratch
                .object_gradient
                .iter_mut()
                .zip(&result.object_delta)
            {
                *sum += value;
            }
            if self.recover_pupil {
                for (sum, &value) in state
                    .scratch
                    .pupil_gradient
                    .iter_mut()
                    .zip(&result.pupil_delta)
                {
                    *sum += value;
                }
            }
            if self.recover_illumination {
                for source in 0..problem.model.source_count() {
                    state.scratch.illumination_gradient[source].0 +=
                        result.illumination_gradient[source].0;
                    state.scratch.illumination_gradient[source].1 +=
                        result.illumination_gradient[source].1;
                    state.scratch.illumination_curvature[source].0 +=
                        result.illumination_curvature[source].0;
                    state.scratch.illumination_curvature[source].1 +=
                        result.illumination_curvature[source].1;
                    state.scratch.illumination_weight[source] += result.illumination_weight[source];
                }
            }
            diagnostics.merge(result.diagnostics);
        }
        if active_frames > 0 {
            let normalization = active_frames as f64;
            for (object, &sum) in state
                .object_spectrum
                .as_slice_mut()
                .iter_mut()
                .zip(&state.scratch.object_gradient)
            {
                *object += sum / normalization;
            }
            if self.recover_pupil {
                for (pupil, &sum) in state
                    .pupil
                    .values
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&state.scratch.pupil_gradient)
                {
                    *pupil += sum / normalization;
                    if !pupil.re.is_finite() || !pupil.im.is_finite() {
                        return Err(Error::Numerical(
                            "parallel pupil update produced a non-finite value".into(),
                        ));
                    }
                }
            }
        }

        let model = &problem.model;
        let batch_fraction = batch.indices.len() as f64 / model.frame_count() as f64;
        if self.object_tv_weight > 0.0 {
            apply_object_tv(
                state,
                model.reconstruction_shape,
                batch_fraction * self.object_tv_weight,
                self.object_tv_epsilon,
            )?;
        }
        if self.recover_pupil && self.pupil_smoothing_weight > 0.0 {
            apply_quadratic_smoothing_step(
                state.pupil.values.as_slice_mut(),
                model.image_shape,
                batch_fraction * self.pupil_smoothing_weight,
                &mut state.scratch.pupil_gradient,
            )?;
        }
        if self.recover_pupil && self.constrain_pupil_support {
            state.pupil.apply_support();
        }
        if self.recover_illumination {
            self.apply_illumination_update(model, state)?;
        }
        Ok(diagnostics.into())
    }

    fn apply_illumination_update(
        &self,
        model: &crate::model::ImagePlaneModel,
        state: &mut ReconstructionState,
    ) -> Result<()> {
        let corrections = state.illumination_corrections.as_mut().ok_or_else(|| {
            Error::InvalidModel("illumination corrections were not initialized".into())
        })?;
        for (source, correction) in corrections.iter_mut().enumerate() {
            let weight = state.scratch.illumination_weight[source];
            if weight == 0.0 {
                continue;
            }
            let gradient = state.scratch.illumination_gradient[source];
            let curvature = state.scratch.illumination_curvature[source];
            if !gradient.0.is_finite()
                || !gradient.1.is_finite()
                || !curvature.0.is_finite()
                || !curvature.1.is_finite()
                || curvature.0 < 0.0
                || curvature.1 < 0.0
            {
                return Err(Error::Numerical(format!(
                    "illumination gradient or curvature for source {source} is invalid"
                )));
            }
            let candidate = (
                (correction.0 - self.illumination_step * gradient.0 / (curvature.0 + self.epsilon))
                    .clamp(
                        -self.maximum_illumination_correction,
                        self.maximum_illumination_correction,
                    ),
                (correction.1 - self.illumination_step * gradient.1 / (curvature.1 + self.epsilon))
                    .clamp(
                        -self.maximum_illumination_correction,
                        self.maximum_illumination_correction,
                    ),
            );
            let base = model.source_offset(source)?;
            let effective = FourierOffset::new(base.row + candidate.0, base.column + candidate.1);
            if model.validate_source_offset(source, effective).is_ok() {
                *correction = candidate;
            }
        }
        Ok(())
    }
}

fn prepare_illumination_accumulators(
    model: &crate::model::ImagePlaneModel,
    state: &mut ReconstructionState,
) -> Result<()> {
    match &state.illumination_corrections {
        None => {
            state.illumination_corrections = Some(vec![(0.0, 0.0); model.source_count()]);
        }
        Some(corrections) if corrections.len() != model.source_count() => {
            return Err(Error::InvalidModel(
                "illumination correction count does not match source count".into(),
            ));
        }
        Some(_) => {}
    }
    state
        .scratch
        .illumination_gradient
        .resize(model.source_count(), (0.0, 0.0));
    state.scratch.illumination_gradient.fill((0.0, 0.0));
    state
        .scratch
        .illumination_curvature
        .resize(model.source_count(), (0.0, 0.0));
    state.scratch.illumination_curvature.fill((0.0, 0.0));
    state
        .scratch
        .illumination_weight
        .resize(model.source_count(), 0.0);
    state.scratch.illumination_weight.fill(0.0);
    Ok(())
}

fn apply_object_tv(
    state: &mut ReconstructionState,
    shape: (usize, usize),
    weight: f64,
    epsilon: f64,
) -> Result<()> {
    state
        .scratch
        .regularization_field
        .resize(state.object_spectrum.len(), Complex64::default());
    ifftshift_copy(
        state.object_spectrum.as_slice(),
        &mut state.scratch.regularization_field,
        shape,
    );
    state.backend.fft2(
        &mut state.scratch.regularization_field,
        shape,
        FftDirection::Inverse,
        &mut state.scratch.column,
    )?;
    apply_complex_tv_step(
        &mut state.scratch.regularization_field,
        shape,
        weight,
        epsilon,
        &mut state.scratch.object_gradient,
    )?;
    state.backend.fft2(
        &mut state.scratch.regularization_field,
        shape,
        FftDirection::Forward,
        &mut state.scratch.column,
    )?;
    fftshift_copy(
        &state.scratch.regularization_field,
        state.object_spectrum.as_slice_mut(),
        shape,
    );
    Ok(())
}

fn descent_factor(predicted: f64, measured: f64, loss_type: LossType, epsilon: f64) -> f64 {
    match loss_type {
        LossType::AmplitudeMse => 1.0 - measured.max(0.0).sqrt() / predicted.max(epsilon).sqrt(),
        LossType::IntensityMse => 2.0 * (predicted - measured),
        LossType::PoissonNegativeLogLikelihood => 1.0 - measured.max(0.0) / predicted.max(epsilon),
        LossType::HuberAmplitude => {
            let predicted_amplitude = predicted.max(epsilon).sqrt();
            let residual = predicted_amplitude - measured.max(0.0).sqrt();
            residual.clamp(-1.0, 1.0) / (2.0 * predicted_amplitude)
        }
    }
}

fn compute_source_field<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
    source: usize,
    offset: FourierOffset,
) -> Result<()> {
    let model = &problem.model;
    let shape = model.image_shape;
    model.extract_patch_at_offset(
        state.object_spectrum.view(),
        source,
        offset,
        &mut state.scratch.patch,
    )?;
    for pixel in 0..state.scratch.patch.len() {
        state.scratch.exit_spectrum[pixel] =
            state.scratch.patch[pixel] * state.pupil.values.as_slice()[pixel];
    }
    ifftshift_copy(
        &state.scratch.exit_spectrum,
        &mut state.scratch.field,
        shape,
    );
    state.backend.fft2(
        &mut state.scratch.field,
        shape,
        FftDirection::Inverse,
        &mut state.scratch.column,
    )
}

#[derive(Clone, Copy)]
struct IlluminationGradientConfiguration {
    source_weight: f64,
    valid_pixels: usize,
    distance: f64,
    loss_type: LossType,
    epsilon: f64,
}

#[derive(Clone, Copy, Default)]
struct AxisGradient {
    gradient: f64,
    curvature: f64,
}

#[derive(Clone, Copy, Default)]
struct IlluminationGradient {
    row: AxisGradient,
    column: AxisGradient,
}

fn illumination_gradient<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
    source: usize,
    offset: FourierOffset,
    configuration: IlluminationGradientConfiguration,
) -> Result<IlluminationGradient> {
    let row = illumination_axis_gradient(
        problem,
        state,
        source,
        offset,
        FourierOffset::new(configuration.distance, 0.0),
        configuration,
    )?;
    let column = illumination_axis_gradient(
        problem,
        state,
        source,
        offset,
        FourierOffset::new(0.0, configuration.distance),
        configuration,
    )?;
    let pixels = configuration.valid_pixels as f64;
    Ok(IlluminationGradient {
        row: AxisGradient {
            gradient: row.gradient / pixels,
            curvature: row.curvature / pixels,
        },
        column: AxisGradient {
            gradient: column.gradient / pixels,
            curvature: column.curvature / pixels,
        },
    })
}

fn illumination_axis_gradient<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
    source: usize,
    offset: FourierOffset,
    displacement: FourierOffset,
    configuration: IlluminationGradientConfiguration,
) -> Result<AxisGradient> {
    let model = &problem.model;
    let distance = displacement.row.abs() + displacement.column.abs();
    let plus = FourierOffset::new(
        offset.row + displacement.row,
        offset.column + displacement.column,
    );
    let minus = FourierOffset::new(
        offset.row - displacement.row,
        offset.column - displacement.column,
    );
    let plus_valid = model.validate_source_offset(source, plus).is_ok();
    let minus_valid = model.validate_source_offset(source, minus).is_ok();
    if !plus_valid && !minus_valid {
        return Ok(AxisGradient::default());
    }
    if plus_valid {
        compute_source_field(problem, state, source, plus)?;
        for (candidate, field) in state
            .scratch
            .difference
            .iter_mut()
            .zip(&state.scratch.field)
        {
            candidate.re = field.norm_sqr();
        }
    }
    if minus_valid {
        compute_source_field(problem, state, source, minus)?;
    }
    let mut gradient = 0.0;
    let mut curvature = 0.0;
    for pixel in 0..state.scratch.field.len() {
        let derivative = match (plus_valid, minus_valid) {
            (true, true) => {
                (state.scratch.difference[pixel].re - state.scratch.field[pixel].norm_sqr())
                    / (2.0 * distance)
            }
            (true, false) => {
                (state.scratch.difference[pixel].re - state.scratch.calibration_reference[pixel])
                    / distance
            }
            (false, true) => {
                (state.scratch.calibration_reference[pixel] - state.scratch.field[pixel].norm_sqr())
                    / distance
            }
            (false, false) => 0.0,
        };
        let intensity_derivative = configuration.source_weight * derivative;
        let loss_derivative = state.scratch.projected_field[pixel].re;
        let predicted = state.scratch.projected_field[pixel].im;
        gradient += loss_derivative * intensity_derivative;
        curvature += descent_curvature(
            predicted,
            loss_derivative,
            configuration.loss_type,
            configuration.epsilon,
        ) * intensity_derivative
            * intensity_derivative;
    }
    Ok(AxisGradient {
        gradient,
        curvature,
    })
}

fn descent_curvature(
    predicted: f64,
    descent_factor: f64,
    loss_type: LossType,
    epsilon: f64,
) -> f64 {
    let mean = predicted.max(epsilon);
    match loss_type {
        LossType::AmplitudeMse => 0.5 / mean,
        LossType::IntensityMse => 2.0,
        LossType::PoissonNegativeLogLikelihood => 1.0 / mean,
        LossType::HuberAmplitude => {
            let clipped_residual = descent_factor * 2.0 * mean.sqrt();
            if clipped_residual.abs() < 1.0 {
                0.25 / mean
            } else {
                epsilon
            }
        }
    }
}

fn background_value(
    state: &ReconstructionState,
    frame: usize,
    pixel: usize,
    image_len: usize,
) -> f64 {
    state.background.as_ref().map_or(0.0, |values| {
        values[if values.len() == image_len {
            pixel
        } else {
            frame * image_len + pixel
        }]
    })
}
