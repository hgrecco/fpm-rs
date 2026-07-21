use num_complex::Complex64;

use crate::{
    Result,
    algorithms::objective::{LossType, point_loss},
    array_layout::checked_len_2d,
    backend::FftDirection,
    diagnostics::StepDiagnostics,
    error::Error,
    measurements::MeasurementRead,
    model::{FourierOffset, ImagePlaneModel, fftshift_copy, ifftshift_copy},
    reconstruction::{
        AdmmAuxiliaryState, AlgorithmAuxiliaryState, Batch, ReconstructionProblem,
        ReconstructionState,
    },
};

use super::ReconstructionAlgorithm;

/// Linearized ADMM reconstruction for Fourier ptychographic microscopy.
///
/// # Method
///
/// The algorithm splits each predicted detector field from the shared object
/// by introducing an auxiliary field and a scaled dual variable. Each step
/// alternates among a measurement-amplitude proximal update of the auxiliary
/// fields, a pupil-preconditioned linearized update of the common object
/// spectrum, and a scaled-dual update that drives the auxiliary and predicted
/// fields toward consensus. This separates the nonlinear measurement
/// constraint from the overlapping Fourier-patch consistency constraint.
///
/// For incoherently multiplexed data, the amplitude proximal is joint across
/// all source modes in a frame. Auxiliary and scaled-dual fields are stored per
/// frame-source mode in [`ReconstructionState`] for exact checkpoint resumption.
/// The fixed-pupil linearization and multiplexed proximal used here are crate
/// adaptations of the reference ADMM-FPM formulation.
///
/// Each iteration reports detector-field RMS residuals. The primal residual is
/// `r_k = A x_k - z_k`, evaluated after the linearized object update, and the
/// dual residual is `s_k = rho (z_k - z_{k-1})`. Here `A x` denotes the
/// concatenated per-mode detector fields, including each mode of a multiplexed
/// frame. Masked pixels and zero-weight frames are excluded. These residuals
/// diagnose consensus and auxiliary-field motion; the solver does not use them
/// as stopping criteria.
///
/// # Reference
///
/// A. Wang, Z. Zhang, S. Wang, A. Pan, C. Ma, and B. Yao, “Fourier
/// Ptychographic Microscopy via Alternating Direction Method of Multipliers,”
/// *Cells* **11**(9), 1512 (2022),
/// [doi:10.3390/cells11091512](https://doi.org/10.3390/cells11091512).
#[derive(Clone, Debug)]
pub struct Admm {
    /// Number of complete passes through the acquisition schedule.
    pub iterations: usize,
    /// Step size of the linearized, pupil-preconditioned object update.
    pub object_step: f64,
    /// Positive augmented-Lagrangian penalty tying auxiliary fields to the
    /// fields predicted by the shared object.
    pub penalty: f64,
    /// Scaled-dual update relaxation in the inclusive range `0..=2`.
    pub dual_relaxation: f64,
    /// Number of measured frames supplied to each reconstruction step; the
    /// default processes every frame together.
    pub batch_size: usize,
    /// Positive numerical floor used in normalizations and dark-field handling.
    pub epsilon: f64,
}

impl Default for Admm {
    fn default() -> Self {
        Self {
            iterations: 100,
            object_step: 0.8,
            penalty: 1.0,
            dual_relaxation: 1.0,
            batch_size: usize::MAX,
            epsilon: 1e-10,
        }
    }
}

impl Admm {
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn object_step(mut self, step: f64) -> Self {
        self.object_step = step;
        self
    }

    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn dual_relaxation(mut self, relaxation: f64) -> Self {
        self.dual_relaxation = relaxation;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl ReconstructionAlgorithm for Admm {
    fn validate(&self) -> Result<()> {
        if !self.object_step.is_finite() || self.object_step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "object_step",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.penalty.is_finite() || self.penalty <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "penalty",
                reason: "must be finite and positive".into(),
            });
        }
        if !self.dual_relaxation.is_finite() || !(0.0..=2.0).contains(&self.dual_relaxation) {
            return Err(Error::InvalidParameter {
                name: "dual_relaxation",
                reason: "must be finite and between zero and two".into(),
            });
        }
        if self.batch_size == 0 || !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "batch_size/epsilon",
                reason: "batch size must be non-zero and epsilon finite and positive".into(),
            });
        }
        Ok(())
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepDiagnostics> {
        let expected = admm_auxiliary_len(&problem.model)?;
        let mut auxiliary = match state.algorithm_auxiliary.take() {
            None => AdmmAuxiliaryState {
                auxiliary_fields: vec![Complex64::default(); expected],
                dual_fields: vec![Complex64::default(); expected],
            },
            Some(AlgorithmAuxiliaryState::Admm(auxiliary)) => auxiliary,
        };
        let result = self.step_with_auxiliary(problem, state, batch, &mut auxiliary);
        state.algorithm_auxiliary = Some(AlgorithmAuxiliaryState::Admm(auxiliary));
        result
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl Admm {
    fn step_with_auxiliary<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        auxiliary: &mut AdmmAuxiliaryState,
    ) -> Result<StepDiagnostics> {
        let model = &problem.model;
        let shape = model.image_shape;
        let image_len = checked_len_2d(shape)?;
        let expected = admm_auxiliary_len(model)?;
        if auxiliary.auxiliary_fields.len() != expected || auxiliary.dual_fields.len() != expected {
            return Err(Error::InvalidModel(
                "ADMM auxiliary state does not match the problem modes".into(),
            ));
        }
        state
            .scratch
            .object_gradient
            .resize(state.object_spectrum.len(), Complex64::default());
        state.scratch.object_gradient.fill(Complex64::default());
        let maximum_pupil_power = state
            .pupil
            .values
            .as_slice()
            .iter()
            .map(|value| value.norm_sqr())
            .fold(0.0, f64::max)
            .max(self.epsilon);
        let mut diagnostics = StepDiagnostics::default();
        let mut active_frames = 0;

        for &frame in &batch.indices {
            let frame_weight = problem.measurements.frame_weight(frame)?;
            if frame_weight == 0.0 {
                diagnostics.push_frame(frame, 0.0, 0.0);
                continue;
            }
            active_frames += 1;
            let single_source = [(frame, 1.0)];
            let sources = frame_sources(model, frame, &single_source);
            let source_weight_sum: f64 = sources.iter().map(|&(_, weight)| weight).sum();
            let mode_start = frame_mode_start(model, frame);
            let multiplex_len =
                image_len
                    .checked_mul(sources.len())
                    .ok_or_else(|| Error::ShapeOverflow {
                        shape: vec![sources.len(), shape.0, shape.1],
                    })?;
            state
                .scratch
                .multiplex_fields
                .resize(multiplex_len, Complex64::default());
            state.scratch.multiplex_offsets.clear();
            state.scratch.projected_field.fill(Complex64::default());

            // The real component accumulates the physical prediction; the
            // imaginary component accumulates the dual-shifted proximal norm.
            for (local_mode, &(source, source_weight)) in sources.iter().enumerate() {
                let offset = state.effective_source_offset(model, source)?;
                state.scratch.multiplex_offsets.push(offset);
                compute_source_field(problem, state, source, offset)?;
                let local_start = local_mode * image_len;
                state.scratch.multiplex_fields[local_start..local_start + image_len]
                    .copy_from_slice(&state.scratch.field);
                let auxiliary_start = (mode_start + local_mode) * image_len;
                for pixel in 0..image_len {
                    let field = state.scratch.field[pixel];
                    let consensus = field + auxiliary.dual_fields[auxiliary_start + pixel];
                    state.scratch.projected_field[pixel].re += source_weight * field.norm_sqr();
                    state.scratch.projected_field[pixel].im += source_weight * consensus.norm_sqr();
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
                    continue;
                }
                valid_pixels += 1;
                let background = background_value(state, frame, pixel, image_len);
                let predicted = gain * state.scratch.projected_field[pixel].re + background;
                frame_loss += point_loss(predicted, measured[pixel], LossType::AmplitudeMse);
            }
            if valid_pixels == 0 {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {frame} has no unmasked pixels"
                )));
            }
            diagnostics.push_frame(frame, frame_loss / valid_pixels as f64, frame_weight);

            for (local_mode, &(source, source_weight)) in sources.iter().enumerate() {
                let local_start = local_mode * image_len;
                state.scratch.field.copy_from_slice(
                    &state.scratch.multiplex_fields[local_start..local_start + image_len],
                );
                let auxiliary_start = (mode_start + local_mode) * image_len;
                for pixel in 0..image_len {
                    let index = auxiliary_start + pixel;
                    if mask.is_some_and(|values| values[pixel] == 0) {
                        auxiliary.auxiliary_fields[index] = state.scratch.field[pixel];
                        auxiliary.dual_fields[index] = Complex64::default();
                        state.scratch.difference[pixel] = Complex64::default();
                        continue;
                    }
                    let background = background_value(state, frame, pixel, image_len);
                    let target = ((measured[pixel] - background) / gain).max(0.0).sqrt();
                    let consensus = state.scratch.field[pixel] + auxiliary.dual_fields[index];
                    let consensus_norm = state.scratch.projected_field[pixel].im;
                    let projected = if consensus_norm > self.epsilon {
                        consensus * (target / consensus_norm.sqrt())
                    } else {
                        Complex64::new(target / source_weight_sum.sqrt(), 0.0)
                    };
                    let auxiliary_value = (self.penalty * consensus + frame_weight * projected)
                        / (self.penalty + frame_weight);
                    diagnostics.push_admm_dual_change(
                        auxiliary_value - auxiliary.auxiliary_fields[index],
                        self.penalty,
                    );
                    auxiliary.auxiliary_fields[index] = auxiliary_value;
                    state.scratch.difference[pixel] =
                        auxiliary_value - auxiliary.dual_fields[index] - state.scratch.field[pixel];
                }
                state.backend.fft2(
                    &mut state.scratch.difference,
                    shape,
                    FftDirection::Forward,
                    &mut state.scratch.column,
                )?;
                fftshift_copy(
                    &state.scratch.difference,
                    &mut state.scratch.projected_spectrum,
                    shape,
                );
                for pixel in 0..image_len {
                    state.scratch.difference[pixel] = state.pupil.values.as_slice()[pixel].conj()
                        * state.scratch.projected_spectrum[pixel]
                        / (maximum_pupil_power + self.epsilon);
                }
                model.insert_patch_adjoint_slice_at_offset(
                    &mut state.scratch.object_gradient,
                    source,
                    &state.scratch.difference,
                    source_weight / source_weight_sum,
                    state.scratch.multiplex_offsets[local_mode],
                )?;
            }
        }

        if active_frames > 0 {
            let step = self.object_step / active_frames as f64;
            for (object, &update) in state
                .object_spectrum
                .as_slice_mut()
                .iter_mut()
                .zip(&state.scratch.object_gradient)
            {
                *object += step * update;
            }
        }

        for &frame in &batch.indices {
            if problem.measurements.frame_weight(frame)? == 0.0 {
                continue;
            }
            let single_source = [(frame, 1.0)];
            let sources = frame_sources(model, frame, &single_source);
            let mode_start = frame_mode_start(model, frame);
            let mask = problem.measurements.frame_mask(frame)?;
            for (local_mode, &(source, _)) in sources.iter().enumerate() {
                let offset = state.effective_source_offset(model, source)?;
                compute_source_field(problem, state, source, offset)?;
                let auxiliary_start = (mode_start + local_mode) * image_len;
                for pixel in 0..image_len {
                    let index = auxiliary_start + pixel;
                    if mask.is_some_and(|values| values[pixel] == 0) {
                        auxiliary.auxiliary_fields[index] = state.scratch.field[pixel];
                        auxiliary.dual_fields[index] = Complex64::default();
                    } else {
                        let primal_residual =
                            state.scratch.field[pixel] - auxiliary.auxiliary_fields[index];
                        diagnostics.push_admm_primal_residual(primal_residual);
                        auxiliary.dual_fields[index] += self.dual_relaxation * primal_residual;
                    }
                }
            }
        }
        Ok(diagnostics)
    }
}

pub(crate) fn admm_auxiliary_len(model: &ImagePlaneModel) -> Result<usize> {
    let mode_count = model.multiplexing_matrix.as_ref().map_or_else(
        || Ok(model.frame_count()),
        |matrix| {
            matrix.iter().try_fold(0_usize, |count, row| {
                count
                    .checked_add(row.len())
                    .ok_or_else(|| Error::InvalidShape("ADMM source mode count overflows".into()))
            })
        },
    )?;
    checked_len_2d(model.image_shape)?
        .checked_mul(mode_count)
        .ok_or_else(|| Error::InvalidShape("ADMM auxiliary length overflows".into()))
}

fn frame_mode_start(model: &ImagePlaneModel, frame: usize) -> usize {
    model
        .multiplexing_matrix
        .as_ref()
        .map_or(frame, |matrix| matrix[..frame].iter().map(Vec::len).sum())
}

fn frame_sources<'a>(
    model: &'a ImagePlaneModel,
    frame: usize,
    single_source: &'a [(usize, f64); 1],
) -> &'a [(usize, f64)] {
    model
        .multiplexing_matrix
        .as_ref()
        .map_or(single_source, |matrix| matrix[frame].as_slice())
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
