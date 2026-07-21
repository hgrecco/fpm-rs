use num_complex::Complex64;

use crate::{
    Result,
    algorithms::objective::{LossType, point_loss},
    array_layout::checked_len_2d,
    backend::FftDirection,
    diagnostics::StepDiagnostics,
    error::Error,
    measurements::MeasurementRead,
    model::{fftshift_copy, ifftshift_copy},
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ObjectDenominator {
    Local,
    Rpie(f64),
    Global,
}

pub(crate) struct UpdateConfiguration {
    pub object_step: f64,
    pub pupil_step: Option<f64>,
    pub epsilon: f64,
    pub loss_type: LossType,
    pub object_denominator: ObjectDenominator,
    pub constrain_pupil: bool,
    pub gain_update: Option<GainUpdateConfiguration>,
    pub background_update: Option<BackgroundUpdateConfiguration>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GainUpdateConfiguration {
    pub step: f64,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BackgroundUpdateConfiguration {
    pub step: f64,
    pub minimum: f64,
    pub maximum: f64,
}

pub(crate) fn projection_update<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
    batch: &Batch,
    configuration: UpdateConfiguration,
) -> Result<StepDiagnostics> {
    let model = &problem.model;
    let shape = model.image_shape;
    let image_len = checked_len_2d(shape)?;
    let mut diagnostics = StepDiagnostics::default();
    for &frame in &batch.indices {
        let frame_weight = problem.measurements.frame_weight(frame)?;
        if frame_weight == 0.0 {
            diagnostics.push_frame(frame, 0.0, 0.0);
            continue;
        }
        let single_source = [(frame, 1.0)];
        let sources: &[(usize, f64)] = match &model.multiplexing_matrix {
            Some(matrix) => &matrix[frame],
            None => &single_source,
        };
        let source_weight_sum: f64 = sources.iter().map(|&(_, weight)| weight).sum();
        let multiplex_len = image_len.checked_mul(sources.len()).ok_or_else(|| {
            Error::InvalidShape("multiplexed projection scratch length overflows".into())
        })?;
        state
            .scratch
            .multiplex_fields
            .resize(multiplex_len, Complex64::default());
        state
            .scratch
            .multiplex_patches
            .resize(multiplex_len, Complex64::default());
        state.scratch.multiplex_offsets.clear();
        state.scratch.calibration_reference.fill(0.0);

        // All modes are evaluated from the same pre-frame object and pupil.
        for (mode, &(source, source_weight)) in sources.iter().enumerate() {
            let offset = state.effective_source_offset(model, source)?;
            state.scratch.multiplex_offsets.push(offset);
            model.extract_patch_at_offset(
                state.object_spectrum.view(),
                source,
                offset,
                &mut state.scratch.patch,
            )?;
            let start = mode * image_len;
            state.scratch.multiplex_patches[start..start + image_len]
                .copy_from_slice(&state.scratch.patch);
            for pixel in 0..image_len {
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
            )?;
            state.scratch.multiplex_fields[start..start + image_len]
                .copy_from_slice(&state.scratch.field);
            for (prediction, field) in state
                .scratch
                .calibration_reference
                .iter_mut()
                .zip(&state.scratch.field)
            {
                *prediction += source_weight * field.norm_sqr();
            }
        }

        let measured = problem.measurements.frame(frame)?;
        let mask = problem.measurements.frame_mask(frame)?;
        let mut gain = state
            .frame_gains
            .as_ref()
            .map_or(1.0, |values| values[frame]);
        if !gain.is_finite() || gain <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "state frame {frame} has invalid gain {gain}"
            )));
        }
        if let Some(gain_update) = configuration.gain_update {
            update_gain(
                state,
                frame,
                &measured,
                mask,
                image_len,
                model.frame_count(),
                &mut gain,
                gain_update,
                configuration.epsilon,
            );
        }
        if let Some(background_update) = configuration.background_update {
            update_background(
                state,
                frame,
                &measured,
                mask,
                image_len,
                model.frame_count(),
                gain,
                background_update,
            )?;
        }

        let mut frame_loss = 0.0;
        let mut valid_pixels = 0;
        for pixel in 0..image_len {
            let intrinsic_prediction = state.scratch.calibration_reference[pixel];
            if mask.is_some_and(|values| values[pixel] == 0) {
                state.scratch.projected_field[pixel] = Complex64::new(1.0, 0.0);
                continue;
            }
            valid_pixels += 1;
            let background = background_value(state, frame, pixel, image_len);
            let predicted = gain * intrinsic_prediction + background;
            frame_loss += point_loss(predicted, measured[pixel], configuration.loss_type);
            let target_intensity = ((measured[pixel] - background) / gain).max(0.0);
            state.scratch.projected_field[pixel] = if intrinsic_prediction > configuration.epsilon {
                Complex64::new((target_intensity / intrinsic_prediction).sqrt(), 0.0)
            } else {
                // Store a real-axis fallback amplitude in the imaginary
                // component when every predicted mode is dark.
                Complex64::new(0.0, (target_intensity / source_weight_sum).sqrt())
            };
        }
        if valid_pixels == 0 {
            return Err(Error::InvalidMeasurements(format!(
                "frame {frame} has no unmasked pixels"
            )));
        }
        diagnostics.push_frame(frame, frame_loss / valid_pixels as f64, frame_weight);

        let maximum_pupil_power = state
            .pupil
            .values
            .as_slice()
            .iter()
            .map(|value| value.norm_sqr())
            .fold(0.0, f64::max)
            .max(configuration.epsilon);
        if configuration.pupil_step.is_some() {
            state.scratch.pupil_gradient.fill(Complex64::default());
        }
        for (mode, &(source, source_weight)) in sources.iter().enumerate() {
            let start = mode * image_len;
            state
                .scratch
                .patch
                .copy_from_slice(&state.scratch.multiplex_patches[start..start + image_len]);
            state
                .scratch
                .field
                .copy_from_slice(&state.scratch.multiplex_fields[start..start + image_len]);
            for pixel in 0..image_len {
                let projection = state.scratch.projected_field[pixel];
                state.scratch.difference[pixel] = if mask.is_some_and(|values| values[pixel] == 0) {
                    state.scratch.field[pixel]
                } else if state.scratch.calibration_reference[pixel] > configuration.epsilon {
                    state.scratch.field[pixel] * projection.re
                } else {
                    Complex64::new(projection.im, 0.0)
                };
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
                let original_exit =
                    state.scratch.patch[pixel] * state.pupil.values.as_slice()[pixel];
                state.scratch.difference[pixel] =
                    state.scratch.projected_spectrum[pixel] - original_exit;
            }
            let normalized_source_weight = source_weight / source_weight_sum;
            for pixel in 0..image_len {
                let pupil = state.pupil.values.as_slice()[pixel];
                let local_power = pupil.norm_sqr();
                let denominator = match configuration.object_denominator {
                    ObjectDenominator::Local => local_power + configuration.epsilon,
                    ObjectDenominator::Global => maximum_pupil_power + configuration.epsilon,
                    ObjectDenominator::Rpie(alpha) => {
                        (1.0 - alpha) * local_power
                            + alpha * maximum_pupil_power
                            + configuration.epsilon
                    }
                };
                state.scratch.exit_spectrum[pixel] =
                    pupil.conj() * state.scratch.difference[pixel] / denominator;
            }
            let offset = state.scratch.multiplex_offsets[mode];
            model.insert_patch_adjoint_at_offset(
                state.object_spectrum.view_mut(),
                source,
                &state.scratch.exit_spectrum,
                frame_weight * configuration.object_step * normalized_source_weight,
                offset,
            )?;

            if configuration.pupil_step.is_some() {
                let maximum_object_power = state
                    .scratch
                    .patch
                    .iter()
                    .map(|value| value.norm_sqr())
                    .fold(0.0, f64::max)
                    .max(configuration.epsilon);
                for pixel in 0..image_len {
                    state.scratch.pupil_gradient[pixel] += normalized_source_weight
                        * state.scratch.patch[pixel].conj()
                        * state.scratch.difference[pixel]
                        / (maximum_object_power + configuration.epsilon);
                }
            }
        }
        if let Some(pupil_step) = configuration.pupil_step {
            for (pupil, &gradient) in state
                .pupil
                .values
                .as_slice_mut()
                .iter_mut()
                .zip(&state.scratch.pupil_gradient)
            {
                *pupil += frame_weight * pupil_step * gradient;
            }
            if configuration.constrain_pupil {
                state.pupil.apply_support();
            }
        }
    }
    Ok(diagnostics)
}

#[allow(clippy::too_many_arguments)]
fn update_gain(
    state: &mut ReconstructionState,
    frame: usize,
    measured: &[f64],
    mask: Option<&[u8]>,
    image_len: usize,
    frame_count: usize,
    gain: &mut f64,
    configuration: GainUpdateConfiguration,
    epsilon: f64,
) {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pixel in 0..image_len {
        if mask.is_some_and(|values| values[pixel] == 0) {
            continue;
        }
        let background = background_value(state, frame, pixel, image_len);
        let predicted = state.scratch.calibration_reference[pixel];
        numerator += predicted * (measured[pixel] - background).max(0.0);
        denominator += predicted * predicted;
    }
    if denominator > epsilon {
        let estimate =
            (numerator / denominator).clamp(configuration.minimum, configuration.maximum);
        *gain = ((1.0 - configuration.step) * *gain + configuration.step * estimate)
            .clamp(configuration.minimum, configuration.maximum);
        let gains = state
            .frame_gains
            .get_or_insert_with(|| vec![1.0; frame_count]);
        gains[frame] = *gain;
    }
}

#[allow(clippy::too_many_arguments)]
fn update_background(
    state: &mut ReconstructionState,
    frame: usize,
    measured: &[f64],
    mask: Option<&[u8]>,
    image_len: usize,
    frame_count: usize,
    gain: f64,
    configuration: BackgroundUpdateConfiguration,
) -> Result<()> {
    let mut residual_sum = 0.0;
    let mut residual_count = 0;
    for pixel in 0..image_len {
        if mask.is_some_and(|values| values[pixel] == 0) {
            continue;
        }
        residual_sum += measured[pixel]
            - gain * state.scratch.calibration_reference[pixel]
            - background_value(state, frame, pixel, image_len);
        residual_count += 1;
    }
    if residual_count == 0 {
        return Ok(());
    }
    let correction = configuration.step * residual_sum / residual_count as f64;
    let stack_len = image_len
        .checked_mul(frame_count)
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![frame_count, image_len],
        })?;
    let mut background = match state.background.take() {
        None => vec![0.0; stack_len],
        Some(values) if values.len() == image_len => {
            let mut expanded = Vec::with_capacity(stack_len);
            for _ in 0..frame_count {
                expanded.extend_from_slice(&values);
            }
            expanded
        }
        Some(values) if values.len() == stack_len => values,
        Some(_) => {
            return Err(Error::InvalidModel(
                "state background length is inconsistent with the model".into(),
            ));
        }
    };
    let start = frame
        .checked_mul(image_len)
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![frame, image_len],
        })?;
    let end = start
        .checked_add(image_len)
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![frame.saturating_add(1), image_len],
        })?;
    let frame_background = background
        .get_mut(start..end)
        .ok_or_else(|| Error::InvalidModel("state background frame is out of range".into()))?;
    for value in frame_background {
        *value = (*value + correction).clamp(configuration.minimum, configuration.maximum);
    }
    state.background = Some(background);
    Ok(())
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
