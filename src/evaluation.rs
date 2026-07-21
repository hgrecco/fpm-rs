//! Composed evaluation of a reconstruction against reference data.

use ndarray::ArrayView2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    array_layout::checked_len_2d,
    measurements::MeasurementRead,
    metrics::{
        complex_field::{ComplexFieldComparisonMetrics, compare_complex_fields_masked},
        intensity::{IntensityComparisonMetrics, compare_intensity_u8_masked},
        model::{
            FrameGainComparisonMetrics, IlluminationPositionMetrics, PupilComparisonMetrics,
            compare_frame_gains, compare_illumination_positions, compare_pupils,
        },
    },
    model::{ForwardModel, FourierOffset, ImagePlaneModel},
    reconstruction::{ReconstructionProblem, ReconstructionResult},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameIntensityEvaluation {
    pub per_frame: Vec<IntensityComparisonMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionEvaluation {
    pub object: ComplexFieldComparisonMetrics,
    pub pupil: Option<PupilComparisonMetrics>,
    pub illumination: Option<IlluminationPositionMetrics>,
    pub frame_gains: Option<FrameGainComparisonMetrics>,
    pub intensity: Option<FrameIntensityEvaluation>,
}

pub fn evaluate_reconstruction(
    result: &ReconstructionResult,
    reference_object: ArrayView2<'_, Complex64>,
    reference_model: Option<&ImagePlaneModel>,
    valid_object_mask: Option<ArrayView2<'_, u8>>,
) -> Result<ReconstructionEvaluation> {
    let object =
        compare_complex_fields_masked(reference_object, result.object.view(), valid_object_mask)?;
    let (pupil, illumination, frame_gains) = if let Some(reference_model) = reference_model {
        let pupil = Some(compare_pupils(
            reference_model.pupil.values(),
            result.recovered_pupil.values(),
            reference_model.pupil.support(),
        )?);
        let illumination = None;
        let frame_gains = result
            .recovered_frame_gains
            .as_ref()
            .map(|candidate| {
                let reference = reference_model.frame_gains.as_deref().unwrap_or(&[]);
                if reference.is_empty() {
                    compare_frame_gains(&vec![1.0; reference_model.frame_count()], candidate)
                } else {
                    compare_frame_gains(reference, candidate)
                }
            })
            .transpose()?;
        (pupil, illumination, frame_gains)
    } else {
        (None, None, None)
    };
    Ok(ReconstructionEvaluation {
        object,
        pupil,
        illumination,
        frame_gains,
        intensity: None,
    })
}

pub fn evaluate_reconstruction_with_problem<M: MeasurementRead>(
    result: &ReconstructionResult,
    problem: &ReconstructionProblem<M>,
    reference_object: ArrayView2<'_, Complex64>,
    reference_model: Option<&ImagePlaneModel>,
    valid_object_mask: Option<ArrayView2<'_, u8>>,
) -> Result<ReconstructionEvaluation> {
    let mut evaluation =
        evaluate_reconstruction(result, reference_object, reference_model, valid_object_mask)?;
    problem.validate()?;
    let intensity = evaluate_frame_intensity(result, problem)?;
    if let Some(reference_model) = reference_model {
        evaluation.illumination = Some(compare_illumination_positions(
            reference_model,
            &problem.model,
            result.calibrated_illumination.as_deref(),
        )?);
    }
    evaluation.intensity = Some(intensity);
    Ok(evaluation)
}

pub fn evaluate_frame_intensity<M: MeasurementRead>(
    result: &ReconstructionResult,
    problem: &ReconstructionProblem<M>,
) -> Result<FrameIntensityEvaluation> {
    problem.validate()?;
    let model = model_with_result_calibration(problem, result)?;
    let forward = ForwardModel::new(&model)?;
    let mut workspace = forward.workspace()?;
    let mut candidate = vec![0.0; checked_len_2d(model.image_shape)?];
    let mut per_frame = Vec::with_capacity(model.frame_count());
    for frame in 0..model.frame_count() {
        if problem.measurements.frame_weight(frame)? == 0.0 {
            per_frame.push(compare_intensity_u8_masked(
                &candidate, &candidate, None, None,
            )?);
            continue;
        }
        forward.forward_intensity_into(
            result.object_spectrum.view(),
            &result.recovered_pupil,
            frame,
            &mut workspace,
            &mut candidate,
        )?;
        let reference = problem.measurements.frame(frame)?;
        per_frame.push(compare_intensity_u8_masked(
            &reference,
            &candidate,
            problem.measurements.frame_mask(frame)?,
            None,
        )?);
    }
    Ok(FrameIntensityEvaluation { per_frame })
}

fn model_with_result_calibration<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    result: &ReconstructionResult,
) -> Result<ImagePlaneModel> {
    let mut model = problem.model.clone();
    model.frame_gains = result.recovered_frame_gains.clone();
    model.background = result.recovered_background.clone();
    if let Some(corrections) = &result.calibrated_illumination {
        model.subpixel_offsets = Some(
            corrections
                .iter()
                .enumerate()
                .map(|(source, &(row, column))| {
                    let base = problem.model.source_offset(source)?;
                    Ok(FourierOffset::new(base.row + row, base.column + column))
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    model.validate()?;
    Ok(model)
}
