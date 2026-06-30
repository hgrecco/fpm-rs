use num_complex::Complex64;

use crate::{
    Array2, Result,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
    measurements::MeasurementRead,
    model::{ForwardModel, FourierOffset, ImagePlaneModel, fftshift_copy},
    reconstruction::{ReconstructionProblem, ReconstructionResult},
};

#[derive(Clone, Debug, Default)]
pub struct GroundTruthMetrics {
    pub amplitude_rmse: f64,
    pub phase_rmse: f64,
    pub complex_field_error: f64,
    pub fourier_domain_error: f64,
    pub pupil_amplitude_error: Option<f64>,
    pub pupil_phase_error: Option<f64>,
    pub frame_gain_relative_error: Option<f64>,
    /// Fourier-grid pixel RMSE, available from [`compare_with_problem`] when a
    /// true model is supplied.
    pub illumination_position_rmse: Option<f64>,
    pub per_frame_residuals: Option<Vec<f64>>,
    pub global_phase_offset: f64,
}

pub fn compare_to_ground_truth(
    result: &ReconstructionResult,
    ground_truth: &Array2<Complex64>,
) -> Result<GroundTruthMetrics> {
    compare(result, ground_truth, None)
}

pub fn compare_with_true_model(
    result: &ReconstructionResult,
    ground_truth: &Array2<Complex64>,
    true_model: &ImagePlaneModel,
) -> Result<GroundTruthMetrics> {
    compare(result, ground_truth, Some(true_model))
}

/// Ground-truth metrics plus normalized intensity residuals against a problem's
/// measured frames. Masks are honored and zero-weight frames report zero.
pub fn compare_with_problem<M: MeasurementRead>(
    result: &ReconstructionResult,
    problem: &ReconstructionProblem<M>,
    ground_truth: &Array2<Complex64>,
    true_model: Option<&ImagePlaneModel>,
) -> Result<GroundTruthMetrics> {
    problem.validate()?;
    let mut metrics = compare(result, ground_truth, true_model)?;
    let mut recovered_model = problem.model.clone();
    recovered_model.frame_gains = result.recovered_frame_gains.clone();
    recovered_model.background = result.recovered_background.clone();
    if let Some(corrections) = &result.calibrated_illumination {
        if corrections.len() != recovered_model.source_count()
            || corrections
                .iter()
                .any(|&(row, column)| !row.is_finite() || !column.is_finite())
        {
            return Err(Error::InvalidModel(
                "recovered illumination corrections are invalid for the model".into(),
            ));
        }
        recovered_model.subpixel_offsets = Some(
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
    recovered_model.validate()?;
    metrics.illumination_position_rmse = true_model
        .map(|model| illumination_position_rmse(result, &problem.model, model))
        .transpose()?;
    let forward = ForwardModel::new(&recovered_model)?;
    let mut workspace = forward.workspace();
    let mut predicted = vec![0.0; recovered_model.image_shape.0 * recovered_model.image_shape.1];
    let mut residuals = Vec::with_capacity(recovered_model.frame_count());
    for frame in 0..recovered_model.frame_count() {
        if problem.measurements.frame_weight(frame)? == 0.0 {
            residuals.push(0.0);
            continue;
        }
        forward.forward_intensity_into(
            &result.object_spectrum,
            &result.recovered_pupil,
            frame,
            &mut workspace,
            &mut predicted,
        )?;
        let measured = problem.measurements.frame(frame)?;
        let mask = problem.measurements.frame_mask(frame)?;
        let mut squared_error = 0.0;
        let mut squared_measurement = 0.0;
        for pixel in 0..predicted.len() {
            if mask.is_none_or(|values| values[pixel] != 0) {
                squared_error += (predicted[pixel] - measured[pixel]).powi(2);
                squared_measurement += measured[pixel] * measured[pixel];
            }
        }
        residuals.push((squared_error / squared_measurement.max(f64::EPSILON)).sqrt());
    }
    metrics.per_frame_residuals = Some(residuals);
    Ok(metrics)
}

fn compare(
    result: &ReconstructionResult,
    ground_truth: &Array2<Complex64>,
    true_model: Option<&ImagePlaneModel>,
) -> Result<GroundTruthMetrics> {
    if result.object.shape() != ground_truth.shape() || ground_truth.is_empty() {
        return Err(Error::InvalidShape(format!(
            "reconstruction shape {:?} differs from ground truth {:?}",
            result.object.shape(),
            ground_truth.shape()
        )));
    }
    let cross: Complex64 = result
        .object
        .as_slice()
        .iter()
        .zip(ground_truth.as_slice())
        .map(|(&reconstructed, &truth)| reconstructed * truth.conj())
        .sum();
    let phase_offset = cross.arg();
    let correction = Complex64::from_polar(1.0, -phase_offset);
    let mut amplitude_squared = 0.0;
    let mut phase_squared = 0.0;
    let mut complex_squared = 0.0;
    let mut truth_squared = 0.0;
    for (&reconstructed, &truth) in result.object.as_slice().iter().zip(ground_truth.as_slice()) {
        let aligned = reconstructed * correction;
        amplitude_squared += (aligned.norm() - truth.norm()).powi(2);
        let phase_error = wrap_phase(aligned.arg() - truth.arg());
        phase_squared += phase_error * phase_error;
        complex_squared += (aligned - truth).norm_sqr();
        truth_squared += truth.norm_sqr();
    }
    let count = ground_truth.len() as f64;
    let truth_spectrum = spectrum(ground_truth, result.object_spectrum.shape())?;
    let fourier_squared: f64 = result
        .object_spectrum
        .as_slice()
        .iter()
        .zip(truth_spectrum.as_slice())
        .map(|(&reconstructed, &truth)| (reconstructed * correction - truth).norm_sqr())
        .sum();
    let truth_fourier_squared: f64 = truth_spectrum
        .as_slice()
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let (pupil_amplitude_error, pupil_phase_error) = true_model.map_or((None, None), |model| {
        // A global complex scale is jointly ambiguous between object and pupil.
        // Align the recovered pupil by its least-squares scalar first.
        let (alignment_numerator, alignment_denominator) = result
            .recovered_pupil
            .values
            .as_slice()
            .iter()
            .zip(model.pupil.values.as_slice())
            .zip(&model.pupil.support)
            .filter(|&(_, &inside)| inside)
            .fold(
                (Complex64::default(), 0.0),
                |(numerator, denominator), ((&recovered, &truth), _)| {
                    (
                        numerator + recovered.conj() * truth,
                        denominator + recovered.norm_sqr(),
                    )
                },
            );
        let pupil_alignment = if alignment_denominator > f64::EPSILON {
            alignment_numerator / alignment_denominator
        } else {
            Complex64::new(1.0, 0.0)
        };
        let mut amplitude = 0.0;
        let mut phase = 0.0;
        let mut supported = 0;
        for ((&recovered, &truth), &inside) in result
            .recovered_pupil
            .values
            .as_slice()
            .iter()
            .zip(model.pupil.values.as_slice())
            .zip(&model.pupil.support)
        {
            if inside {
                let aligned = recovered * pupil_alignment;
                amplitude += (aligned.norm() - truth.norm()).powi(2);
                phase += wrap_phase(aligned.arg() - truth.arg()).powi(2);
                supported += 1;
            }
        }
        if supported == 0 {
            (None, None)
        } else {
            (
                Some((amplitude / supported as f64).sqrt()),
                Some((phase / supported as f64).sqrt()),
            )
        }
    });
    let frame_gain_relative_error =
        if let (Some(model), Some(recovered)) = (true_model, &result.recovered_frame_gains) {
            if recovered.len() != model.frame_count()
                || recovered
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::InvalidModel(
                    "recovered frame gains are invalid for the true model".into(),
                ));
            }
            let truth = model
                .frame_gains
                .clone()
                .unwrap_or_else(|| vec![1.0; model.frame_count()]);
            let numerator: f64 = recovered
                .iter()
                .zip(&truth)
                .map(|(&recovered, &truth)| recovered * truth)
                .sum();
            let denominator: f64 = recovered.iter().map(|value| value * value).sum();
            let alignment = numerator / denominator.max(f64::EPSILON);
            let squared_error: f64 = recovered
                .iter()
                .zip(&truth)
                .map(|(&recovered, &truth)| (alignment * recovered - truth).powi(2))
                .sum();
            let truth_power: f64 = truth.iter().map(|value| value * value).sum();
            Some((squared_error / truth_power.max(f64::EPSILON)).sqrt())
        } else {
            None
        };
    Ok(GroundTruthMetrics {
        amplitude_rmse: (amplitude_squared / count).sqrt(),
        phase_rmse: (phase_squared / count).sqrt(),
        complex_field_error: (complex_squared / truth_squared.max(f64::EPSILON)).sqrt(),
        fourier_domain_error: (fourier_squared / truth_fourier_squared.max(f64::EPSILON)).sqrt(),
        pupil_amplitude_error,
        pupil_phase_error,
        frame_gain_relative_error,
        illumination_position_rmse: None,
        per_frame_residuals: None,
        global_phase_offset: phase_offset,
    })
}

fn illumination_position_rmse(
    result: &ReconstructionResult,
    reconstruction_model: &ImagePlaneModel,
    true_model: &ImagePlaneModel,
) -> Result<f64> {
    if reconstruction_model.source_count() != true_model.source_count() {
        return Err(Error::InvalidModel(
            "true and reconstruction models have different source counts".into(),
        ));
    }
    let corrections = result.calibrated_illumination.as_deref();
    if corrections.is_some_and(|values| {
        values.len() != reconstruction_model.source_count()
            || values
                .iter()
                .any(|&(row, column)| !row.is_finite() || !column.is_finite())
    }) {
        return Err(Error::InvalidModel(
            "recovered illumination corrections are invalid for the model".into(),
        ));
    }
    let mut squared_error = 0.0;
    for source in 0..reconstruction_model.source_count() {
        let recovered_crop = reconstruction_model.crop_indices.get(source)?;
        let recovered_offset = reconstruction_model.source_offset(source)?;
        let correction = corrections.map_or((0.0, 0.0), |values| values[source]);
        let true_crop = true_model.crop_indices.get(source)?;
        let true_offset = true_model.source_offset(source)?;
        let row_error = recovered_crop.start_row as f64 + recovered_offset.row + correction.0
            - true_crop.start_row as f64
            - true_offset.row;
        let column_error = recovered_crop.start_col as f64 + recovered_offset.column + correction.1
            - true_crop.start_col as f64
            - true_offset.column;
        squared_error += row_error * row_error + column_error * column_error;
    }
    Ok((squared_error / reconstruction_model.source_count() as f64).sqrt())
}

fn spectrum(
    object: &Array2<Complex64>,
    expected_shape: (usize, usize),
) -> Result<Array2<Complex64>> {
    if object.shape() != expected_shape {
        return Err(Error::InvalidShape(
            "ground truth and expected spectrum shapes differ".into(),
        ));
    }
    let backend = CpuBackend::new(expected_shape, expected_shape)?;
    let mut values = object.as_slice().to_vec();
    let mut column = vec![Complex64::default(); expected_shape.0];
    backend.fft2(
        &mut values,
        expected_shape,
        FftDirection::Forward,
        &mut column,
    )?;
    let mut centered = vec![Complex64::default(); values.len()];
    fftshift_copy(&values, &mut centered, expected_shape);
    Array2::from_vec(expected_shape, centered)
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
