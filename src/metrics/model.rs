//! Atomic comparisons for calibrated optical-model quantities.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{Array2, Result, error::Error, model::ImagePlaneModel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PupilComparisonMetrics {
    pub amplitude_rmse: f64,
    pub phase_rmse: f64,
}

pub fn compare_pupils(
    reference: &Array2<Complex64>,
    candidate: &Array2<Complex64>,
    support: &[bool],
) -> Result<PupilComparisonMetrics> {
    if reference.shape() != candidate.shape() || support.len() != reference.len() {
        return Err(Error::InvalidShape(
            "pupil comparison inputs have incompatible shapes".into(),
        ));
    }
    let (numerator, denominator) = candidate
        .as_slice()
        .iter()
        .zip(reference.as_slice())
        .zip(support)
        .filter(|&(_, &inside)| inside)
        .fold(
            (Complex64::default(), 0.0),
            |(numerator, denominator), ((&candidate, &reference), _)| {
                (
                    numerator + candidate.conj() * reference,
                    denominator + candidate.norm_sqr(),
                )
            },
        );
    let alignment = if denominator > f64::EPSILON {
        numerator / denominator
    } else {
        Complex64::new(1.0, 0.0)
    };
    let mut amplitude = 0.0;
    let mut phase = 0.0;
    let mut count = 0usize;
    for ((&reference, &candidate), &inside) in reference
        .as_slice()
        .iter()
        .zip(candidate.as_slice())
        .zip(support)
    {
        if !inside {
            continue;
        }
        let candidate = candidate * alignment;
        amplitude += (candidate.norm() - reference.norm()).powi(2);
        phase += wrap_phase(candidate.arg() - reference.arg()).powi(2);
        count += 1;
    }
    if count == 0 {
        return Err(Error::InvalidParameter {
            name: "support",
            reason: "must select at least one pupil sample".into(),
        });
    }
    Ok(PupilComparisonMetrics {
        amplitude_rmse: (amplitude / count as f64).sqrt(),
        phase_rmse: (phase / count as f64).sqrt(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IlluminationPositionMetrics {
    pub position_rmse: f64,
}

pub fn compare_illumination_positions(
    reference: &ImagePlaneModel,
    candidate: &ImagePlaneModel,
    candidate_corrections: Option<&[(f64, f64)]>,
) -> Result<IlluminationPositionMetrics> {
    if reference.source_count() != candidate.source_count() {
        return Err(Error::InvalidModel(
            "reference and candidate models have different source counts".into(),
        ));
    }
    if candidate_corrections.is_some_and(|values| {
        values.len() != candidate.source_count()
            || values
                .iter()
                .any(|&(row, column)| !row.is_finite() || !column.is_finite())
    }) {
        return Err(Error::InvalidModel(
            "candidate illumination corrections are invalid".into(),
        ));
    }
    let mut squared = 0.0;
    for source in 0..candidate.source_count() {
        let candidate_crop = candidate.crop_indices.get(source)?;
        let reference_crop = reference.crop_indices.get(source)?;
        let candidate_offset = candidate.source_offset(source)?;
        let reference_offset = reference.source_offset(source)?;
        let correction = candidate_corrections.map_or((0.0, 0.0), |values| values[source]);
        let row = candidate_crop.start_row as f64 + candidate_offset.row + correction.0
            - reference_crop.start_row as f64
            - reference_offset.row;
        let column = candidate_crop.start_col as f64 + candidate_offset.column + correction.1
            - reference_crop.start_col as f64
            - reference_offset.column;
        squared += row * row + column * column;
    }
    Ok(IlluminationPositionMetrics {
        position_rmse: (squared / candidate.source_count() as f64).sqrt(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameGainComparisonMetrics {
    pub relative_error: f64,
}

pub fn compare_frame_gains(
    reference: &[f64],
    candidate: &[f64],
) -> Result<FrameGainComparisonMetrics> {
    if reference.len() != candidate.len()
        || reference.is_empty()
        || candidate
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::InvalidParameter {
            name: "candidate_frame_gains",
            reason: "must be finite, positive, and match reference length".into(),
        });
    }
    let numerator: f64 = candidate
        .iter()
        .zip(reference)
        .map(|(&candidate, &reference)| candidate * reference)
        .sum();
    let denominator: f64 = candidate.iter().map(|value| value * value).sum();
    let alignment = numerator / denominator.max(f64::EPSILON);
    let error: f64 = candidate
        .iter()
        .zip(reference)
        .map(|(&candidate, &reference)| (alignment * candidate - reference).powi(2))
        .sum();
    let reference_power: f64 = reference.iter().map(|value| value * value).sum();
    Ok(FrameGainComparisonMetrics {
        relative_error: (error / reference_power.max(f64::EPSILON)).sqrt(),
    })
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
