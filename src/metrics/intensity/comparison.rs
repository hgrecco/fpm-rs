//! Metrics comparing a reference and candidate intensity image.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntensityComparisonMetrics {
    /// Serialized as `measured_sum` inside diagnostic records for wire compatibility.
    #[serde(rename = "measured_sum", alias = "reference_sum")]
    pub reference_sum: f64,
    /// Serialized as `predicted_sum` inside diagnostic records for wire compatibility.
    #[serde(rename = "predicted_sum", alias = "candidate_sum")]
    pub candidate_sum: f64,
    pub residual_l1: f64,
    pub residual_l2: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
    pub residual_max_abs: f64,
    pub normalized_l2: f64,
    pub saturated_pixels: Option<usize>,
}

pub fn compare_intensity(
    reference: &[f64],
    candidate: &[f64],
    saturation_value: Option<f64>,
) -> crate::Result<IntensityComparisonMetrics> {
    compare_intensity_masked(reference, candidate, None, saturation_value)
}

pub fn compare_intensity_masked(
    reference: &[f64],
    candidate: &[f64],
    mask: Option<&[u8]>,
    saturation_value: Option<f64>,
) -> crate::Result<IntensityComparisonMetrics> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(crate::Error::InvalidShape(format!(
            "intensity comparison inputs have lengths {} and {}",
            reference.len(),
            candidate.len()
        )));
    }
    if mask.is_some_and(|mask| mask.len() != reference.len()) {
        return Err(crate::Error::LengthMismatch {
            actual: mask.map_or(0, <[u8]>::len),
            expected: reference.len(),
            shape: (1, reference.len()),
        });
    }
    let mut reference_sum = 0.0;
    let mut candidate_sum = 0.0;
    let mut residual_l1 = 0.0;
    let mut residual_squared = 0.0;
    let mut residual_sum = 0.0;
    let mut residual_max_abs: f64 = 0.0;
    let mut reference_squared = 0.0;
    let mut saturated_pixels = 0;
    let mut count = 0usize;
    for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
        if mask.is_some_and(|mask| mask[index] == 0) {
            continue;
        }
        if !reference.is_finite() || !candidate.is_finite() {
            return Err(crate::Error::Numerical(
                "intensity comparison contains a non-finite value".into(),
            ));
        }
        let residual = candidate - reference;
        reference_sum += reference;
        candidate_sum += candidate;
        residual_l1 += residual.abs();
        residual_squared += residual * residual;
        residual_sum += residual;
        residual_max_abs = residual_max_abs.max(residual.abs());
        reference_squared += reference * reference;
        saturated_pixels += usize::from(saturation_value.is_some_and(|limit| reference >= limit));
        count += 1;
    }
    let count_f64 = count as f64;
    let residual_mean = (count > 0)
        .then_some(residual_sum / count_f64)
        .unwrap_or(0.0);
    Ok(IntensityComparisonMetrics {
        reference_sum,
        candidate_sum,
        residual_l1,
        residual_l2: residual_squared.sqrt(),
        residual_mean,
        residual_std: if count == 0 {
            0.0
        } else {
            (residual_squared / count_f64 - residual_mean * residual_mean)
                .max(0.0)
                .sqrt()
        },
        residual_max_abs,
        normalized_l2: if count == 0 {
            0.0
        } else {
            residual_squared.sqrt() / (reference_squared.sqrt() + f64::EPSILON)
        },
        saturated_pixels: saturation_value.map(|_| saturated_pixels),
    })
}
