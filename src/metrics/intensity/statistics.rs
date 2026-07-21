//! Statistics calculated from one intensity image.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntensityStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub saturated_pixels: usize,
    pub zero_pixels: usize,
}

pub fn intensity_statistics(
    values: &[f64],
    saturation_value: Option<f64>,
) -> crate::Result<IntensityStatistics> {
    if values.is_empty() {
        return Err(crate::Error::InvalidShape(
            "intensity statistics require non-empty values".into(),
        ));
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut saturated_pixels = 0;
    let mut zero_pixels = 0;
    for &value in values {
        if !value.is_finite() {
            return Err(crate::Error::Numerical(
                "intensity statistics contain a non-finite value".into(),
            ));
        }
        sum += value;
        sum_squares += value * value;
        min = min.min(value);
        max = max.max(value);
        zero_pixels += usize::from(value == 0.0);
        saturated_pixels += usize::from(saturation_value.is_some_and(|limit| value >= limit));
    }
    let count = values.len() as f64;
    let mean = sum / count;
    Ok(IntensityStatistics {
        mean,
        std: (sum_squares / count - mean * mean).max(0.0).sqrt(),
        min,
        max,
        sum,
        saturated_pixels,
        zero_pixels,
    })
}
